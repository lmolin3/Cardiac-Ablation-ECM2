// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "monodomain_solver.hpp"
#include "general/forall.hpp"

using namespace mfem;
using namespace mfem::electrophysiology;

MonodomainDiffusionSolver::MonodomainDiffusionSolver(ParFiniteElementSpace *fes_,
                                                     BCHandler *bcs_,
                                                     MatrixCoefficient *sigma_coeff_,
                                                     Coefficient *chi_coeff_, Coefficient *Cm_coeff_,
                                                     int ode_solver_type,
                                                     bool verbose_)
    : TimeDependentOperator(fes_->GetTrueVSize(), 0.0),
      pmesh(fes_->GetParMesh()), fes(fes_), bcs(bcs_), pa(false), cached_dt(0.0),
      sigma_coeff(sigma_coeff_), verbose(verbose_)
{
   //<--- Check if the parameters are set
   if (!chi_coeff_ || !Cm_coeff_)
      mfem_error("Coefficients chi and Cm must be set");
   chi_Cm_coeff = std::make_unique<ProductCoefficient>(*chi_coeff_, *Cm_coeff_);

   //<--- Get the true size of the finite element space
   fes_truevsize = fes->GetTrueVSize();
   this->height = fes_truevsize;
   this->width = fes_truevsize;

   //<--- ParGridFunctions
   u_gf.SetSpace(fes);
   u_gf = 0.0;
   du_dt_gf.SetSpace(fes);
   du_dt_gf = 0.0;

   //<--- Vectors
   // Mark them as device vectors up front: these are only ever consumed by the PA
   // operators and the CG solver, and without the flag the first vector algebra on
   // them (z.Neg(), z.Add(), ...) would take the host path and force a migration.
   z.SetSize(fes_truevsize); z.UseDevice(true); z = 0.0;
   b.SetSize(fes_truevsize); b.UseDevice(true); b = 0.0;

   //<--- Initialize the ODEStateDataVector for previous solution
   //auto mem_type = GetMemoryType(this->GetMemoryClass());
   //int state_size = 2; // Store two previous time steps
   //u_prev = new ODEStateDataVector(state_size);
   //u_prev->SetSize(this->Width(), mem_type);
   //Vector tmp(fes_truevsize); tmp = 0.0;
   //u_prev->Append(tmp);

   //<--- Create ODESolver
   ode_solver = ODESolver::Select(ode_solver_type);
   implicit_time_integration = ode_solver_type > 20;
}

MonodomainDiffusionSolver::~MonodomainDiffusionSolver()
{
   delete bcs;

   if (Mfull)
   {
      delete Mfull;
   }
}


////////////////////////////////////////////////////////////////////////////
// ----- Setup API -----
////////////////////////////////////////////////////////////////////////////

void MonodomainDiffusionSolver::Setup(real_t dt, int prec_type_, real_t rel_tol_,
                                     bool warm_start_)
{
   cached_dt = dt;
   prec_type = prec_type_;
   lin_rel_tol = rel_tol_;
   warm_start = warm_start_;

   ///<--- Check partial assembly
   bool tensor = UsesTensorBasis(*fes);
   MFEM_VERIFY(!(pa && !tensor), "Partial assembly is only supported for tensor elements.");

   // NOTE: "full assembly" here means "assemble a matrix" (the -fa driver flag), as
   // opposed to matrix-free partial assembly. That is a different axis from MFEM's
   // AssemblyLevel::FULL vs ::LEGACY, which selects *how* that matrix is built and
   // is decided below.
   if (pmesh->GetMyRank() == 0 && verbose)
   {
      if (pa)
         out << "Using Partial Assembly (matrix-free). " << std::endl;
      else
         out << "Using assembled matrix. " << std::endl;
   }

   ///<--- Extract the list of essential BC degrees of freedom
   if ((bcs->GetDirichletDbcs()).size() > 0) 
   {
      fes->GetEssentialTrueDofs(bcs->GetDirichletAttr(), ess_tdof_list);
   }

   ///<--- Setup bilinear forms
   // Mass matrix
   M_form = std::make_unique<ParBilinearForm>(fes);
   auto *mass_integrator = new MassIntegrator(*chi_Cm_coeff);
   // A separate mass rule, when one was supplied, exists so that a collocated
   // Gauss-Lobatto rule can make the mass matrix diagonal (lumping).
   mass_integrator->SetIntRule(MassIntegrationRule());
   M_form->AddDomainIntegrator(mass_integrator);
   // Diffusion matrix
   K_form = std::make_unique<ParBilinearForm>(fes);
   auto *diff_integrator = new DiffusionIntegrator(*sigma_coeff);
   diff_integrator->SetIntRule(integration_rule);
   K_form->AddDomainIntegrator(diff_integrator);
   // Finalize (based on assembly level)
   if (pa)
   {
      M_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      K_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   else
   {
      // Non-PA path: choose between LEGACY (host) and FULL (device) assembly.
      //
      // FULL builds the same sparse matrix but fills it with device kernels via the
      // EA path, which is ~4x faster than LEGACY on GPU (measured 4.79 s -> 1.14 s
      // for 216k dofs) and *slower* on CPU, hence the device-backend condition.
      //
      // The symmetry condition is a correctness guard, not an optimisation.
      // DiffusionIntegrator::AssemblePA() sets `symmetric = (coeff_dim != dim*dim)`,
      // so a general MatrixCoefficient stores dim*dim entries per quadrature point
      // (bilininteg_diffusion_pa.cpp:136-137), while the EA kernels unconditionally
      // reshape pa_data as the symmetric dim*(dim+1)/2 layout -- the string
      // "symmetric" does not occur anywhere in bilininteg_diffusion_ea.cpp. With a
      // non-symmetric-typed coefficient the element matrices are built from misread
      // data, with no assert and no warning: measured ||u||_2 = 8.487e+03 against a
      // correct 1.259e+04, a ~33% error. So we only opt in when sigma is declared as
      // a SymmetricMatrixCoefficient, and fall back to LEGACY otherwise.
      //
      // This is not a restriction in practice: a conductivity tensor is symmetric by
      // Onsager reciprocity, and fiber-based orthotropy
      //     sigma = s_f (f x f) + s_s (s x s) + s_n (n x n)
      // is a sum of symmetric rank-one terms, hence symmetric. It must also be SPD
      // for the CG solvers used here to be valid at all.
      // The tensor condition is a second correctness guard. The EA kernels go
      // through FiniteElement::GetDofToQuad(ir, DofToQuad::FULL), which only
      // tensor-product bases implement; on simplices the base-class version
      // aborts with "invalid mode requested" (fem/fe/fe_base.cpp:379).
      const bool on_device = Device::Allows(Backend::DEVICE_MASK);
      const bool sym_sigma =
         dynamic_cast<SymmetricMatrixCoefficient *>(sigma_coeff) != nullptr;

      if (on_device && sym_sigma && tensor)
      {
         M_form->SetAssemblyLevel(AssemblyLevel::FULL);
         K_form->SetAssemblyLevel(AssemblyLevel::FULL);
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            out << "Using Full Assembly (device kernels). " << std::endl;
         }
      }
      else if (on_device && pmesh->GetMyRank() == 0 && verbose)
      {
         out << "Using Legacy Assembly (host): device-side full assembly needs ";
         if (!tensor)
         {
            out << "tensor-product\n  elements (quad/hex); the element-assembly "
                << "kernels have no simplex path." << std::endl;
         }
         else
         {
            out << "the conductivity\n  declared as a SymmetricMatrixCoefficient, "
                << "because DiffusionIntegrator's\n  element-assembly kernels assume "
                << "the symmetric quadrature-data layout." << std::endl;
         }
      }
   }

   // Assemble
   AssembleOperators();

   ///<--- Linear Solvers
   // Mass matrix
   if (pa)
   {
      M_prec = std::make_unique<OperatorJacobiSmoother>(*M_form, ess_tdof_list);
   }
   else
   {
      M_prec = std::make_unique<HypreSmoother>();
      static_cast<HypreSmoother *>(M_prec.get())->SetType(HypreSmoother::Jacobi); // See hypre.hpp for more options
   }
   M_solver = std::make_unique<CGSolver>(fes->GetComm());
   M_solver->iterative_mode = false; 
   M_solver->SetRelTol(solver_opts.rel_tol);
   M_solver->SetAbsTol(solver_opts.abs_tol);
   M_solver->SetMaxIter(solver_opts.max_iter);
   M_solver->SetPrintLevel(solver_opts.print_level);
   M_solver->SetPreconditioner(*M_prec);
   M_solver->SetOperator(*opM);

   // Solver for the implicit operator. Only built when an implicit ODE solver was
   // selected: BuildImplicitSolver() asserts on that flag, and on the explicit path
   // T = M + dt K is never formed, so building it would waste the assembly and, with
   // prec_type 1, an entire LOR discretization and AMG hierarchy that nothing uses.
   if (implicit_time_integration) { BuildImplicitSolver(); }

   /// 5. Assemble linear form for rhs
   fform = std::make_unique<ParLinearForm>(fes);

   // Adding neuman bcs
   for (auto &neumann_bc : bcs->GetNeumannBcs())
   {
      fform->AddBoundaryIntegrator(new BoundaryLFIntegrator(*(neumann_bc.coeff)), neumann_bc.attr);
   }

   //<--- Setup ODE solver
   ode_solver->Init(*this);

   assembled = true;
}

void MonodomainDiffusionSolver::Update()
{
   //<--- Update the space: recalculate the number of DOFs and construct a matrix
   // that will adjust any GridFunctions to the new mesh state.
   fes->Update();

   //<--- Interpolate the solution on the new mesh by applying the transformation
   // matrix computed in the finite element space.
   u_gf.Update();
   du_dt_gf.Update();

   //<--- Rebuild the bilinear forms 
   M_form->Update();
   K_form->Update();

   //<--- Update the linear form for the rhs (assembly will be done on next time step)
   fform->Update();

   //<--- Update size of vectors
   fes_truevsize = fes->GetTrueVSize();
   this->height = fes_truevsize;
   this->width = fes_truevsize;
   z.SetSize(fes_truevsize); z.UseDevice(true);
   b.SetSize(fes_truevsize); b.UseDevice(true);

   //<--- Update the ODE solver
   ode_solver->Init(*this);

   //<--- Flag that we need to reassemble the operators
   if (T_solver)
   {
      T_solver.reset(nullptr);
   }

   assembled = false;
}

inline void MonodomainDiffusionSolver::AssembleAndSetupSolver()
{
   AssembleOperators();

   //<--- Recreate the linear solver for the mass matrix
   if (pa)
   {
      M_prec = std::make_unique<OperatorJacobiSmoother>(*M_form, ess_tdof_list);
   }
   else
   {
      M_prec = std::make_unique<HypreSmoother>();
      static_cast<HypreSmoother *>(M_prec.get())->SetType(HypreSmoother::Jacobi); // See hypre.hpp for more options
   }
   M_solver = std::make_unique<CGSolver>(fes->GetComm());
   M_solver->iterative_mode = false;
   M_solver->SetRelTol(solver_opts.rel_tol);
   M_solver->SetAbsTol(solver_opts.abs_tol);
   M_solver->SetMaxIter(solver_opts.max_iter);
   M_solver->SetPrintLevel(solver_opts.print_level);
   M_solver->SetPreconditioner(*M_prec);
   M_solver->SetOperator(*opM);
}

void MonodomainDiffusionSolver::BuildImplicitSolver()
{
   MFEM_ASSERT(implicit_time_integration, "Solver is not using implicit time integration.");

   if (pa)
   {
      T_solver = std::make_unique<ImplicitSolverPA>(fes, cached_dt, bcs, ess_tdof_list, sigma_coeff, chi_Cm_coeff.get(), prec_type, lin_rel_tol, integration_rule, MassIntegrationRule());
   }
   else
   {
      T_solver = std::make_unique<ImplicitSolverFA>(ess_tdof_list, pmesh->Dimension(), cached_dt, Mfull, opK.As<HypreParMatrix>(), prec_type, lin_rel_tol);
   }
}

////////////////////////////////////////////////////////////////////////////
// ----- ODESolver API -----
////////////////////////////////////////////////////////////////////////////

void MonodomainDiffusionSolver::Step(Vector &x, real_t &t, real_t &dt, bool provisional)
{
   //<--- Set time
   this->SetTime(t+dt);

   //<--- Re-assemble if required
   if (!assembled)
   {
      AssembleAndSetupSolver();
      assembled = true;
   }

   //<--- Step
   ode_solver->Step(x, t, dt);

   //<--- Enforce essential boundary conditions again (avoid round-off)
   u_gf.SetFromTrueDofs(x);
   for (auto &ess_bc : bcs->GetDirichletDbcs())
   {
      u_gf.ProjectBdrCoefficient(*ess_bc.GetCoeff(0), ess_bc.attr);
   }
   u_gf.GetTrueDofs(x);

   //<--- Update time (if provisional, restore previous time and return)
   if (provisional)
   {
      t-=dt;
      return;
   }

   //<--- Update the time step history
   UpdateTimeStepHistory(x);
}

////////////////////////////////////////////////////////////////////////////
// ----- TimeDependentOperator API -----
////////////////////////////////////////////////////////////////////////////

void MonodomainDiffusionSolver::Mult(const Vector &u, Vector &du_dt) const
{
   // Compute:
   //    du_dt = M^{-1}*[-K(u) + bcs] = M^{-1}*[-K(u) + bcs]

   //<--- Compute the rhs
   opK->Mult(u, z); // z = K(u)
   z.Neg();         // z = -K(u)
   z.Add(1.0, b);  // z = -K(u) + f

   //<--- Apply bcs
   du_dt_gf = 0.0;
   for (auto &ess_bc : bcs->GetDirichletDbcs())   
   {
      du_dt_gf.ProjectBdrCoefficient(*ess_bc.GetCoeff(1), ess_bc.attr);
   }
   du_dt_gf.GetTrueDofs(du_dt);

   if (pa)
   {
      auto *MC = opM.As<ConstrainedOperator>();
      MC->EliminateRHS(du_dt, z);
   }
   else
   {
      opM.EliminateBC(opMe, ess_tdof_list, du_dt, z);
   }

   //<---  Solve
   M_solver->Mult(z, du_dt);

   //<---  Enforce essential boundary conditions again (avoid round-off)
   du_dt_gf.SetFromTrueDofs(du_dt);
   for (auto &ess_bc : bcs->GetDirichletDbcs())
   {
      du_dt_gf.ProjectBdrCoefficient(*ess_bc.GetCoeff(1), ess_bc.attr);
   }
   du_dt_gf.GetTrueDofs(du_dt);
}

void MonodomainDiffusionSolver::ImplicitSolve(const real_t dt, const Vector &u,
                                                       Vector &du_dt)
{
   // Solve the equation:
   //    M du_dt = [-K(u + dt*du_dt) + bcs]
   // (M + dt K) du_dt = -K(u) + f

   //<--- Update the implicit solver 
   SetTimeStep(dt);
   if (T_solver == nullptr)
   {
      BuildImplicitSolver();
   }

   //<--- Compute the rhs
   opK->Mult(u, z); // z = K_form(u)
   z.Neg();         // z = -K_form(u)
   z.Add(1.0, b);  // z = -K_form(u) + f

   //<--- Apply bcs
   du_dt_gf = 0.0;
   for (auto &ess_bc : bcs->GetDirichletDbcs())   
   {
      du_dt_gf.ProjectBdrCoefficient(*ess_bc.GetCoeff(1), ess_bc.attr);
   }
   du_dt_gf.GetTrueDofs(du_dt);
   T_solver->EliminateBC(du_dt, z);

   //<---  Solve. With warm starting the previous du/dt replaces the (zero) initial
   // guess; see ImplicitSolverBase::SetWarmStart for why this needs an absolute
   // stopping test. The first solve necessarily runs cold and calibrates it.
   if (warm_start && du_dt_prev.Size() == du_dt.Size())
   {
      du_dt = du_dt_prev;
      T_solver->EnableWarmStart(z);
   }
   else
   {
      T_solver->DisableWarmStart();
   }
   T_solver->Mult(z, du_dt);
   if (warm_start) { du_dt_prev = du_dt; du_dt_prev.UseDevice(true); }

   //<---  Enforce essential boundary conditions again (avoid round-off)
   du_dt_gf.SetFromTrueDofs(du_dt);
   for (auto &ess_bc : bcs->GetDirichletDbcs())
   {
      du_dt_gf.ProjectBdrCoefficient(*ess_bc.GetCoeff(1), ess_bc.attr);
   }
   du_dt_gf.GetTrueDofs(du_dt);
}


// NOTE: this can be optimized by  assembling only if required (e.g. any parameter is time dependent)
void MonodomainDiffusionSolver::SetTime(const real_t time)
{
   TimeDependentOperator::SetTime(time);

   // Update time for parameters, volumetric terms and bcs
   SetCoefficientsTime(time);
   bcs->SetTime(time);

   // Assemble rhs
   fform->Assemble();
   fform->ParallelAssemble(b);

   // Return if reassembling is not enabled (e.g. constant coefficients)
   if (enable_rebuild)
   {
      AssembleOperators();
   }
}

void MonodomainDiffusionSolver::SetCoefficientsTime(const real_t &time)
{
   // Set time for coefficients
   sigma_coeff->SetTime(time);
   chi_Cm_coeff->SetTime(time);
}

inline void MonodomainDiffusionSolver::SetTimeStep(const real_t dt)
{
   if (dt != cached_dt)
   {
      cached_dt = dt;
      T_solver = nullptr;
   }
}

inline void MonodomainDiffusionSolver::AssembleOperators()
{
   delete Mfull;
   Mfull = nullptr;

   // Re-assemble operators if needed
   int skip_zeros = 0;
   Array<int> empty;
   M_form->Update();
   M_form->Assemble(skip_zeros);
   if (pa)
   {
      M_form->FormSystemMatrix(ess_tdof_list, opM);
   }
   else
   {
      M_form->FormSystemMatrix(empty, opM);
      Mfull = new HypreParMatrix(*(opM.As<HypreParMatrix>()));
      opMe.EliminateRowsCols(opM, ess_tdof_list);
   }

   K_form->Update();
   K_form->Assemble(skip_zeros);
   K_form->FormSystemMatrix(empty, opK);

   // Delete the implicit solver
   T_solver.reset();
}

////////////////////////////////////////////////////////////////////////////
// ----- Other methods -----
////////////////////////////////////////////////////////////////////////////

void MonodomainDiffusionSolver::EnablePA(bool pa_) { pa = pa_; }

void MonodomainDiffusionSolver::UpdateTimeStepHistory(Vector &x)
{
   //u_prev->Append(x);
}
