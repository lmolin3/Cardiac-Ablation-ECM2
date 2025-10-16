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
#include "../../../general/forall.hpp"

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
   z.SetSize(fes_truevsize); z = 0.0;
   b.SetSize(fes_truevsize); b = 0.0;

   //<--- Initialize the ODEStateDataVector for previous solution
   auto mem_type = GetMemoryType(this->GetMemoryClass());
   int state_size = 2; // Store two previous time steps
   u_prev = new ODEStateDataVector(state_size);
   u_prev->SetSize(this->Width(), mem_type);
   Vector tmp(fes_truevsize); tmp = 0.0;
   u_prev->Append(tmp);

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

void MonodomainDiffusionSolver::Setup(real_t dt, int prec_type)
{
   cached_dt = dt;

   ///<--- Check partial assembly
   bool tensor = UsesTensorBasis(*fes);
   MFEM_VERIFY(!(pa && !tensor), "Partial assembly is only supported for tensor elements.");

   if (pmesh->GetMyRank() == 0 && verbose)
   {
      if (pa)
         out << "Using Partial Assembly. " << std::endl;
      else
         out << "Using Full Assembly. " << std::endl;
   }

   ///<--- Extract the list of essential BC degrees of freedom
   if ((bcs->GetDirichletDbcs()).size() > 0) 
   {
      fes->GetEssentialTrueDofs(bcs->GetDirichletAttr(), ess_tdof_list);
   }

   ///<--- Setup bilinear forms
   // Mass matrix
   M_form = std::make_unique<ParBilinearForm>(fes);
   M_form->AddDomainIntegrator(new MassIntegrator(*chi_Cm_coeff));
   // Diffusion matrix
   K_form = std::make_unique<ParBilinearForm>(fes);
   K_form->AddDomainIntegrator(new DiffusionIntegrator(*sigma_coeff));
   // Robin mass
   RobinMass_form = std::make_unique<ParBilinearForm>(fes);
   for (auto &robin_bc : bcs->GetRobinBcs())
   {
      RobinMass_form->AddBoundaryIntegrator(new MassIntegrator(*robin_bc.h_coeff), robin_bc.attr);
   }
   // Finalize (based on assembly level)
   if (pa)
   {
      M_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      K_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      RobinMass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   // Assemble
   Array<int> empty;
   int skip_zeros = 0;                   // keep sparsity pattern of M_form and K_form the same
   M_form->Assemble(skip_zeros);         
   K_form->Assemble(skip_zeros);         
   RobinMass_form->Assemble(skip_zeros); 
   K_form->FormSystemMatrix(empty, opK);
   RobinMass_form->FormSystemMatrix(empty, opRobinMass); 
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

   // Solver for implicit operator
   BuildImplicitSolver();

   /// 5. Assemble linear form for rhs
   fform = std::make_unique<ParLinearForm>(fes);

   // Adding neuman bcs
   for (auto &neumann_bc : bcs->GetNeumannBcs())
   {
      fform->AddBoundaryIntegrator(new BoundaryLFIntegrator(*(neumann_bc.coeff)), neumann_bc.attr);
   }

   // Adding robin bcs
   for (auto &robin_bc : bcs->GetRobinBcs())
   {
      fform->AddBoundaryIntegrator(new BoundaryLFIntegrator(*(robin_bc.hT0_coeff)), robin_bc.attr);
   }

   //<--- Setup ODE solver
   ode_solver->Init(*this);
}

void MonodomainDiffusionSolver::Update()
{
   int skip_zeros = 0;
   Array<int> empty;
   // Inform the bilinear forms that the space has changed and reassemble.
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

   if (bcs->GetRobinBcs().size() > 0)
   {
      RobinMass_form->Update(); 
   }

   // Inform the linear forms that the space has changed.
   fform->Update();


   //<--- Reset the implicit solver
   BuildImplicitSolver(); 
}

void MonodomainDiffusionSolver::BuildImplicitSolver()
{
   MFEM_ASSERT(implicit_time_integration, "Solver is not using implicit time integration.");

   if (pa)
   {
      T_solver = std::make_unique<ImplicitSolverPA>(fes, cached_dt, bcs, ess_tdof_list, sigma_coeff, chi_Cm_coeff.get());
   }
   else
   {
      T_solver = std::make_unique<ImplicitSolverFA>(ess_tdof_list, pmesh->Dimension(), cached_dt, Mfull, opK.As<HypreParMatrix>(), opRobinMass.As<HypreParMatrix>());
   }
}

////////////////////////////////////////////////////////////////////////////
// ----- ODESolver API -----
////////////////////////////////////////////////////////////////////////////

void MonodomainDiffusionSolver::Step(Vector &x, real_t &t, real_t &dt, bool provisional)
{
   //<--- Set time
   this->SetTime(t+dt);

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

   if (bcs->GetRobinBcs().size() > 0)
   {
      opRobinMass->AddMult(u, z, -1.0);
   }

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

   if (bcs->GetRobinBcs().size() > 0) // Mass matrix for Robin bc
   {
      opRobinMass->AddMult(u, z, -1.0);
   }

   //<--- Apply bcs
   du_dt_gf = 0.0;
   for (auto &ess_bc : bcs->GetDirichletDbcs())   
   {
      du_dt_gf.ProjectBdrCoefficient(*ess_bc.GetCoeff(1), ess_bc.attr);
   }
   du_dt_gf.GetTrueDofs(du_dt);
   T_solver->EliminateBC(du_dt, z);

   //<---  Solve
   T_solver->Mult(z, du_dt);

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

      // Assemble matrix for robin bcs
      if (bcs->GetRobinBcs().size() > 0)
      {
         RobinMass_form->Update();
         RobinMass_form->Assemble(skip_zeros);
         RobinMass_form->FormSystemMatrix(empty, opRobinMass);
      }

      // Delete the implicit solver 
      T_solver = nullptr;
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


////////////////////////////////////////////////////////////////////////////
// ----- Other methods -----
////////////////////////////////////////////////////////////////////////////

void MonodomainDiffusionSolver::EnablePA(bool pa_) { pa = pa_; }

void MonodomainDiffusionSolver::UpdateTimeStepHistory(Vector &x)
{
   u_prev->Append(x);
}
