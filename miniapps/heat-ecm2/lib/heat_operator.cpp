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

#include "heat_operator.hpp"
#include "../../../general/forall.hpp"

#ifdef MFEM_USE_MPI

using namespace std;
namespace mfem
{

   using namespace common;

   namespace heat
   {

      AdvectionReactionDiffusionOperator::AdvectionReactionDiffusionOperator(std::shared_ptr<ParMesh> pmesh_,
                                                                             ParFiniteElementSpace &f, BCHandler *bcs,
                                                                             MatrixCoefficient *Kappa_,
                                                                             Coefficient *c_, Coefficient *rho_,
                                                                             real_t alpha_, VectorCoefficient *u_,
                                                                             real_t beta_,
                                                                             bool verbose_)
          : TimeDependentOperator(f.GetTrueVSize(), 0.0), pmesh(pmesh_), H1FESpace(f),
            bcs(bcs), Kappa(Kappa_), alpha(alpha_), u(u_), 
            fform(nullptr), pa(false), cached_dt(0.0), verbose(verbose_)
      {
         ///<--- Check if the parameters are set
         if (!c_ || !rho_)
            mfem_error("Coefficients c and rho must be set");
         rhoC = new ProductCoefficient(*c_, *rho_);

         ///<--- Check contributions
         has_reaction = beta_ != 0.0;
         has_diffusion = Kappa ? true : false;
         has_advection = alpha_ != 0 && u ? true : false;
         if (!has_diffusion && !has_advection && !has_reaction)
         {
            mfem_error("At least one of the coefficients (diffusion, advection, reaction) must be set");
         }

         ///<--- Set reaction term coefficient
         beta = has_reaction ? new ConstantCoefficient(beta_) : nullptr;

         ///<--- Define convection coefficient (rho c u)
         conv_coeff = has_advection ? new ScalarVectorProductCoefficient(*rhoC, *u) : nullptr;

         ///<--- Get the true size of the finite element space
         fes_truevsize = H1FESpace.GetTrueVSize();

         ///<--- Create the ParGridFunction and Vector for Temperature
         dT_approx = new Vector(fes_truevsize);
         *dT_approx = 0.0;
         z.SetSize(fes_truevsize);
         z = 0.0;
         fvec.SetSize(fes_truevsize);
         fvec = 0.0;
      }

      void AdvectionReactionDiffusionOperator::EnablePA(bool pa_) { pa = pa_; }

      void AdvectionReactionDiffusionOperator::Setup(real_t dt, bool implicit_time_integration_, int prec_type_)
      {
         implicit_time_integration = implicit_time_integration_;
         prec_type = prec_type_;
         cached_dt = dt;

         ///<--- 1. Check partial assembly
         bool tensor = UsesTensorBasis(H1FESpace);
         MFEM_VERIFY(!(pa && !tensor), "Partial assembly is only supported for tensor elements.");

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            if (pa)
               out << "Using Partial Assembly. " << endl;
            else
               out << "Using Full Assembly. " << endl;
         }

         ///<--- 2. Extract the list of essential BC degrees of freedom
         if ((bcs->GetDirichletDbcs()).size() > 0) // Applied temperature
         {
            H1FESpace.GetEssentialTrueDofs(bcs->GetDirichletAttr(), ess_tdof_list);
         }

         ///<--- 3. Setup bilinear forms
         // Mass matrix
         M_form = std::make_unique<ParBilinearForm>(&H1FESpace);
         M_form->AddDomainIntegrator(new MassIntegrator(*rhoC));
         // Advection-Reaction-Diffusion matrix   K = ∇•(k∇) - α • u ∇ - β = -(D + A + R)
         K_form = std::make_unique<ParBilinearForm>(&H1FESpace);
         if (has_diffusion)
            K_form->AddDomainIntegrator(new DiffusionIntegrator(*Kappa));
         if (has_advection)
            K_form->AddDomainIntegrator(new ConvectionIntegrator(*conv_coeff, alpha));
         if (has_reaction)
            K_form->AddDomainIntegrator(new MassIntegrator(*beta)); // beta > 0 consumes heat
         // Robin mass
         RobinMass_form = std::make_unique<ParBilinearForm>(&H1FESpace);
         for (auto &robin_bc : bcs->GetRobinBcs())
         {
            // Add a Mass integrator on the Robin boundary
            RobinMass_form->AddBoundaryIntegrator(new MassIntegrator(*robin_bc.h_coeff), robin_bc.attr);
         }

         for (auto &robin_bc : bcs->GetGeneralRobinBcs())
         {
            // Add a Mass integrator on the Robin boundary
            RobinMass_form->AddBoundaryIntegrator(new MassIntegrator(*robin_bc.alpha1), robin_bc.attr);
         }

         // Finalize (based on assembly level)
         if (pa)
         {
            M_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            K_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            RobinMass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         }

         ///<--- 4. Assemble and create Solvers
         AssembleAndSetupSolver();

         // Solver for implicit operator
         if (implicit_time_integration)
         {
            BuildImplicitSolver();
         }

         ///<--- 5. Assemble linear form for rhs
         fform = std::make_unique<ParLinearForm>(&H1FESpace);

         // Adding Volumetric heat terms
         for (auto &volumetric_term : volumetric_terms)
         {
            fform->AddDomainIntegrator(new DomainLFIntegrator(*(volumetric_term.coeff)),volumetric_term.attr);
         }

         // Adding neuman bcs
         for (auto &neumann_bc : bcs->GetNeumannBcs())
         {
            fform->AddBoundaryIntegrator(new BoundaryLFIntegrator(*(neumann_bc.coeff)),neumann_bc.attr);
         }

         // Adding neuman vector bcs
         for (auto &neumann_vec_bc : bcs->GetNeumannVectorBcs())
         {
            fform->AddBoundaryIntegrator(
                new BoundaryNormalLFIntegrator(*(neumann_vec_bc.coeff)),neumann_vec_bc.attr);
         }

         // Adding robin bcs
         for (auto &robin_bc : bcs->GetRobinBcs())
         {
            fform->AddBoundaryIntegrator(new BoundaryLFIntegrator(*(robin_bc.hT0_coeff)), robin_bc.attr);
         }

         for (auto &robin_bc : bcs->GetGeneralRobinBcs())
         {
            fform->AddBoundaryIntegrator(new BoundaryLFIntegrator(*(robin_bc.alpha2_u2)), robin_bc.attr);         // alpha2 * u
            fform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*(robin_bc.mu2_grad_u2)), robin_bc.attr); // mu2 * grad(u2)
         }

         // Will be assembled in AdvectionReactionDiffusionOperator::SetTime

         ///<--- Set Assembled flag
         assembled = true;
      }


      void AdvectionReactionDiffusionOperator::Rebuild()
      {
         if (!assembled)
         {
            AssembleAndSetupSolver();
         }
      }


      void AdvectionReactionDiffusionOperator::Update()
      {
         int skip_zeros = 0;
         Array<int> empty;

         ///<--- Prepare linear/bilinear forms for re-assembly
         M_form->Update();
         K_form->Update();
         if (bcs->GetRobinBcs().size() > 0 || bcs->GetGeneralRobinBcs().size() > 0)
         {
            RobinMass_form->Update(); // TODO: Should i also re-assemble it?
         }

         fform->Update();

         //<--- Flag that we need to reassemble the operators
         if (T_solver)
         {
            T_solver.reset(nullptr);
         }

         assembled = false;
      }

      inline void AdvectionReactionDiffusionOperator::AssembleOperators()
      {
         // Prevent re-assembly when not explicitly flagged
         if (assembled)
            return;

         delete Mfull;
         Mfull = nullptr;

         // Re-assemble operators
         int skip_zeros = 0;
         Array<int> empty;
         //M_form->Update();
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

         //K_form->Update();
         K_form->Assemble(skip_zeros);
         K_form->FormSystemMatrix(empty, opK);

         if (bcs->GetRobinBcs().size() > 0 || bcs->GetGeneralRobinBcs().size() > 0)
         {
            //RobinMass_form->Update();
            RobinMass_form->Assemble(skip_zeros);
            RobinMass_form->FormSystemMatrix(empty, opRobinMass);
         }

         // Delete the implicit solver
         T_solver.reset();

         assembled = true;
      }

      inline void AdvectionReactionDiffusionOperator::AssembleAndSetupSolver()
      {
         ///<--- Assemble operators
         AssembleOperators();

         ///<--- Recreate the linear solvers (implicit solver will be created in ImplicitSolve if needed )
         // Solver for mass matrix
         if (pa)
         {
            M_prec = std::make_unique<OperatorJacobiSmoother>(*M_form, ess_tdof_list);
         }
         else
         {
            M_prec = std::make_unique<HypreSmoother>();
            static_cast<HypreSmoother *>(M_prec.get())->SetType(HypreSmoother::Jacobi); // See hypre.hpp for more options
         }

         const real_t rel_tol = 1e-8;
         M_solver = std::make_unique<CGSolver>(H1FESpace.GetComm());
         M_solver->iterative_mode = false;
         M_solver->SetRelTol(rel_tol);
         M_solver->SetAbsTol(0.0);
         M_solver->SetMaxIter(1000);
         M_solver->SetPrintLevel(0);
         M_solver->SetPreconditioner(*M_prec);
         M_solver->SetOperator(*opM);
      }

      inline void AdvectionReactionDiffusionOperator::BuildImplicitSolver()
      {
         MFEM_ASSERT(implicit_time_integration, "Solver is not using implicit time integration.");

         if (pa)
         {
            T_solver = std::make_unique<ImplicitSolverPA>(&H1FESpace, cached_dt, bcs, ess_tdof_list, Kappa, rhoC, alpha, u, beta, prec_type);
         }
         else
         {
            T_solver = std::make_unique<ImplicitSolverFA>(ess_tdof_list, pmesh->Dimension(), has_advection, cached_dt, Mfull, opK.As<HypreParMatrix>(), opRobinMass.As<HypreParMatrix>());
         }
      }

      void AdvectionReactionDiffusionOperator::SetTime(const real_t time)
      {
         TimeDependentOperator::SetTime(time);

         // Update time for parameters, volumetric terms and bcs
         SetCoefficientsTime(time);
         bcs->SetTime(time);
         
         for (auto &volumetric_term : volumetric_terms)
         {
            volumetric_term.coeff->SetTime(time);
         }

         // Assemble rhs
         fform->Assemble();
         fform->ParallelAssemble(fvec);

         // Re-assemble operators and create solvers if enabled
         if (enable_rebuild)
         { 
            this->Update();
            AssembleAndSetupSolver();
            BuildImplicitSolver();
         }
      }

      void AdvectionReactionDiffusionOperator::ProjectDirichletBCS(const real_t &time,
                                                                   ParGridFunction &gf)
      {
         // Projection of coeffs for dirichlet bcs
         for (auto &dbc : bcs->GetDirichletDbcs())
         {
            dbc.coeff->SetTime(time); // Set coefficient time to selected time
            gf.ProjectBdrCoefficient(*dbc.coeff, dbc.attr);
            dbc.coeff->SetTime(this->t); // Set coefficient time back to current time
                                         // stored in the operator
         }
      }

      void AdvectionReactionDiffusionOperator::Mult(const Vector &u, Vector &du_dt) const
      {
         // Compute:
         //    du_dt = M^{-1}*(-K(u) + f + Neumann + Robin) = M^{-1}*(-D(u) - A(u) + R(u) + f + Neumann + Robin)
  
         //<--- Compute the rhs
         opK->Mult(u, z); // z = K(u)
         z.Neg();         // z = -K(u)
         z.Add(1.0, fvec); // z = -K(u) + f     Neumann + Robin + Volumetric heat

         if (bcs->GetRobinBcs().size() > 0 || bcs->GetGeneralRobinBcs().size() > 0) // Mass matrix for Robin bc
         {
            opRobinMass->AddMult(u, z, -1.0);
         }

         //<--- Apply bcs
         // Here we assume Dirichlet is prescribed without an analytical function for du/dt
         // In case bcs where du/dt is known analytically, we can add them like in 
         // electrophysiology branch 
         du_dt = *dT_approx;

         if (pa)
         {
            auto *MC = opM.As<ConstrainedOperator>();
            MC->EliminateRHS(du_dt, z);
         }
         else
         {
            opM.EliminateBC(opMe, ess_tdof_list, du_dt, z);
            //M_form->EliminateVDofsInRHS(ess_tdof_list, du_dt, z);  // TODO: CHANGE, there's no bcs in M as it's assembled with empty 
         }

         //<---  Solve
         M_solver->Mult(z, du_dt);

         //<---  Enforce essential boundary conditions again (avoid round-off)
         for (int i = 0; i < ess_tdof_list.Size(); i++)
         {
            int dof = ess_tdof_list[i];
            du_dt[dof] = (*dT_approx)[dof];
         }
      }

      void AdvectionReactionDiffusionOperator::ImplicitSolve(const real_t dt, const Vector &u,
                                                             Vector &du_dt)
      {
         // Solve the equation:
         //    du_dt = M^{-1}*[-K(u + f + dt*du_dt)]
         
         //<--- Update the implicit solver 
         SetTimeStep(dt);
         if (T_solver == nullptr)
         {
            BuildImplicitSolver();
         }

         //<--- Compute the rhs
         opK->Mult(u, z); // z = K(u)
         z.Neg();         // z = -K(u)
         z.Add(1.0, fvec); // z = -K(u) + f     Neumann + Robin + Volumetric heat

         if (bcs->GetRobinBcs().size() > 0 || bcs->GetGeneralRobinBcs().size() > 0) // Mass matrix for Robin bc
         {
            opRobinMass->AddMult(u, z, -1.0);
         }

         //<--- Apply bcs
         // Here we assume Dirichlet is prescribed without an analytical function for du/dt
         // In case bcs where du/dt is known analytically, we can add them like in 
         // electrophysiology branch 
         du_dt = *dT_approx;

         // Eliminate essential dofs
         T_solver->EliminateBC(du_dt, z);

         //<---  Solve
         T_solver->Mult(z, du_dt);

         //<---  Enforce essential boundary conditions again (avoid round-off)
         for (int i = 0; i < ess_tdof_list.Size(); i++)
         {
            int dof = ess_tdof_list[i];
            du_dt[dof] = (*dT_approx)[dof];
         }
      }

      inline void AdvectionReactionDiffusionOperator::SetTimeStep(const real_t dt)
      {
         if (dt != cached_dt)
         {
            cached_dt = dt;
            T_solver = nullptr;
         }
      }

      void AdvectionReactionDiffusionOperator::SetCoefficientsTime(const real_t &time)
      {
         // Set time for coefficients
         rhoC->SetTime(time);

         if (has_diffusion)
            Kappa->SetTime(time);
         if (has_advection)
            u->SetTime(time);
         if (has_reaction)
            beta->SetTime(time);
      }

      void AdvectionReactionDiffusionOperator::AddVolumetricTerm(Coefficient *coeff,
                                                                 Array<int> &attr,
                                                                 bool own)
      {
         volumetric_terms.emplace_back(attr, coeff, own);
      }

      void AdvectionReactionDiffusionOperator::AddVolumetricTerm(ScalarFuncT func, Array<int> &attr, bool own)
      {
         AddVolumetricTerm(new FunctionCoefficient(func), attr, own);
      }

      AdvectionReactionDiffusionOperator::~AdvectionReactionDiffusionOperator()
      {
         delete bcs;

         if (Mfull)
         {
            delete Mfull;
         }

         delete dT_approx;

         delete rhoC;
         delete conv_coeff;
      }

   } // namespace heat

} // namespace mfem

#endif // MFEM_USE_MPI
