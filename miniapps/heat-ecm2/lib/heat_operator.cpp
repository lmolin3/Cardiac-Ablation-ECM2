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
#include "../../general/forall.hpp"

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
                                                                             real_t beta_)
          : TimeDependentOperator(f.GetTrueVSize(), 0.0), pmesh(pmesh_), H1FESpace(f),
            bcs(bcs), Kappa(Kappa_), alpha(alpha_), u(u_), M(nullptr), RobinMass(nullptr), K(nullptr),
            Mfull(nullptr), M_solver(nullptr), T_solver(nullptr), M_prec(nullptr),
            fform(nullptr), fvec(nullptr), pa(false), current_dt(0.0)
      {
         // Check if the parameters are set
         if (!c_ || !rho_)
            mfem_error("Coefficients c and rho must be set");
         rhoC = new ProductCoefficient(*c_, *rho_);

         // Check contributions
         has_reaction = beta_ != 0.0;
         has_diffusion = Kappa ? true : false;
         has_advection = alpha_ != 0 && u ? true : false;
         if (!has_diffusion && !has_advection && !has_reaction)
         {
            mfem_error("At least one of the coefficients (diffusion, advection, reaction) must be set");
         }

         // Set reaction term coefficient -> set to negative because we assemble Advection-Diffusion as (D+A) and change sign after multiplying
         beta = has_reaction ? new ConstantCoefficient(-beta_) : nullptr;

         // Get the true size of the finite element space
         fes_truevsize = H1FESpace.GetTrueVSize();

         // Create the ParGridFunction and Vector for Temperature
         dT_approx = new Vector(fes_truevsize);
         *dT_approx = 0.0;
         z.SetSize(fes_truevsize);
         z = 0.0;
      }

      void AdvectionReactionDiffusionOperator::EnablePA(bool pa_) { pa = pa_; }

      void AdvectionReactionDiffusionOperator::Setup()
      {
         /// 1. Check partial assembly
         bool tensor = UsesTensorBasis(H1FESpace);

         if (pmesh->GetMyRank() == 0)
         {
            if (pa && tensor)
            {
               cout << "Using Partial Assembly. " << endl;
            }
            else if (pa && !tensor)
            {
               mfem_error(
                   "Partial assembly is only supported for tensor elements...Aborting");
            }
            else
            {
               cout << "Using Full Assembly. " << endl;
            }
         }

         /// 2. Extract the list of essential BC degrees of freedom
         if ((bcs->GetDirichletDbcs()).size() > 0) // Applied temperature
         {
            H1FESpace.GetEssentialTrueDofs(bcs->GetDirichletAttr(), ess_tdof_list);
         }

         /// 3. Setup bilinear forms
         // Mass matrix
         M = new ParBilinearForm(&H1FESpace);
         M->AddDomainIntegrator(new MassIntegrator(*rhoC));
         // Advection-Reaction-Diffusion matrix   K = ∇•(k∇) - α • u ∇ + β = -(D + A - R)
         K = new ParBilinearForm(&H1FESpace);
         if (has_diffusion)
            K->AddDomainIntegrator(new DiffusionIntegrator(*Kappa));
         if (has_advection)
            K->AddDomainIntegrator(new ConvectionIntegrator(*u, alpha));
         if (has_reaction)
            K->AddDomainIntegrator(new MassIntegrator(*beta));
         // Robin mass
         RobinMass = new ParBilinearForm(&H1FESpace);
         for (auto &robin_bc : bcs->GetRobinBcs())
         {
            // Add a Mass integrator on the Robin boundary
            RobinMass->AddBoundaryIntegrator(new MassIntegrator(*robin_bc.h_coeff),
                                             robin_bc.attr);
         }

         // Finalize (based on assembly level)
         if (pa)
         {
            M->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            K->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            RobinMass->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         }

         Array<int> empty;
         int skip_zeros = 0;
         M->Assemble(skip_zeros); // keep sparsity pattern of M and K the same
         K->Assemble(skip_zeros); // keep sparsity pattern of M and K the same
         // RobinMass->Assemble(skip_zeros); // Done in SetTime

         K->FormSystemMatrix(empty, opK);
         // RobinMass->FormSystemMatrix(empty, opRobinMass); // Done in SetTime
         if (pa)
         {
            M->FormSystemMatrix(ess_tdof_list, opM);
         }
         else
         {
            M->FormSystemMatrix(ess_tdof_list, opM);
            opMe = M->GetEliminatedOperator();
         }

         /// 4. Solvers

         // Solver for mass matrix
         if (pa)
         {
            M_prec = new OperatorJacobiSmoother(*M, ess_tdof_list);
            T_solver = nullptr; // Implicit time integration not supported with partial
                                // assembly
         }
         else
         {
            M_prec = new HypreSmoother();
            static_cast<HypreSmoother *>(M_prec)->SetType(
                HypreSmoother::Jacobi); // See hypre.hpp for more options

            // Solver for implicit solve (T_prec will be set in ImplicitSolve)
            Mfull =
                Add(1.0, *opM.As<HypreParMatrix>(), 1.0, *opMe.As<HypreParMatrix>());
            T_solver =
                new ImplicitSolver(Mfull, opK.As<HypreParMatrix>(), ess_tdof_list);
         }

         const double rel_tol = 1e-8;
         M_solver = CGSolver(H1FESpace.GetComm());
         M_solver.iterative_mode = false;
         M_solver.SetRelTol(rel_tol);
         M_solver.SetAbsTol(0.0);
         M_solver.SetMaxIter(1000);
         M_solver.SetPrintLevel(0);
         M_solver.SetPreconditioner(*M_prec);
         M_solver.SetOperator(*opM);

         /// 5. Assemble linear form for rhs
         fform = new ParLinearForm(&H1FESpace);

         // Adding Volumetric heat terms
         for (auto &volumetric_term : volumetric_terms)
         {
            fform->AddDomainIntegrator(new DomainLFIntegrator(*(volumetric_term.coeff)),
                                       volumetric_term.attr);
         }

         // Adding neuman bcs
         for (auto &neumann_bc : bcs->GetNeumannBcs())
         {
            fform->AddBoundaryIntegrator(new BoundaryLFIntegrator(*(neumann_bc.coeff)),
                                         neumann_bc.attr);
         }

         // Adding neuman vector bcs
         for (auto &neumann_vec_bc : bcs->GetNeumannVectorBcs())
         {
            fform->AddBoundaryIntegrator(
                new BoundaryNormalLFIntegrator(*(neumann_vec_bc.coeff)),
                neumann_vec_bc.attr);
         }

         // Adding robin bcs
         for (auto &robin_bc : bcs->GetRobinBcs())
         {
            fform->AddBoundaryIntegrator(
                new BoundaryLFIntegrator(*(robin_bc.hT0_coeff)), robin_bc.attr);
         }

         // Will be assembled in AdvectionReactionDiffusionOperator::UpdateBcsRhs
      }

      void AdvectionReactionDiffusionOperator::Update()
      {
         Array<int> empty;
         // Inform the bilinear forms that the space has changed and reassemble.
         M->Update();
         M->Assemble(0);

         if (pa)
         {
            M->FormSystemMatrix(ess_tdof_list, opM);
         }
         else
         {
            M->FormSystemMatrix(ess_tdof_list, opM);
            opMe = M->GetEliminatedOperator();
            delete Mfull;
            Mfull = nullptr;
            Mfull =
                Add(1.0, *opM.As<HypreParMatrix>(), 1.0, *opMe.As<HypreParMatrix>());
         }

         K->Update();
         K->Assemble(0);
         K->FormSystemMatrix(empty, opK);

         if (bcs->GetRobinBcs().size() > 0)
         {
            RobinMass->Update();
         }

         // Inform the linear forms that the space has changed.
         fform->Update();

         // Reset the ImplicitSolver
         T_solver->Reset();
      }

      void AdvectionReactionDiffusionOperator::SetTime(const double time)
      {
         TimeDependentOperator::SetTime(time);

         // Update the advection coefficient and re-assemble the stiffness matrix
         if (has_advection)
         {
            u->SetTime(time);
            K->Update();
            K->Assemble(0);
            Array<int> empty;
            K->FormSystemMatrix(empty, opK);
         }

         // Update the time dependent boundary conditions
         bcs->SetTime(time);

         // Volumetric heat terms
         for (auto &volumetric_term : volumetric_terms)
         {
            volumetric_term.coeff->SetTime(time);
         }

         // Assemble matrix for robin bcs
         if (bcs->GetRobinBcs().size() > 0)
         {
            Array<int> empty;
            RobinMass->Update();
            RobinMass->Assemble(0);
            RobinMass->FormSystemMatrix(empty, opRobinMass);
         }

         // Assemble rhs
         delete fvec;
         fvec = nullptr;
         fform->Assemble();
         fvec = fform->ParallelAssemble();
      }

      void AdvectionReactionDiffusionOperator::ProjectDirichletBCS(const double &time,
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
         opK->Mult(u, z); // z = K(u)
         z.Neg();         // z = -K(u)

         if (fvec) // Neumann + Robin + Volumetric heat
         {
            z.Add(1.0, *fvec); // z = -K(u) + f
         }

         if (bcs->GetRobinBcs().size() > 0) // Mass matrix for Robin bc
         {
            opRobinMass->AddMult(u, z, -1.0);
         }

         // Set bcs (set du_dt on dofs with approximation)
         du_dt = *dT_approx;

         if (pa)
         {
            auto *MC = opM.As<ConstrainedOperator>();
            MC->EliminateRHS(du_dt, z);
         }
         else
         {
            M->EliminateVDofsInRHS(ess_tdof_list, du_dt, z);
         }

         // Solve
         M_solver.Mult(z, du_dt);

         // set du_dt on dofs with approximation (avoid roundoff errors)
         for (int i = 0; i < ess_tdof_list.Size(); i++)
         {
            int dof = ess_tdof_list[i];
            du_dt[dof] = (*dT_approx)[dof];
         }
      }

      void AdvectionReactionDiffusionOperator::ImplicitSolve(const double dt, const Vector &u,
                                                             Vector &du_dt)
      {
         MFEM_VERIFY(!pa,
                     "Partial assembly not supported for implicit time integraton");

         // Solve the equation:
         //    du_dt = M^{-1}*[-K(u + f + dt*du_dt)]
         if (!T_solver->IsFinalized())
         {
            T_solver->BuildOperator(Mfull, opK.As<HypreParMatrix>(),
                                    opRobinMass.As<HypreParMatrix>());
         }

         // MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt
         opK->Mult(u, z); // z = K(u)
         z.Neg();         // z = -K(u)

         if (fvec) // Neumann + Robin + Volumetric heat
         {
            z.Add(1.0, *fvec); // z = -K(u) + f
         }

         if (bcs->GetRobinBcs().size() > 0) // Mass matrix for Robin bc
         {
            opRobinMass->AddMult(u, z, -1.0);
         }

         // Set bcs (set du_dt on dofs with approximation)
         du_dt = *dT_approx;

         // Eliminate essential dofs
         T_solver->EliminateBC(du_dt, z);

         // Solve
         T_solver->Mult(z, du_dt);

         // set du_dt on dofs with approximation (avoid roundoff errors)
         for (int i = 0; i < ess_tdof_list.Size(); i++)
         {
            int dof = ess_tdof_list[i];
            du_dt[dof] = (*dT_approx)[dof];
         }
      }

      void AdvectionReactionDiffusionOperator::SetTimeStep(double dt)
      {
         if (dt != current_dt && T_solver)
         {
            T_solver->SetTimeStep(dt); // This also resets the solver internally to recompute the operator
         }
         current_dt = dt;
      }

      void AdvectionReactionDiffusionOperator::AddVolumetricTerm(Coefficient *coeff,
                                                                 Array<int> &attr)
      {
         volumetric_terms.emplace_back(attr, coeff);
      }

      void AdvectionReactionDiffusionOperator::AddVolumetricTerm(ScalarFuncT func, Array<int> &attr)
      {
         AddVolumetricTerm(new FunctionCoefficient(func), attr);
      }

      AdvectionReactionDiffusionOperator::~AdvectionReactionDiffusionOperator()
      {
         delete bcs;

         delete RobinMass;
         delete M;
         delete K;
         delete Mfull;
         delete fform;

         delete T_solver;
         delete M_prec;

         delete fvec;

         delete dT_approx;

         delete rhoC;
      }

      // Class for solver used in implicit time integration
      ImplicitSolver::ImplicitSolver(HypreParMatrix *M_, HypreParMatrix *K_,
                                     Array<int> &ess_tdof_list_)
          : M(M_), K(K_), RobinMassMat(nullptr), T(nullptr), Te(nullptr),
            current_dt(-1.0), ess_tdof_list(ess_tdof_list_), finalized(false)
      {
         prec = new HypreSmoother();
         prec->SetType(HypreSmoother::Jacobi); // See hypre.hpp for more options
         linear_solver = new CGSolver(M->GetComm());
         linear_solver->iterative_mode = false;
         linear_solver->SetRelTol(1e-8);
         linear_solver->SetAbsTol(0.0);
         linear_solver->SetMaxIter(500);
         linear_solver->SetPrintLevel(0);
         linear_solver->SetPreconditioner(*prec);
      };

      void ImplicitSolver::SetTimeStep(double dt_)
      {
         if (dt_ != current_dt)
         {
            current_dt = dt_;
            this->Reset();
         }
      }

      void ImplicitSolver::SetOperator(const Operator &op)
      {
         linear_solver->SetOperator(op);
      }

      void ImplicitSolver::Mult(const Vector &x, Vector &y) const
      {
         linear_solver->Mult(x, y);
      }

      void ImplicitSolver::Reset()
      {
         delete T;
         T = nullptr;
         delete Te;
         Te = nullptr;
         finalized = false;
      }

      bool ImplicitSolver::IsFinalized() const { return finalized; }

      void ImplicitSolver::BuildOperator(HypreParMatrix *M_, HypreParMatrix *K_,
                                         HypreParMatrix *RobinMass_)
      {
         // T = M + dt*K + dt RobinMass = M + dt*(D + A - R) + dt RobinMass
         RobinMassMat = RobinMass_;
         M = M_;
         K = K_;

         T = new HypreParMatrix(*M);
         if (RobinMassMat)
         {
            auto tmp = ParAdd(K, RobinMassMat);
            T->Add(current_dt, *tmp);
            delete tmp;
            tmp = nullptr;
         }
         else
         {
            T->Add(current_dt, *K);
         }

         Te = T->EliminateRowsCols(ess_tdof_list);
         linear_solver->SetOperator(*T);

         finalized = true;
      }

      void ImplicitSolver::EliminateBC(const Vector &x, Vector &b) const
      {
         if (finalized)
         {
            T->EliminateBC(*Te, ess_tdof_list, x, b);
         }
         else
         {
            MFEM_ABORT("ImplicitSolver not finalized");
         }
      }

      ImplicitSolver::~ImplicitSolver()
      {
         delete linear_solver;
         linear_solver = nullptr;
         delete prec;
         prec = nullptr;
         delete T;
         T = nullptr;
         delete Te;
         Te = nullptr;
      }

   } // namespace heat

} // namespace mfem

#endif // MFEM_USE_MPI
