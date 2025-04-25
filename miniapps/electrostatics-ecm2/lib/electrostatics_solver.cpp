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

#include "electrostatics_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;
namespace mfem
{

   using namespace common;

   namespace electrostatics
   {

      ElectrostaticsSolver::ElectrostaticsSolver(std::shared_ptr<ParMesh> pmesh_, int order_,
                                                 BCHandler *bcs,
                                                 Coefficient *Sigma_,
                                                 bool verbose_)
          : order(order_),
            bcs(bcs),
            pmesh(pmesh_),
            visit_dc(nullptr),
            paraview_dc(nullptr),
            H1_fes(nullptr),
            HCurl_fes(nullptr),
            divEpsGrad(nullptr),
            SigmaMass(nullptr),
            rhs_form(nullptr),
            prec(nullptr),
            SigmaQ(Sigma_), // Must be deleted outside solver
            SigmaMQ(nullptr),
            verbose(verbose_)
      {
         // Initialize MPI variables
         const int dim = pmesh->Dimension();

         // Define compatible parallel finite element spaces on the parallel
         // mesh. Here we use arbitrary order H1 for potential and ND for the electric field.
         H1_fec = new H1_FECollection(order, dim);
         L2_fec = new L2_FECollection(order-1, dim);
         HCurl_fec = new ND_FECollection(order, dim);

         H1_fes = new ParFiniteElementSpace(pmesh.get(), H1_fec);
         HCurl_fes = new ParFiniteElementSpace(pmesh.get(), HCurl_fec);
         L2_fes = new ParFiniteElementSpace(pmesh.get(), L2_fec);

         // Discrete derivative operator
         grad = new ParDiscreteGradOperator(H1_fes, HCurl_fes);

         // Build grid functions
         phi = new ParGridFunction(H1_fes);
         *phi = 0.0;

         E = new ParGridFunction(HCurl_fes);
         *E = 0.0;

         // Define Joule Heating Coefficient
         w_coeff = new JouleHeatingCoefficient(SigmaQ, E);

         // Initialize vector/s
         B.UseDevice(true);
         B.SetSize(H1_fes->GetTrueVSize());
         Phi.UseDevice(true);
         Phi.SetSize(H1_fes->GetTrueVSize());

         tmp_domain_attr.SetSize(pmesh->attributes.Max());
      }

      ElectrostaticsSolver::ElectrostaticsSolver(std::shared_ptr<ParMesh> pmesh_, int order_,
                                                 BCHandler *bcs,
                                                 MatrixCoefficient *Sigma_,
                                                 bool verbose_)
          : order(order_),
            bcs(bcs),
            pmesh(pmesh_),
            visit_dc(nullptr),
            paraview_dc(nullptr),
            H1_fes(nullptr),
            HCurl_fes(nullptr),
            divEpsGrad(nullptr),
            SigmaMass(nullptr),
            rhs_form(nullptr),
            prec(nullptr),
            SigmaQ(nullptr), // Must be deleted outside solver
            SigmaMQ(Sigma_),
            verbose(verbose_)
      {
         // Initialize MPI variables
         const int dim = pmesh->Dimension();

         // Define compatible parallel finite element spaces on the parallel
         // mesh. Here we use arbitrary order H1 for potential and ND for the electric field.         
         H1_fec = new H1_FECollection(order, dim);
         L2_fec = new L2_FECollection(order-1, dim);
         HCurl_fec = new ND_FECollection(order, dim);

         H1_fes = new ParFiniteElementSpace(pmesh.get(), H1_fec);
         HCurl_fes = new ParFiniteElementSpace(pmesh.get(), HCurl_fec);
         L2_fes = new ParFiniteElementSpace(pmesh.get(), L2_fec);

         // Discrete derivative operator
         grad = new ParDiscreteGradOperator(H1_fes, HCurl_fes);

         // Build grid functions
         phi = new ParGridFunction(H1_fes);
         *phi = 0.0;

         E = new ParGridFunction(HCurl_fes);
         *E = 0.0;

         // Define Joule Heating Coefficient
         w_coeff = new JouleHeatingCoefficient(SigmaMQ, E);

         // Initialize vector/s
         B.UseDevice(true);
         B.SetSize(H1_fes->GetTrueVSize());
         Phi.UseDevice(true);
         Phi.SetSize(H1_fes->GetTrueVSize());

         tmp_domain_attr.SetSize(pmesh->attributes.Max()); 

      }


      ElectrostaticsSolver::~ElectrostaticsSolver()
      {
         delete phi;
         delete E;
         delete rhs_form;

         delete divEpsGrad;
         delete SigmaMass;

         delete grad;

         delete H1_fec;
         delete L2_fec;
         delete HCurl_fec;

         delete H1_fes;
         delete HCurl_fes;
         delete L2_fes;

         delete solver;
         delete prec;

         delete w_coeff;

         delete bcs; // Solver takes ownership of the BCHandler

         map<string, socketstream *>::iterator mit;
         for (mit = socks.begin(); mit != socks.end(); mit++)
         {
            delete mit->second;
         }
      }

      HYPRE_BigInt
      ElectrostaticsSolver::GetProblemSize()
      {
         return H1_fes->GlobalTrueVSize();
      }

      int
      ElectrostaticsSolver::GetLocalProblemSize()
      {
         return H1_fes->GetTrueVSize();
      }
      
      void
      ElectrostaticsSolver::PrintSizes()
      {
         HYPRE_BigInt size_h1 = H1_fes->GlobalTrueVSize();
         HYPRE_BigInt size_nd = HCurl_fes->GlobalTrueVSize();
         if (Mpi::Root() && verbose)
         {
            mfem::out << "Number of H1      unknowns: " << size_h1 << std::endl;
            mfem::out << "Number of H(Curl) unknowns: " << size_nd << std::endl;
         }
      }

      void ElectrostaticsSolver::SetAssemblyLevel(AssemblyLevel level)
      {
         assembly_level = level;
      }

      void ElectrostaticsSolver::Setup(int prec_type_, int pl)
      {
         prec_type = prec_type_;

         sw_setup.Start();

         if (Mpi::Root() && verbose)
         {
            mfem::out << "Setting up Electrostatics solver... " << std::endl;
         }

         /// 1. Check partial assembly
         bool tensor = UsesTensorBasis(*H1_fes);

         MFEM_VERIFY( (!((assembly_level == AssemblyLevel::PARTIAL) && !tensor)) , "Partial assembly is only supported for tensor elements.");

         if (Mpi::Root() && verbose)
         {
            if (assembly_level == AssemblyLevel::PARTIAL)
               mfem::out << "Using PARTIAL Assembly. " << std::endl;
            else if (assembly_level == AssemblyLevel::FULL)
               mfem::out << "Using FULL Assembly. " << std::endl;
            else
               mfem::out << "Using LEGACY Assembly. " << std::endl;
         }

         /// 1. Determine the essential BC degrees of freedom
         if ((bcs->GetDirichletDbcs()).size() > 0) // Applied potential
         {
            H1_fes->GetEssentialTrueDofs(bcs->GetDirichletAttr(), ess_bdr_phi_tdofs);
            ess_tdof_list.Append(ess_bdr_phi_tdofs);
         }

         if ((bcs->GetDirichletEFieldDbcs()).size() > 0) // Applied potential (uniform electric field)
         {
            H1_fes->GetEssentialTrueDofs(bcs->GetDirichletEFieldAttr(), ess_bdr_EField_tdofs);
            ess_tdof_list.Append(ess_bdr_EField_tdofs);
         }

         // Makes sure the problem is well-posed (in case of pure Neumann fix dof)
         FixEssentialTDofs(ess_tdof_list);

#ifdef MFEM_DEBUG
         mfem::out << "ess_tdof_list size: " << ess_tdof_list.Size() << " on rank " << my_id << std::endl;
#endif

         /// 2. Bilinear Forms, Linear Forms and Discrete Interpolators
         divEpsGrad = new ParBilinearForm(H1_fes);

         if (SigmaQ)
            divEpsGrad->AddDomainIntegrator(new DiffusionIntegrator(*SigmaQ));
         else
            divEpsGrad->AddDomainIntegrator(new DiffusionIntegrator(*SigmaMQ));
         
         // Add contribution to bilinear form from Robin BCs
         // NOTE: here we assume that the coefficient is constant, otherwise we'll need to re-assemble the operator
         if (bcs->GetRobinBcs().size() > 0)
         {
            for (auto &robin_bc : bcs->GetRobinBcs())
            {
               divEpsGrad->AddBoundaryIntegrator(new MassIntegrator(*robin_bc.alpha1), robin_bc.attr);
            }
         }

         SigmaMass = new ParBilinearForm(HCurl_fes);
         if (SigmaQ)
            SigmaMass->AddDomainIntegrator(new VectorFEMassIntegrator(*SigmaQ));
         else
            SigmaMass->AddDomainIntegrator(new VectorFEMassIntegrator(*SigmaMQ));

         if (assembly_level == AssemblyLevel::PARTIAL)
         {
            divEpsGrad->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            SigmaMass->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            grad->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         }
         else if (assembly_level == AssemblyLevel::FULL)
         {
            divEpsGrad->SetAssemblyLevel(AssemblyLevel::FULL);
            SigmaMass->SetAssemblyLevel(AssemblyLevel::LEGACY); // Legacy assembly for HCurl mass matrix since not yet supported
            //grad->SetAssemblyLevel(AssemblyLevel::FULL);
            divEpsGrad->EnableSparseMatrixSorting(Device::IsEnabled());
            //SigmaMass->EnableSparseMatrixSorting(Device::IsEnabled());
         }


         rhs_form = new ParLinearForm(H1_fes);

         // Add Domain Integrators to rhs_form
         if (volumetric_terms.size() > 0)
         {
            for (auto &volumetric_term : volumetric_terms)
            {
               rhs_form->AddDomainIntegrator(new DomainLFIntegrator(*(volumetric_term.coeff)), volumetric_term.attr);
            }
         }

         // Add neumann boundary conditions
         if (bcs->GetNeumannBcs().size() > 0)
         {
            for (auto &neumann_bc : bcs->GetNeumannBcs())
            {
               rhs_form->AddBoundaryIntegrator(new BoundaryLFIntegrator(*(neumann_bc.coeff)), neumann_bc.attr);
            }
         }

         // Add neumann boundary conditions (vector version)
         if (bcs->GetNeumannVectorBcs().size() > 0)
         {
            for (auto &neumann_bc : bcs->GetNeumannVectorBcs())
            {
               rhs_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*(neumann_bc.coeff)), neumann_bc.attr);
            }
         }
         
         // Add Robin boundary conditions
         if (bcs->GetRobinBcs().size() > 0)
         {
            symmetric = false;

            for (auto &robin_bc : bcs->GetRobinBcs())
            {
               rhs_form->AddBoundaryIntegrator(new BoundaryLFIntegrator(*(robin_bc.alpha2_u2)), robin_bc.attr);         // alpha2 * u
               rhs_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*(robin_bc.mu2_grad_u2)), robin_bc.attr); // mu2 * grad(u2)
            }
         }

         sw_setup.Stop();

         // Assemble bilinear and linear forms
         Assemble();

#ifdef MFEM_DEBUG
         if (Mpi::Root())
            mfem::out << "global problem size: " << this->GetProblemSize() << std::endl;
         mfem::out << "local problem size: " << this->GetLocalProblemSize() << " on rank " << my_id << std::endl;
#endif

         // Solver
         if ( assembly_level == AssemblyLevel::PARTIAL )
         {
            switch (prec_type)
            {
            case 0: // Jacobi Smoother
               prec = new OperatorJacobiSmoother(*divEpsGrad, ess_tdof_list);
               break;
            case 1: // LOR
               prec = new LORSolver<HypreBoomerAMG>(*divEpsGrad, ess_tdof_list);
               static_cast<LORSolver<HypreBoomerAMG>*>(prec)->GetSolver().SetPrintLevel(0);
               break;
            default:
               MFEM_ABORT("Unknown preconditioner type.");
            }
         }
         else
         {
            switch (prec_type)
            {
            case 0:
               prec = new HypreSmoother(*opA.As<HypreParMatrix>());
               dynamic_cast<HypreSmoother *>(prec)->SetType(HypreSmoother::Jacobi, 1);
               break;
            case 1:
               prec = new HypreBoomerAMG(*opA.As<HypreParMatrix>());
               static_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
               break;
            }
         }

         const real_t rel_tol = 1e-8;
         if (symmetric)
            solver = new CGSolver(H1_fes->GetComm());
         else
            solver = new GMRESSolver(H1_fes->GetComm());
         solver->iterative_mode = false;
         solver->SetRelTol(rel_tol);
         solver->SetAbsTol(0.0);
         solver->SetMaxIter(1000);
         solver->SetPrintLevel(pl);
         solver->SetOperator(*opA);
         solver->SetPreconditioner(*prec);
      }

      void ElectrostaticsSolver::FixEssentialTDofs(Array<int> &ess_tdof_list)
      {
         // In Parallel we need to check if any rank has essential DoFs and fix the solution on a suitable rank 
         // (e.g. rank 0 might not own any DoFs in the current domain)
#ifdef MFEM_USE_MPI       
         // Synchronize MPI to avoid communication issues
         MPI_Barrier(pmesh->GetComm());
         
         // Perform a global reduction to check if any rank has essential DoFs
         int local_has_ess_tdof = (ess_tdof_list.Size() > 0) ? 1 : 0;
         int global_has_ess_tdof = 0;
         MPI_Allreduce(&local_has_ess_tdof, &global_has_ess_tdof, 1, MPI_INT, MPI_LOR, pmesh->GetComm());

#ifdef MFEM_DEBUG
         if (Mpi::Root())
            mfem::out << "global_has_ess_tdof: " << global_has_ess_tdof << std::endl;
         mfem::out << "local_has_ess_tdof: " << local_has_ess_tdof << " on rank " << my_id << std::endl;
#endif

         if (global_has_ess_tdof == 0) // No Dirichlet BCs in the entire domain
         {
#ifdef MFEM_DEBUG
            if (Mpi::Root())
               mfem::out << "Fixing essential DoFs!" << std::endl;
#endif
            // Determine the rank that will fix the solution
            int fixed_rank = -1;

            HYPRE_BigInt INVALID_DOF = this->GetProblemSize();
            HYPRE_BigInt fixed_dof = INVALID_DOF + 1; // Use a large value for invalid DoFs

            // Each rank checks if it owns a valid DoF
            if (H1_fes->GetTrueVSize() > 0)
            {
               int local_dof = H1_fes->GetLocalTDofNumber(0); // Get the local index of the first DoF
               if (local_dof >= 0)                               // Ensure the DoF is valid
               {
                  fixed_rank = my_id;
                  fixed_dof = H1_fes->GetGlobalTDofNumber(0); // Get the global index of the first DoF
               }
            }

            
            // Find the first rank with a valid DoF
            struct
            {
                HYPRE_BigInt dof;
                int rank;
            } local_data = {fixed_dof, my_id}, global_data;
            
            MPI_Allreduce(&local_data, &global_data, 1, MPI_2INT, MPI_MINLOC, pmesh->GetComm());

            fixed_rank = global_data.rank;
            fixed_dof = global_data.dof;

#ifdef MFEM_DEBUG
            mfem::out << "fixed_dof: " << fixed_dof << " on rank " << fixed_rank << " (rank " << my_id << ")" << std::endl;
#endif

            // Check if no valid DoF was found
            if (fixed_dof >= INVALID_DOF)
            {
               mfem_error("No valid DoF found on any rank.");
            }

            // Fix the solution on the determined rank
            if (my_id == fixed_rank)
            {
               ess_tdof_list.SetSize(1);
               ess_tdof_list[0] = fixed_dof; 
            }
         }
#else
         // Serial case: If no essential DoFs are provided, fix the first DoF on rank 0
         if (ess_tdof_list.Size() == 0)
         {
               ess_tdof_list.SetSize(1);
               ess_tdof_list[0] = 0; // Use the first DoF on rank 0 by default
         }
#endif
      }

      void ElectrostaticsSolver::Assemble()
      {

         sw_assemble.Start();

         if (Mpi::Root() && verbose)
         {
            mfem::out << "Assembling ... " << flush;
         }

         // Assemble the divEpsGrad operator
         divEpsGrad->Assemble();

         divEpsGrad->FormSystemMatrix(ess_tdof_list, opA);

         // Assemble the mass matrix with conductivity
         Array<int> empty;
         SigmaMass->Assemble();
         SigmaMass->FormSystemMatrix(empty, opM);

         // Assemble the ParDiscreteGradOperator to compute gradient of Phi
         grad->Assemble();
         if (assembly_level != AssemblyLevel::PARTIAL)
         {
            grad->Finalize();
         }

         // Assemble rhs
         AssembleRHS();

         if (Mpi::Root() && verbose)
         {
            mfem::out << "done." << std::endl
                 << flush;
         }

         sw_assemble.Stop();
      }

      void ElectrostaticsSolver::AssembleRHS()
      {
         // Assemble rhs
         rhs_form->Assemble();
         rhs_form->ParallelAssemble(B);
      }

      void
      ElectrostaticsSolver::ProjectDirichletBCS(ParGridFunction &gf)
      {
            // Apply piecewise constant boundary condition
            for (auto &dirichlet_bc : bcs->GetDirichletDbcs())
            {
               gf.ProjectBdrCoefficient(*dirichlet_bc.coeff, dirichlet_bc.attr);
            }

            // Apply piecewise constant boundary condition
            for (auto &dirichlet_bc : bcs->GetDirichletEFieldDbcs())
            {
               gf.ProjectBdrCoefficient(*dirichlet_bc.coeff, dirichlet_bc.attr);
            }
      }

      void
      ElectrostaticsSolver::Update() // TODO: maybe for transient simulations we can add Update(real_t time) and update coeffs and bcs
      {
         if (Mpi::Root() && verbose)
         {
            mfem::out << "Updating ..." << std::endl;
         }

         // Inform the spaces that the mesh has changed
         // Note: we don't need to interpolate any GridFunctions on the new mesh
         // so we pass 'false' to skip creation of any transformation matrices.
         H1_fes->Update(false);
         HCurl_fes->Update(false);

         // Inform the grid functions that the space has changed.
         phi->Update();
         E->Update();
         rhs_form->Update();

         // Inform the bilinear forms that the space has changed.
         divEpsGrad->Update();
         SigmaMass->Update();

         // Inform the other objects that the space has changed.
         grad->Update();

         // Re-assemble the system
         Assemble();

         // Setup solver
         solver->SetOperator(*opA);

         delete prec;
         prec = nullptr;
         if (assembly_level == AssemblyLevel::PARTIAL)
         {
            switch (prec_type)
            {
            case 0: // Jacobi Smoother
               prec = new OperatorJacobiSmoother(*divEpsGrad, ess_tdof_list);
               break;
            case 1: // LOR
               prec = new LORSolver<HypreBoomerAMG>(*divEpsGrad, ess_tdof_list);
               static_cast<LORSolver<HypreBoomerAMG>*>(prec)->GetSolver().SetPrintLevel(0);
               break;
            default:
               MFEM_ABORT("Unknown preconditioner type.");
            }
         }
         else
         {
            switch (prec_type)
            {
            case 0:
               prec = new HypreSmoother(*opA.As<HypreParMatrix>());
               dynamic_cast<HypreSmoother *>(prec)->SetType(HypreSmoother::Jacobi, 1);
               break;
            case 1:
               prec = new HypreBoomerAMG(*opA.As<HypreParMatrix>());
               static_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
               break;
            }
         }
      }

      void
      ElectrostaticsSolver::Solve(bool updateRhs)
      {

         sw_solve.Start();

         if (Mpi::Root() && verbose)
         {
            mfem::out << "Running solver ... " << std::endl;
         }

         /// 1. Project dirichlet BCs in the electric potential grid function
         *phi = 0.0;
         Phi = 0.0;
         ProjectDirichletBCS(*phi);
         phi->GetTrueDofs(Phi);

         if (updateRhs)
         {
            AssembleRHS();
         }

         /// 2. Apply essential boundary conditions
         if (assembly_level == AssemblyLevel::PARTIAL)
         {
            auto *divEpsGrad_C = opA.As<ConstrainedOperator>();
            divEpsGrad_C->EliminateRHS(Phi, B);
         }
         else
         {
            divEpsGrad->EliminateVDofsInRHS(ess_tdof_list, Phi, B);
         }

         /// 3. Solve the system
         solver->Mult(B, Phi);

         /// 4. Update the solution gf with the new values
         phi->SetFromTrueDofs(Phi);

         /// 5. Compute the negative Gradient of the solution vector.  This is
         // the electric field corresponding to the scalar potential
         // represented by phi.
         grad->Mult(*phi, *E);
         *E *= -1.0;

         if (Mpi::Root() && verbose)
         {
            mfem::out << "Solver done. " << std::endl;
         }

         sw_solve.Stop();
      }

      void
      ElectrostaticsSolver::AddVolumetricTerm(Coefficient *coeff, Array<int> &attr)
      {
         volumetric_terms.emplace_back(attr, coeff);

         if (Mpi::Root() && verbose)
         {
            mfem::out << "Adding Volumetric heat term to domain attributes: ";
            for (int i = 0; i < attr.Size(); ++i)
            {
               if (attr[i] == 1)
               {
                  mfem::out << attr[i] << " ";
               }
            }
            mfem::out << std::endl;
         }
      }

      void
      ElectrostaticsSolver::AddVolumetricTerm(ScalarFuncT func, Array<int> &attr)
      {
         AddVolumetricTerm(new FunctionCoefficient(func), attr);

         if (Mpi::Root() && verbose)
         {
            mfem::out << "Adding  Volumetric term to domain attributes: ";
            for (int i = 0; i < attr.Size(); ++i)
            {
               if (attr[i] == 1)
               {
                  mfem::out << i << " ";
               }
            }
            mfem::out << std::endl;
         }
      }

      void
      ElectrostaticsSolver::AddVolumetricTerm(Coefficient *coeff, int &attr)
      {
         // Create array for attributes and mark given mesh boundary
         tmp_domain_attr = 0;
         tmp_domain_attr[attr - 1] = 1;

         // Add the volumetric term to the operator
         AddVolumetricTerm(coeff, tmp_domain_attr);
      }

      void
      ElectrostaticsSolver::AddVolumetricTerm(ScalarFuncT func, int &attr)
      {
         // Create array for attributes and mark given mesh boundary
         tmp_domain_attr = 0;
         tmp_domain_attr[attr - 1] = 1;

         // Add the volumetric term to the operator
         AddVolumetricTerm(func, tmp_domain_attr);
      }

      real_t ElectrostaticsSolver::ElectricLosses(ParGridFunction &E_gf) const
      {
         // Compute E^T M1 E, where M1 is the HCurl mass matrix with conductivity
         real_t el = 0.0;

         int true_vsize =  HCurl_fes->GetTrueVSize(); 

         Vector x; x.SetSize(true_vsize);
         Vector y; y.SetSize(true_vsize);
         E_gf.GetTrueDofs(x);
         opM->Mult(x, y);
         el = InnerProduct(x, y);

         return el;
      }

      void ElectrostaticsSolver::GetJouleHeating(ParGridFunction &w_gf) const
      {
         // This applies the definition of the finite element degrees-of-freedom
         // to convert the function to a set of discrete values
         w_gf.ProjectCoefficient(*w_coeff);
      }

      real_t JouleHeatingCoefficient::Eval(ElementTransformation &T,
                                           const IntegrationPoint &ip)
      {
         Vector E, J;
         E_gf->GetVectorValue(T, ip, E);
         J.SetSize(E.Size());

         if(Q)
         {
            real_t q = Q->Eval(T, ip); // Evaluate sigma at the point
            J = E;
            J *= q;
         }
         else if(MQ)
         {
            DenseMatrix thisSigma;
            MQ->Eval(thisSigma, T, ip); // Evaluate sigma at the point
            thisSigma.Mult(E, J);          // J = sigma * E
         }

         return InnerProduct(J, E);     // W = J dot E
      }

      void ElectrostaticsSolver::GetErrorEstimates(Vector &errors)
      {
         if (Mpi::Root() && verbose)
         {
            mfem::out << "Estimating Error ... " << flush;
         }

         // Space for the discontinuous (original) flux
         DiffusionIntegrator *flux_integrator;
         if (SigmaQ)
            flux_integrator = new DiffusionIntegrator(*SigmaQ);
         else
            flux_integrator = new DiffusionIntegrator(*SigmaMQ);

         L2_FECollection flux_fec(order, pmesh->Dimension());
         // ND_FECollection flux_fec(order, pmesh->Dimension());
         ParFiniteElementSpace flux_fes(pmesh.get(), &flux_fec, pmesh->SpaceDimension());

         // Space for the smoothed (conforming) flux
         int norm_p = 1;
         RT_FECollection smooth_flux_fec(order - 1, pmesh->Dimension());
         ParFiniteElementSpace smooth_flux_fes(pmesh.get(), &smooth_flux_fec);

         L2ZZErrorEstimator(*flux_integrator, *phi,
                            smooth_flux_fes, flux_fes, errors, norm_p);

         if (Mpi::Root() && verbose)
         {
            mfem::out << "done." << std::endl;
         }
      }

      void
      ElectrostaticsSolver::RegisterParaviewFields(ParaViewDataCollection &paraview_dc_)
      {
         paraview_dc = &paraview_dc_;

         if ( order > 1 )
         {
            paraview_dc->SetHighOrderOutput(true);
            paraview_dc->SetLevelsOfDetail(order);
         }

         paraview_dc->RegisterField("Phi", phi);
         paraview_dc->RegisterField("E", E);
      }

      void
      ElectrostaticsSolver::RegisterVisItFields(VisItDataCollection &visit_dc_)
      {
         visit_dc = &visit_dc_;

         visit_dc->RegisterField("Phi", phi);
         visit_dc->RegisterField("E", E);
      }

      void
      ElectrostaticsSolver::AddParaviewField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(paraview_dc,
                     "Paraview data collection not initialized. Call RegisterParaviewFields first.");
         paraview_dc->RegisterField(field_name, gf);
      }

      void
      ElectrostaticsSolver::AddVisItField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(visit_dc,
                     "VisIt data collection not initialized. Call RegisterVisItFields first.");
         visit_dc->RegisterField(field_name, gf);
      }

      void
      ElectrostaticsSolver::WriteFields(int it, const real_t &time )
      {
         if (visit_dc)
         {
            if (Mpi::Root() && verbose)
            {
               mfem::out << "Writing VisIt files ..." << flush;
            }

            HYPRE_BigInt prob_size = this->GetProblemSize();
            visit_dc->SetCycle(it);
            visit_dc->SetTime(time);
            visit_dc->Save();

            if (Mpi::Root() && verbose)
            {
               mfem::out << " done." << std::endl;
            }
         }

         if (paraview_dc)
         {
            if (Mpi::Root() && verbose)
            {
               mfem::out << "Writing Paraview files ..." << flush;
            }

            HYPRE_BigInt prob_size = this->GetProblemSize();
            paraview_dc->SetCycle(it);
            paraview_dc->SetTime(time);
            paraview_dc->Save();

            if (Mpi::Root() && verbose)
            {
               mfem::out << " done." << std::endl;
            }
         }
      }

      void
      ElectrostaticsSolver::InitializeGLVis()
      {
         if (Mpi::Root() && verbose)
         {
            mfem::out << "Opening GLVis sockets." << std::endl;
         }

         socks["Phi"] = new socketstream;
         socks["Phi"]->precision(8);

         socks["E"] = new socketstream;
         socks["E"]->precision(8);
      }

      void
      ElectrostaticsSolver::DisplayToGLVis()
      {
         if (Mpi::Root() && verbose)
         {
            mfem::out << "Sending data to GLVis ..." << flush;
         }

         char vishost[] = "localhost";
         int visport = 19916;

         int Wx = 0, Wy = 0;                 // window position
         int Ww = 350, Wh = 350;             // window size
         int offx = Ww + 10, offy = Wh + 45; // window offsets

         VisualizeField(*socks["Phi"], vishost, visport,
                        *phi, "Electric Potential (Phi)", Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(*socks["E"], vishost, visport,
                        *E, "Electric Field (E)", Wx, Wy, Ww, Wh);
         Wx += offx;

         Wx = 0;
         Wy += offy; // next line

         if (Mpi::Root() && verbose)
         {
            mfem::out << " done." << std::endl;
         }
      }

      void ElectrostaticsSolver::PrintTimingData()
      {
          // Declare local timing variables
          real_t my_rt[1], setup_time_max[1], assemble_time_max[1], solve_time_max[1];
      
          // Record setup, assemble, and solve times
          real_t setup_time = sw_setup.RealTime();
          real_t assemble_time = sw_assemble.RealTime();
          real_t solve_time = sw_solve.RealTime();
      
          // Perform MPI reductions to get the maximum times across all ranks
          my_rt[0] = setup_time;
          MPI_Reduce(my_rt, setup_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());
      
          my_rt[0] = assemble_time;
          MPI_Reduce(my_rt, assemble_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());
      
          my_rt[0] = solve_time;
          MPI_Reduce(my_rt, solve_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());
      
          // Print the timing data in a table format
          if (Mpi::Root())
          {
              mfem::out << "-------------------------------------------------" << endl;
              mfem::out << "| " << std::setw(20) << "Timing" << " | " << std::setw(20) << "Value" << " |" << endl;
              mfem::out << "-------------------------------------------------" << endl;
              mfem::out << "| " << std::setw(20) << "Setup time" << " | " << std::setw(20) << setup_time_max[0] << " s |" << endl;
              mfem::out << "| " << std::setw(20) << "Assemble time" << " | " << std::setw(20) << assemble_time_max[0] << " s |" << endl;
              mfem::out << "| " << std::setw(20) << "Solve time" << " | " << std::setw(20) << solve_time_max[0] << " s |" << endl;
              mfem::out << "| " << std::setw(20) << "Total time" << " | " << std::setw(20) << setup_time_max[0] + assemble_time_max[0] + solve_time_max[0] << " s |" << endl;
              mfem::out << "| " << std::setw(20) << "Problem size" << " | " << std::setw(20) << this->GetProblemSize() << "   |" << endl;
              mfem::out << "-------------------------------------------------" << endl;
          }
      }

      void ElectrostaticsSolver::display_banner(ostream &os)
      {
         if (Mpi::Root())
         {
              os << " ____  ____     ____  ___  _  _  ____ " << std::endl
                 << "(  _ \\(  __)___(  __)/ __)( \\/ )(___ \\" << std::endl
                 << " )   / ) _)(___)) _)( (__ / \\/ \\ / __/" << std::endl
                 << "(__\\_)(__)     (____)\\___)\\_)(_/(____)" << std::endl
                 << std::flush;
          }
      }

   } // namespace electrostatics

} // namespace mfem

#endif // MFEM_USE_MPI
