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
                                                 MatrixCoefficient *Sigma_,
                                                 bool verbose_)
          : order(order_),
            bcs(bcs),
            pmesh(pmesh_),
            visit_dc(nullptr),
            paraview_dc(nullptr),
            H1FESpace(nullptr),
            HCurlFESpace(nullptr),
            divEpsGrad(nullptr),
            SigmaMass(nullptr),
            rhs_form(nullptr),
            B(nullptr),
            prec(nullptr),
            pa(false),
            Sigma(Sigma_), // Must be deleted outside solver
            verbose(verbose_)
      {
         const int dim = pmesh->Dimension();

         // Define compatible parallel finite element spaces on the parallel
         // mesh. Here we use arbitrary order H1 for potential and ND for the electric field.
         H1FESpace = new H1_ParFESpace(pmesh.get(), order, pmesh->Dimension(), BasisType::GaussLobatto);
         HCurlFESpace = new ND_ParFESpace(pmesh.get(), order, pmesh->Dimension());

         // Discrete derivative operator
         grad = new ParDiscreteGradOperator(H1FESpace, HCurlFESpace);

         // Build grid functions
         phi = new ParGridFunction(H1FESpace);
         *phi = 0.0;

         E = new ParGridFunction(HCurlFESpace);
         *E = 0.0;

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

         delete H1FESpace;
         delete HCurlFESpace;

         delete B;

         delete prec;

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
         return H1FESpace->GlobalTrueVSize();
      }

      void
      ElectrostaticsSolver::PrintSizes()
      {
         HYPRE_BigInt size_h1 = H1FESpace->GlobalTrueVSize();
         HYPRE_BigInt size_nd = HCurlFESpace->GlobalTrueVSize();
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Number of H1      unknowns: " << size_h1 << endl;
            cout << "Number of H(Curl) unknowns: " << size_nd << endl;
         }
      }

      void ElectrostaticsSolver::EnablePA(bool pa_)
      {
         pa = pa_;
      }

      void ElectrostaticsSolver::Setup(int prec_type, int pl)
      {

         sw_setup.Start();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Setting up Electrostatics solver... " << endl;
         }

         /// 1. Check partial assembly
         bool tensor = UsesTensorBasis(*H1FESpace);

         MFEM_VERIFY(!(pa && !tensor), "Partial assembly is only supported for tensor elements.");

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            if (pa)
            {
               cout << "Using Partial Assembly. " << endl;
            }
            else
            {
               cout << "Using Full Assembly. " << endl;
            }
         }

         /// 1. Determine the essential BC degrees of freedom
         if ((bcs->GetDirichletDbcs()).size() > 0) // Applied potential
         {
            H1FESpace->GetEssentialTrueDofs(bcs->GetDirichletAttr(), ess_bdr_phi_tdofs);
            ess_tdof_list.Append(ess_bdr_phi_tdofs);
         }

         if ((bcs->GetDirichletEFieldDbcs()).size() > 0) // Applied potential (uniform electric field)
         {
            H1FESpace->GetEssentialTrueDofs(bcs->GetDirichletEFieldAttr(), ess_bdr_EField_tdofs);
            ess_tdof_list.Append(ess_bdr_EField_tdofs);
         }

         if (ess_tdof_list.Size() == 0) // Check if any essential BCs were applied (fix at least one point since solution is not unique)
         {
            // If not, use the first DoF on processor zero by default
            if (pmesh->GetMyRank() == 0 && verbose)
            {
               ess_tdof_list.SetSize(1);
               ess_tdof_list[0] = 0;
            }
         }

         /// 2. Bilinear Forms, Linear Forms and Discrete Interpolators
         divEpsGrad = new ParBilinearForm(H1FESpace);
         divEpsGrad->AddDomainIntegrator(new DiffusionIntegrator(*Sigma));

         SigmaMass = new ParBilinearForm(HCurlFESpace);
         SigmaMass->AddDomainIntegrator(new VectorFEMassIntegrator(Sigma));

         if (pa)
         {
            divEpsGrad->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            SigmaMass->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            grad->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         }

         rhs_form = new ParLinearForm(H1FESpace);

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

         sw_setup.Stop();

         // Assemble bilinear and linear forms
         Assemble();

         // Solver
         if (pa)
         {
            switch (prec_type)
            {
            case 0: // Jacobi Smoother
               prec = new OperatorJacobiSmoother(*divEpsGrad, ess_tdof_list);
               break;
            case 1: // LOR
               prec = new LORSolver<HypreBoomerAMG>(*divEpsGrad, ess_tdof_list);
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

         const double rel_tol = 1e-8;
         solver = CGSolver(H1FESpace->GetComm());
         solver.iterative_mode = false;
         solver.SetRelTol(rel_tol);
         solver.SetAbsTol(0.0);
         solver.SetMaxIter(1000);
         solver.SetPrintLevel(pl);
         solver.SetOperator(*opA);
         solver.SetPreconditioner(*prec);
      }

      void ElectrostaticsSolver::Assemble()
      {

         sw_assemble.Start();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Assembling ... " << flush;
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
         if (!pa)
         {
            grad->Finalize();
         }

         // Assemble rhs
         rhs_form->Assemble();
         B = rhs_form->ParallelAssemble();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "done." << endl
                 << flush;
         }

         sw_assemble.Stop();
      }

      void
      ElectrostaticsSolver::ProjectDirichletBCS(ParGridFunction &gf)
      {
         if (bcs->GetDirichletDbcs().size() > 0)
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
      }

      void
      ElectrostaticsSolver::Update() // TODO: maybe for transient simulations we can add Update(double time) and update coeffs and bcs
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Updating ..." << endl;
         }

         // Inform the spaces that the mesh has changed
         // Note: we don't need to interpolate any GridFunctions on the new mesh
         // so we pass 'false' to skip creation of any transformation matrices.
         H1FESpace->Update(false);
         HCurlFESpace->Update(false);

         // Inform the grid functions that the space has changed.
         phi->Update();
         E->Update();
         rhs_form->Update();
         delete B;
         B = nullptr;

         // Inform the bilinear forms that the space has changed.
         divEpsGrad->Update();
         SigmaMass->Update();

         // Inform the other objects that the space has changed.
         grad->Update();

         // Re-assemble the system
         Assemble();

         // Setup solver
         solver.SetOperator(*opA);

         delete prec;
         prec = nullptr;
         if (pa)
         {
            //int cheb_order = 2;
            //Vector diag(H1FESpace->GetTrueVSize());
            //divEpsGrad->AssembleDiagonal(diag);
            //prec = new OperatorChebyshevSmoother(*opA, diag, ess_tdof_list, cheb_order);
            prec = new OperatorJacobiSmoother(*divEpsGrad, ess_tdof_list);
         }
         else
         {
            prec = new HypreBoomerAMG(*opA.As<HypreParMatrix>());
            static_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
         }
      }

      void
      ElectrostaticsSolver::Solve()
      {

         sw_solve.Start();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Running solver ... " << endl;
         }

         /// 1. Project dirichlet BCs in the electric potential grid function
         *phi = 0.0;
         Phi = 0.0;
         ProjectDirichletBCS(*phi);
         phi->GetTrueDofs(Phi);

         /// 2. Apply essential boundary conditions
         if (pa)
         {
            auto *divEpsGrad_C = opA.As<ConstrainedOperator>();
            divEpsGrad_C->EliminateRHS(Phi, *B);
         }
         else
         {
            divEpsGrad->EliminateVDofsInRHS(ess_tdof_list, Phi, *B);
         }

         /// 3. Solve the system
         solver.Mult(*B, Phi);

         /// 4. Update the solution gf with the new values
         phi->SetFromTrueDofs(Phi);

         /// 5. Compute the negative Gradient of the solution vector.  This is
         // the electric field corresponding to the scalar potential
         // represented by phi.
         grad->Mult(*phi, *E);
         *E *= -1.0;

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Solver done. " << endl;
         }

         sw_solve.Stop();
      }

      void
      ElectrostaticsSolver::AddVolumetricTerm(Coefficient *coeff, Array<int> &attr)
      {
         volumetric_terms.emplace_back(attr, coeff);

         if (pmesh->GetMyRank() == 0 && verbose)
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

         if (pmesh->GetMyRank() == 0 && verbose)
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

      double ElectrostaticsSolver::ElectricLosses(ParGridFunction &E_gf) const
      {
         // Compute E^T M1 E, where M1 is the H1 mass matrix with conductivity
         double el = 0.0;

         int true_vsize =  HCurlFESpace->GetTrueVSize(); 

         Vector x; x.SetSize(true_vsize);
         Vector y; y.SetSize(true_vsize);
         E_gf.GetTrueDofs(x);
         opM->Mult(x, y);
         el = InnerProduct(x, y);

         return el;
      }

      void ElectrostaticsSolver::GetJouleHeating(ParGridFunction &E_gf,
                                                 ParGridFunction &w_gf) const
      {
         // The w_coeff object stashes a reference to sigma and E, and it has
         // an Eval method that will be used by ProjectCoefficient.
         JouleHeatingCoefficient w_coeff(Sigma, E_gf);

         // This applies the definition of the finite element degrees-of-freedom
         // to convert the function to a set of discrete values
         w_gf.ProjectCoefficient(w_coeff);
      }

      real_t JouleHeatingCoefficient::Eval(ElementTransformation &T,
                                           const IntegrationPoint &ip)
      {
         Vector E, J;
         DenseMatrix thisSigma;
         E_gf.GetVectorValue(T, ip, E);
         Sigma->Eval(thisSigma, T, ip); // Evaluate sigma at the point
         thisSigma.Mult(E, J);         // J = sigma * E
         return InnerProduct(J, E);    // W = J dot E
      }

      void ElectrostaticsSolver::GetErrorEstimates(Vector &errors)
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Estimating Error ... " << flush;
         }

         // Space for the discontinuous (original) flux
         DiffusionIntegrator flux_integrator(*Sigma);
         L2_FECollection flux_fec(order, pmesh->Dimension());
         // ND_FECollection flux_fec(order, pmesh->Dimension());
         ParFiniteElementSpace flux_fes(pmesh.get(), &flux_fec, pmesh->SpaceDimension());

         // Space for the smoothed (conforming) flux
         int norm_p = 1;
         RT_FECollection smooth_flux_fec(order - 1, pmesh->Dimension());
         ParFiniteElementSpace smooth_flux_fes(pmesh.get(), &smooth_flux_fec);

         L2ZZErrorEstimator(flux_integrator, *phi,
                            smooth_flux_fes, flux_fes, errors, norm_p);

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "done." << endl;
         }
      }

      void
      ElectrostaticsSolver::RegisterParaviewFields(ParaViewDataCollection &paraview_dc_)
      {
         paraview_dc = &paraview_dc_;

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
      ElectrostaticsSolver::WriteFields(int it)
      {
         if (visit_dc)
         {
            if (pmesh->GetMyRank() == 0 && verbose)
            {
               cout << "Writing VisIt files ..." << flush;
            }

            HYPRE_BigInt prob_size = this->GetProblemSize();
            visit_dc->SetCycle(it);
            visit_dc->SetTime(prob_size);
            visit_dc->Save();

            if (pmesh->GetMyRank() == 0 && verbose)
            {
               cout << " done." << endl;
            }
         }

         if (paraview_dc)
         {
            if (pmesh->GetMyRank() == 0 && verbose)
            {
               cout << "Writing Paraview files ..." << flush;
            }

            HYPRE_BigInt prob_size = this->GetProblemSize();
            paraview_dc->SetCycle(it);
            paraview_dc->SetTime(prob_size);
            paraview_dc->Save();

            if (pmesh->GetMyRank() == 0 && verbose)
            {
               cout << " done." << endl;
            }
         }
      }

      void
      ElectrostaticsSolver::InitializeGLVis()
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Opening GLVis sockets." << endl;
         }

         socks["Phi"] = new socketstream;
         socks["Phi"]->precision(8);

         socks["E"] = new socketstream;
         socks["E"]->precision(8);
      }

      void
      ElectrostaticsSolver::DisplayToGLVis()
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Sending data to GLVis ..." << flush;
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

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << " done." << endl;
         }
      }

      void ElectrostaticsSolver::PrintTimingData()
      {
         double my_rt[3], rt_max[3];

         my_rt[0] = sw_setup.RealTime();
         my_rt[1] = sw_assemble.RealTime();
         my_rt[2] = sw_solve.RealTime();

         MPI_Reduce(my_rt, rt_max, 3, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            mfem::out << std::setw(10) << "SETUP" << std::setw(10) << "ASSEMBLE"
                      << std::setw(10) << "SOLVE"
                      << "\n";

            mfem::out << std::setprecision(3) << std::setw(10) << my_rt[0]
                      << std::setw(10) << my_rt[1] << std::setw(10) << my_rt[2]
                      << "\n";

            mfem::out << std::setprecision(3) << std::setw(10) << " " << std::setw(10)
                      << my_rt[1] / my_rt[1] << std::setw(10) << my_rt[2] / my_rt[1]
                      << "\n";

            mfem::out << std::setprecision(8);
         }
      }

      void ElectrostaticsSolver::display_banner(ostream &os)
      {
         if (pmesh->GetMyRank() == 0)
         {
            os << "  ____   ____     __   __            " << endl
               << "  \\   \\ /   /___ |  |_/  |______     " << endl
               << "   \\   Y   /  _ \\|  |\\   __\\__  \\    " << endl
               << "    \\     (  <_> )  |_|  |  / __ \\_  " << endl
               << "     \\___/ \\____/|____/__| (____  /  " << endl
               << "                                \\/   " << endl
               << flush;
         }
      }

   } // namespace electrostatics

} // namespace mfem

#endif // MFEM_USE_MPI
