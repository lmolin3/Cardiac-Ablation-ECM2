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

// celldeath_solver.cpp

#include "celldeath_solver.hpp"
#include "mfem.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{
   namespace celldeath
   {
      CellDeathSolver::CellDeathSolver(std::shared_ptr<ParMesh> pmesh_, int order_,
                                       ParGridFunction *T_,
                                       real_t A1_, real_t A2_, real_t A3_,
                                       real_t deltaE1_, real_t deltaE2_, real_t deltaE3_,
                                       bool verbose)
          : pmesh(pmesh_), A1(A1_), A2(A2_), A3(A3_),
            deltaE1(deltaE1_), deltaE2(deltaE2_), deltaE3(deltaE3_), T_gf(T_), verbose(verbose)
      {
         // TODO: Add som MFEM_ASSERT 
         
         // Initialize the FE spaces for projection
         fesT = T_gf->ParFESpace();
         order = order_;
         orderT = fesT->FEColl()->GetOrder();

         if (order == orderT)
         { // Fes have same order, no need to project
            fes = fesT;
         }
         else
         { // Fes have different order (temperaure = n, cell-death = m), at runtime need to project temperature to target space
            SetupProjection();
         }

         // Initialize the grid functions and vectors
         fes_truevsize = fes->GetTrueVSize();
         fesT_truevsize = fesT->GetTrueVSize();

         N_gf.SetSpace(fes);
         U_gf.SetSpace(fes);
         D_gf.SetSpace(fes);

         N.SetSize(fes_truevsize);
         U.SetSize(fes_truevsize);
         D.SetSize(fes_truevsize);
         T.SetSize(fes_truevsize);
         Tsrc.SetSize(fesT_truevsize);

         Xn = Vector(3);
         X = Vector(3);

         visit_dc = nullptr;
         paraview_dc = nullptr;

         // Initialize the solution
         ConstantCoefficient zero(0.0), one(1.0);
         N_gf.ProjectCoefficient(one);
         U_gf.ProjectCoefficient(zero);
         D_gf.ProjectCoefficient(zero);
         N = 1.0;
         U = 0.0;
         D = 0.0;
      }

      CellDeathSolver::~CellDeathSolver()
      {
         // doesn't take ownership of coefficients  used to create the rate coefficients ki
         if (order != orderT)
         {
            delete fec;
            delete fes;
            delete transferOp;
         }
      }

      HYPRE_BigInt
      CellDeathSolver::GetProblemSize()
      {
         return fes->GlobalTrueVSize();
      }

      void CellDeathSolver::SetupProjection()
      {
         // Create the FE spaces for cell-death variables
         fec = (order == 0)
                   ? static_cast<FiniteElementCollection *>(new L2_FECollection(order, pmesh->Dimension()))
                   : static_cast<FiniteElementCollection *>(new H1_FECollection(order, pmesh->Dimension()));
         fes = new ParFiniteElementSpace(pmesh.get(), fec);

         // Create the TransferOperator
         transferOp = (orderT > order) ? new TrueTransferOperator(*fes, *fesT) : new TrueTransferOperator(*fesT, *fes);
      }

      void CellDeathSolver::ProjectTemperature(Vector &Tin, Vector &Tout)
      {
         if (order == orderT)
         {
            Tout = Tin;
            return;
         }

         // Project the temperature field from the source space to the target space
         if (orderT > order) // Fine to coarse restriction
         {
            transferOp->MultTranspose(Tin, Tout);
         }
         else // Coarse to fine prolongation
         {
            transferOp->Mult(Tin, Tout);
         }
      }

      inline void CellDeathSolver::EigenSystem(real_t k1, real_t k2, real_t k3, Vector &lambda, DenseMatrix &P)
      {
         // Create an index based on which ki are non-zero
         int index = (k1 != 0.0 ? 1 : 0) | (k2 != 0.0 ? 2 : 0) | (k3 != 0.0 ? 4 : 0);
         real_t e1[3], e2[3], e3[3];
         real_t lambda_[3];

         switch (index)
         {
         case 0: // All k1, k2, k3 are zero
         {
            lambda_[0] = 0;
            lambda_[1] = 0;
            lambda_[2] = 0;
            e1[0] = 1.0; e1[1] = 0.0; e1[2] = 0.0;
            e2[0] = 0.0; e2[1] = 1.0; e2[2] = 0.0;
            e3[0] = 0.0; e3[1] = 0.0; e3[2] = 1.0;
            break;
         }
         case 1: // Only k1 is non-zero
         {
            lambda_[0] = 0;
            lambda_[1] = 0;
            lambda_[2] = -1.0 * k1;
            e1[0] = 0; e1[1] = 1; e1[2] = 0;
            e2[0] = 0; e2[1] = 0; e2[2] = 1;
            e3[0] = -1; e3[1] = 1; e3[2] = 0;
            break;
         }
         case 2: // Only k2 is non-zero
         {
            lambda_[0] = 0;
            lambda_[1] = 0;
            lambda_[2] = -1.0 * k2;
            e1[0] = 1; e1[1] = 0; e1[2] = 0;
            e2[0] = 0; e2[1] = 0; e2[2] = 1;
            e3[0] = -1; e3[1] = 1; e3[2] = 0;
            break;
         }
         case 3: // k1 and k2 are non-zero
         {
            lambda_[0] = 0;
            lambda_[1] = 0;
            lambda_[2] = -1.0 * k1 - 1.0 * k2;
            e1[0] = k2 / k1; e1[1] = 1; e1[2] = 0;
            e2[0] = 0; e2[1] = 0; e2[2] = 1;
            e3[0] = -1; e3[1] = 1; e3[2] = 0;
            break;
         }
         case 4: // Only k3 is non-zero
         {
            lambda_[0] = 0;
            lambda_[1] = 0;
            lambda_[2] = -1.0 * k3;
            e1[0] = 1; e1[1] = 0; e1[2] = 0;
            e2[0] = 0; e2[1] = 0; e2[2] = 1;
            e3[0] = 0; e3[1] = -1; e3[2] = 1;
            break;
         }
         case 5: // k1 and k3 are non-zero
         {
            lambda_[0] = 0;
            lambda_[1] = -1.0 * k1;
            lambda_[2] = -1.0 * k3;
            e1[0] = 0; e1[1] = 0; e1[2] = 1;
            e2[0] = (k1 - k3) / k3; e2[1] = -k1 / k3; e2[2] = 1;
            e3[0] = 0; e3[1] = -1; e3[2] = 1;
            break;
         }
         case 6: // k2 and k3 are non-zero
         {
            lambda_[0] = 0;
            lambda_[1] = 0;
            lambda_[2] = -1.0 * k2 - 1.0 * k3;
            e1[0] = 1; e1[1] = 0; e1[2] = 0;
            e2[0] = 0; e2[1] = 0; e2[2] = 1;
            e3[0] = k2 / k3; e3[1] = -(k2 + k3) / k3; e3[2] = 1;
            break;
         }
         case 7: // All k1, k2, k3 are non-zero
         {
            // Precompute factors for eigenvalues
            const real_t k1k2 = k1 * k2;
            const real_t k1k3 = k1 * k3;
            const real_t k2k3 = k2 * k3;
            const real_t sqrt_factor = std::sqrt(k1 * k1 + 2.0 * k1k2 - 2.0 * k1k3 + k2 * k2 + 2.0 * k2k3 + k3 * k3);
            const real_t sum_factor = (k1 + k2 + k3);

            // Compute eigenvalues
            lambda_[0] = 0.0;
            lambda_[1] = -0.5 * sum_factor - 0.5 * sqrt_factor;
            lambda_[2] = -0.5 * sum_factor + 0.5 * sqrt_factor;

            // Compute eigenvectors and fill in matrix P
            e1[0] = 0.0; e1[1] = 0.0; e1[2] = 1.0;
            e2[0] = (k1 + k2 - k3 + sqrt_factor) / (2 * k3); e2[1] = -(sum_factor + sqrt_factor) / (2 * k3); e2[2] = 1.0;
            e3[0] = (k1 + k2 - k3 - sqrt_factor) / (2 * k3); e3[1] = -(sum_factor - sqrt_factor) / (2 * k3); e3[2] = 1.0;
            break;
         }
         default:
         {
            MFEM_ABORT("Invalid index");
            return;
         }
         }

         // Assign the eigenvalues
         lambda = lambda_;

         // Create eigenvector matrix P
         real_t eigenvectors[9] = {e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2]};
         P.UseExternalData(eigenvectors, 3, 3);

         // Normalize the eigenvectors
         Vector norms(3);
         P.Norm2(norms);
         P.InvRightScaling(norms);
      }

      void CellDeathSolver::Solve(real_t t, real_t dt)
      {
         // Allocate reusable resources to avoid redundant allocation in each loop
         DenseMatrix P, Plu;
         Vector lambda(3), exp_lambda_dt(3);

         T_gf->GetTrueDofs(Tsrc);
         N_gf.GetTrueDofs(N);
         U_gf.GetTrueDofs(U);
         D_gf.GetTrueDofs(D);

         ProjectTemperature(Tsrc, T); // Project the temperature field

         static constexpr real_t TOL = 1e-8;

         for (int i = 0; i < fes_truevsize; ++i)
         {
            // Set initial state
            Xn(0) = N[i];
            Xn(1) = U[i];
            Xn(2) = D[i];

            // Precompute temperature-dependent coefficients
            const real_t Tval = T[i];
            bool nnz = (Tval != 0);
            const real_t invTval = nnz ? (invR / Tval) : 0;
            real_t k1 = nnz ? (A1 * exp(-deltaE1 * invTval)) : 0.0;
            real_t k2 = nnz ? (A2 * exp(-deltaE2 * invTval)) : 0.0;
            real_t k3 = nnz ? (A3 * exp(-deltaE3 * invTval)) : 0.0;
            k1 = k1 < TOL ? 0.0 : k1;
            k2 = k2 < TOL ? 0.0 : k2;
            k3 = k3 < TOL ? 0.0 : k3;

            // Compute eigenvalues and eigenvectors based on the index
            EigenSystem( k1, k2, k3, lambda, P);

            // Precompute exp(lambda * dt) for use in constructing the solution
            exp_lambda_dt(0) = 1.0; // exp(0) = 1 for lambda1 = 0
            exp_lambda_dt(1) = exp(lambda(1) * dt);
            exp_lambda_dt(2) = exp(lambda(2) * dt);

            // Solve P C = Xn by creating a deep copy of P (Plu) to preserve it
            Plu = P; // Copy matrix P into Plu for solving
            LinearSolve(Plu, Xn.GetData());

            // Construct the new solution X_n+1 = P * exp(lambda * dt) * C
            Xn *= exp_lambda_dt;
            P.Mult(Xn, X);

            if (std::isnan(X(0)) || std::isnan(X(1)) || std::isnan(X(2)))
            {
               MFEM_ABORT("Solution contains NaNs at index " << i);
            }  

            // Update the solution arrays
            N[i] = X(0);
            U[i] = X(1);
            D[i] = X(2);
         }

         // Set updated solution fields
         N_gf.SetFromTrueDofs(N);
         U_gf.SetFromTrueDofs(U);
         D_gf.SetFromTrueDofs(D);
      }

      // Print the CellDeath-ecm2 ascii logo to the given ostream
      void CellDeathSolver::display_banner(std::ostream &os)
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            os << "             ____    __           __  __  \n"
                  "  ________  / / /___/ /__  ____ _/ /_/ /_ \n"
                  " / ___/ _ \\/ / / __  / _ \\/ __ `/ __/ __ \\\n"
                  "/ /__/  __/ / / /_/ /  __/ /_/ / /_/ / / /\n"
                  "\\___/\\___/_/_/\\__,_/\\___/\\__,_/\\__/_/ /_/ \n"
                  "                                          \n"
               << endl
               << flush;
         }
      }

      void
      CellDeathSolver::RegisterParaviewFields(ParaViewDataCollection &paraview_dc_)
      {
         paraview_dc = &paraview_dc_;

         if (order > 1)
         {
            paraview_dc->SetHighOrderOutput(true);
            paraview_dc->SetLevelsOfDetail(order);
         }

         paraview_dc->RegisterField("N", &N_gf);
         paraview_dc->RegisterField("U", &U_gf);
         paraview_dc->RegisterField("D", &D_gf);
      }

      void
      CellDeathSolver::RegisterVisItFields(VisItDataCollection &visit_dc_)
      {
         visit_dc = &visit_dc_;

         if (order > 1)
         {
            visit_dc->SetLevelsOfDetail(order);
         }

         visit_dc->RegisterField("N", &N_gf);
         visit_dc->RegisterField("U", &U_gf);
         visit_dc->RegisterField("D", &D_gf);
      }

      void CellDeathSolver::AddParaviewField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(paraview_dc,
                     "Paraview data collection not initialized. Call RegisterParaviewFields first.");
         paraview_dc->RegisterField(field_name, gf);
      }

      void CellDeathSolver::AddVisItField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(visit_dc,
                     "VisIt data collection not initialized. Call RegisterVisItFields first.");
         visit_dc->RegisterField(field_name, gf);
      }

      void
      CellDeathSolver::WriteFields(const int &it, const double &time)
      {
         if (visit_dc)
         {
            if (pmesh->GetMyRank() == 0 && verbose)
            {
               cout << "Writing VisIt files ..." << flush;
            }

            visit_dc->SetCycle(it);
            visit_dc->SetTime(time);
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

            paraview_dc->SetCycle(it);
            paraview_dc->SetTime(time);
            paraview_dc->Save();

            if (pmesh->GetMyRank() == 0 && verbose)
            {
               cout << " done." << endl;
            }
         }
      }

   } // namespace celldeath
} // namespace mfem

#endif // MFEM_USE_MPI
