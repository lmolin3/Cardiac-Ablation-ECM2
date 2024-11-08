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

         mat = DenseMatrix(3);
         P = DenseMatrix(3);
         lambda = Vector(3);
         Xn = Vector(3);
         X = Vector(3);

         visit_dc = nullptr;
         paraview_dc = nullptr;

         // Initialize the solution
         ConstantCoefficient zero(0.0), one(1.0);
         N_gf.ProjectCoefficient(one);
         U_gf.ProjectCoefficient(zero);
         D_gf.ProjectCoefficient(zero);
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

      void CellDeathSolver::Solve(real_t t, real_t dt)
      {
         // Solve the system
         // For each quadrature point:
         // 1. Evaluate the rate coefficients and assemble the local matrix
         // 2. Compute eigenvalues and eigenvectors of the local matrix
         // 3. Solve the system for the initial conditions
         // 4. Construct the new solution

         DenseMatrix P(3, 3);
         Vector lambda(3), norms(3);

         T_gf->GetTrueDofs(Tsrc);
         N_gf.GetTrueDofs(N);
         U_gf.GetTrueDofs(U);
         D_gf.GetTrueDofs(D);

         // Project the temperature field to the target space (if needed)
         ProjectTemperature(Tsrc, T);

         for (int i = 0; i < fes_truevsize; ++i)
         {
            // Set the initial state
            Xn(0) = N[i];
            Xn(1) = U[i];
            Xn(2) = D[i];

            // Set the parameters
            real_t Tval = T[i];
            k1 = A1 * exp(-deltaE1 / (R * Tval)); // k1
            k2 = A2 * exp(-deltaE2 / (R * Tval)); // k2
            k3 = A3 * exp(-deltaE3 / (R * Tval)); // k3

            // Symbolic computation
            real_t sqrt_factor = std::sqrt(std::pow(k1, 2) + 2.0 * k1 * k2 - 2.0 * k1 * k3 + std::pow(k2, 2) + 2.0 * k2 * k3 + std::pow(k3, 2));
            real_t sum_factor = -0.5 * (k1 + k2 + k3);

            real_t lambda1 = 0.0;
            real_t lambda2 = sum_factor - 0.5 * sqrt_factor;
            real_t lambda3 = sum_factor + 0.5 * sqrt_factor;

            real_t e1x = 0.0;
            real_t e1y = 0.0;
            real_t e1z = 1.0;

            real_t e2x = (k1 + k2 - k3 + sqrt_factor) / (2 * k3);
            real_t e2y = -(k1 + k2 + k3 + sqrt_factor) / (2 * k3);
            real_t e2z = 1.0;

            real_t e3x = (k1 + k2 - k3 - sqrt_factor) / (2 * k3);
            real_t e3y = -(k1 + k2 + k3 - sqrt_factor) / (2 * k3);
            real_t e3z = 1.0;

            // Create an array to hold the data in column-major order
            real_t eigenvectors[9] = {e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z};
            real_t eigenvalues[3] = {lambda1, lambda2, lambda3};

            // Use the UseExternalData method to wrap the data array
            P.UseExternalData(eigenvectors, 3, 3);
            lambda.SetData(eigenvalues);

            // Normalize the eigenvectors
            P.Norm2(norms);
            P.InvRightScaling(norms);

            // Solve the system for the initial conditions   P C = Xn
            DenseMatrix Plu(P); // Deep copy of P since LinearSolve returns the LU factor for 3x3 matrices
            LinearSolve(Plu, Xn.GetData());

            // Construct the new solution X_n+1 = P * exp(lambda * dt) C
            Xn(0) *= exp(lambda(0) * dt);
            Xn(1) *= exp(lambda(1) * dt);
            Xn(2) *= exp(lambda(2) * dt);
            P.Mult(Xn, X);

            // Update the solution
            N[i] = X(0);
            U[i] = X(1);
            D[i] = X(2);
         }

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
