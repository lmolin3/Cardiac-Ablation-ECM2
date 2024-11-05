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
#include "mfem.hpp"
#include "celldeath_solver_gotran.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{
   namespace celldeathgotran
   {

      CellDeathSolverGotran::CellDeathSolverGotran(std::shared_ptr<ParMesh> pmesh_,
                                       ParGridFunction *T_,
                                       real_t A1_, real_t A2_, real_t A3_,
                                       real_t deltaE1_, real_t deltaE2_, real_t deltaE3_,
                                       bool verbose)
          : pmesh(pmesh_), A1(A1_), A2(A2_), A3(A3_),
            deltaE1(deltaE1_), deltaE2(deltaE2_), deltaE3(deltaE3_), T_gf(T_), verbose(verbose)
      {
         fes = T_gf->ParFESpace();         
         fes_truevsize = fes->GetTrueVSize();

         order = fes->FEColl()->GetOrder();

         N_gf.SetSpace(fes);
         U_gf.SetSpace(fes);
         D_gf.SetSpace(fes);

         N.SetSize(fes_truevsize);
         U.SetSize(fes_truevsize);
         D.SetSize(fes_truevsize);
         T.SetSize(fes_truevsize);

         visit_dc = nullptr;
         paraview_dc = nullptr;

         // Initialize ODE model parameters
         parameters_nodes = new double[fes_truevsize][num_param];
         //init_parameters_values(parameters);
         init_state_values(init_states);

         // Initialize state and parameters
         states = new double[fes_truevsize][num_states];
         N_gf.GetTrueDofs(N);
         U_gf.GetTrueDofs(U);
         D_gf.GetTrueDofs(D);
         T_gf->GetTrueDofs(T);
         
         for (int i = 0; i < fes_truevsize; ++i) {
            // Set the initial state
            states[i][0] = init_states[0];
            states[i][1] = init_states[1];
            states[i][2] = init_states[2];

            N[i] = states[i][0];
            U[i] = states[i][1];
            D[i] = states[i][2];

            // Set the parameters
            real_t Tval = T[i];
            parameters_nodes[i][0] = A1*exp(-deltaE1/(R*Tval)); // k1
            parameters_nodes[i][1] = A2*exp(-deltaE2/(R*Tval)); // k2
            parameters_nodes[i][2] = A3*exp(-deltaE3/(R*Tval)); // k3
         }

         N_gf.SetFromTrueDofs(N);
         U_gf.SetFromTrueDofs(U);
         D_gf.SetFromTrueDofs(D);

      }

      CellDeathSolverGotran::~CellDeathSolverGotran()
      {
         // doesn't take ownership of coefficients  used to create the rate coefficients ki
         //delete fes; // --> Using the Temperature grid function space
         delete [] parameters_nodes;
         delete [] states;
      }

      // Solve the system using the Gotran generated ODE solver
      void CellDeathSolverGotran::Solve(real_t t, real_t dt, int method, int substeps)
      {
         // Check ODE method
         // 0: explicit Euler
         // 1: Rush-Larsen
         // 2: Generalized Rush-Larsen
         // 3: Hybrid Generalized Rush-Larsen
         MFEM_ASSERT((method >= 0 && method < 4), "Invalid method for time integration");

         // Get the solution and state vectors
         N_gf.GetTrueDofs(N);
         U_gf.GetTrueDofs(U);
         D_gf.GetTrueDofs(D);
         T_gf->GetTrueDofs(T);

         for (int i = 0; i < fes_truevsize; ++i)
         {
            // Set the initial state
            states[i][0] = N[i];
            states[i][1] = U[i];
            states[i][2] = D[i];

            // Set the parameters
            real_t Tval = T[i];
            parameters_nodes[i][0] = A1 * exp(-deltaE1 / (R * Tval)); // k1
            parameters_nodes[i][1] = A2 * exp(-deltaE2 / (R * Tval)); // k2
            parameters_nodes[i][2] = A3 * exp(-deltaE3 / (R * Tval)); // k3

            // Solve
            real_t t_int = t;
            real_t dt_ode = dt / substeps;
            for (int k = 0; k < substeps; k++)
            {
               switch (method)
               {
               case 0:
                  forward_explicit_euler(states[i], t_int, dt_ode, parameters_nodes[i]);
                  break;
               case 1:
                  forward_rush_larsen(states[i], t_int, dt_ode, parameters_nodes[i]);
                  break;
               case 2:
                  forward_generalized_rush_larsen(states[i], t_int, dt_ode, parameters_nodes[i]);
                  break;
               case 3:
                  forward_hybrid_generalized_rush_larsen(states[i], t_int, dt_ode, parameters_nodes[i]);
                  break;
               default:
                  // Handle invalid method_index
                  break;
               }
               t_int += dt_ode;
            }

            // Set the solution
            N[i] = states[i][0];
            U[i] = states[i][1];
            D[i] = states[i][2];
         }

         N_gf.SetFromTrueDofs(N);
         U_gf.SetFromTrueDofs(U);
         D_gf.SetFromTrueDofs(D);
   }

      // Print the CellDeath-ecm2 ascii logo to the given ostream
      void CellDeathSolverGotran::display_banner(std::ostream &os)
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
      CellDeathSolverGotran::RegisterParaviewFields(ParaViewDataCollection &paraview_dc_)
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
      CellDeathSolverGotran::RegisterVisItFields(VisItDataCollection &visit_dc_)
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

      void CellDeathSolverGotran::AddParaviewField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(paraview_dc,
                     "Paraview data collection not initialized. Call RegisterParaviewFields first.");
         paraview_dc->RegisterField(field_name, gf);
      }

      void CellDeathSolverGotran::AddVisItField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(visit_dc,
                     "VisIt data collection not initialized. Call RegisterVisItFields first.");
         visit_dc->RegisterField(field_name, gf);
      }

      void
      CellDeathSolverGotran::WriteFields(const int &it, const double &time)
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
