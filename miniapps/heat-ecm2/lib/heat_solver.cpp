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

// NOTE on Robin bcs:
// Since the n.Grad(u) terms arise by integrating -Div(m Grad(u)) by parts we
// must introduce the coefficient 'm' into the boundary conditions.
// Therefore, in the case of the Neumann BC, we actually enforce m n.Grad(u)
// = m g rather than simply n.Grad(u) = g.

#include "heat_solver.hpp"

constexpr int max_bdf_order = 6;

#ifdef MFEM_USE_MPI

using namespace std;
namespace mfem
{

   using namespace common;

   namespace heat
   {

      real_t CelsiusToKelvin(real_t Tc)
      {
         return Tc + 273.15;
      }

      real_t KelvinToCelsius(real_t Tk)
      {
         return Tk - 273.15;
      }

      HeatSolver::HeatSolver(std::shared_ptr<ParMesh> pmesh_, int order_,
                             BCHandler *bcs,
                             MatrixCoefficient *Kappa,
                             Coefficient *c,
                             Coefficient *rho,
                             real_t advection,
                             VectorCoefficient *u,
                             real_t reaction,
                             int ode_solver_type,
                             bool verbose_)
          : op(nullptr),
            order(order_),
            pmesh(pmesh_),
            ode_solver(nullptr),
            visit_dc(nullptr),
            paraview_dc(nullptr),
            verbose(verbose_)
      {
         sw_init.Start();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "\nInitializing Heat solver... " << flush;
         }

         const int dim = pmesh->Dimension();

         ///<--- Define  parallel finite element spaces on the parallel mesh
         // Here we use arbitrary order H1 for the Temperature.
         // Note: Gauss-Lobatto basis is used for H1 space to ensure LOR can be used
         H1FESpace = new H1_ParFESpace(pmesh.get(), order, pmesh->Dimension(), BasisType::GaussLobatto);
         VectorH1FESpace = new H1_ParFESpace(pmesh.get(), order, pmesh->Dimension(), BasisType::GaussLobatto, pmesh->Dimension());

         fes_truevsize = H1FESpace->GetTrueVSize();

         ///<--- Create the ParGridFunction and Vector for Temperature
         T = new Vector(fes_truevsize); 
         *T = 0.0;
         T_gf = new ParGridFunction(H1FESpace);
         *T_gf = 0.0;

         ///<--- Create the ODE Solver (no initialization yet)
         CreateODESolver(ode_solver_type, *op);

         ///<--- Create the AdvectionReactionDiffusionOperator
         op = new AdvectionReactionDiffusionOperator(pmesh, *H1FESpace, bcs, Kappa, c, rho, advection, u, reaction, verbose);

         ///<--- Initialize ODESolver with operator
         ode_solver->Init(*op);

         tmp_domain_attr.SetSize(pmesh->attributes.Max());
         tmp_domain_attr = 0;

         ///<--- Initialize the BDF coefficients
         beta.SetSize(time_scheme_order);
         beta = 0.0;
         alpha = 0.0;

         sw_init.Stop();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "\n...done." << endl;
         }
      }

      HeatSolver::HeatSolver(std::shared_ptr<ParMesh> pmesh_, int order_,
                             BCHandler *bcs,
                             MatrixCoefficient *Kappa,
                             Coefficient *c,
                             Coefficient *rho,
                             int ode_solver_type,
                             bool verbose_)
          : HeatSolver(pmesh_, order_, bcs, Kappa, c, rho, 0.0, nullptr, 0.0, ode_solver_type, verbose_) {}

      HeatSolver::~HeatSolver()
      {
         delete H1FESpace;
         delete VectorH1FESpace;

         delete T;

         delete T_gf;

         delete op;

         map<string, socketstream *>::iterator mit;
         for (mit = socks.begin(); mit != socks.end(); mit++)
         {
            delete mit->second;
         }
      }

      HYPRE_BigInt
      HeatSolver::GetProblemSize()
      {
         return H1FESpace->GlobalTrueVSize();
      }

      void
      HeatSolver::PrintSizes()
      {
         HYPRE_BigInt size_h1 = H1FESpace->GlobalTrueVSize();
         if (pmesh->GetMyRank() == 0)
         {
            cout << "Number of H1      unknowns: " << size_h1 << endl;
         }
      }

      void HeatSolver::CreateODESolver(int ode_solver_type, TimeDependentOperator &op)
      {
         // Remove solver_name and use correct time_scheme_order
         implicit_time_integration = (ode_solver_type >= 20);

         // Select ODESolver using unique_ptr, then release ownership to raw pointer
         ode_solver = ODESolver::Select(ode_solver_type);

         // Set time_scheme_order based on ode_solver_type
         switch (ode_solver_type)
         {
         // Explicit RK methods
         case 1:  // Forward Euler
         case 11: // AB1
         case 21: // Backward Euler
         case 51: // AM1
            time_scheme_order = 1;
            break;
         case 2:  // RK2
         case 12: // AB2
         case 22: // SDIRK23 (order 2)
         case 32: // Implicit Midpoint
         case 52: // AM2
            time_scheme_order = 2;
            break;
         case 3:  // RK3 SSP
         case 13: // AB3
         case 23: // SDIRK33
         case 33: // SDIRK23 (order 3)
         case 53: // AM3
            time_scheme_order = 3;
            break;
         case 4:  // RK4
         case 14: // AB4
         case 34: // SDIRK34
         case 54: // AM4
            time_scheme_order = 4;
            break;
         case 6:  // RK6
         case 15: // AB5
            time_scheme_order = 6;
            break;
         // Generalized Alpha methods (order 2)
         case 40:
         case 41:
         case 42:
         case 43:
         case 44:
         case 45:
         case 46:
         case 47:
         case 48:
         case 49:
         case 50:
            time_scheme_order = 2;
            break;
         default:
            mfem_error("Unknown ODE solver type \n");
         }

         MFEM_ASSERT(time_scheme_order <= max_bdf_order, "Invalid order of the time integrator. Maximum allowed is 6.");

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "\nUsing ODE solver type: " << ode_solver_type << " with order " << time_scheme_order << flush;
         }
      }

      void
      HeatSolver::Setup(real_t dt, int prec_type)
      {
         sw_setup.Start();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "\nSetting up Heat solver... " << flush;
         }

         ///<--- Set up the internal AdvectionReactionDiffusionOperator
         op->Setup(dt, implicit_time_integration, prec_type);

         ///<--- Initialize every entry with zero solution (only stored essential_tdofs)
         ess_tdof_list = op->GetEssTDofList();
         Vector tmp;
         tmp.SetSize(fes_truevsize);
         tmp = 0.0;
         T_prev.resize(time_scheme_order);
         for (int i = 0; i < time_scheme_order; ++i)
         {
            UpdateTimeStepHistory(tmp);
         }

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "done." << endl;
         }

         sw_setup.Stop();
      }

      void
      HeatSolver::SetInitialTemperature(ParGridFunction &T0_gf)
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "\nSetting initial temperature... " << flush;
         }

         ///<--- Retrieve the true dofs of the initial temperature
         T0_gf.GetTrueDofs(*T);

         ///<--- Push vector into the time history
         UpdateTimeStepHistory(*T);

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "...done." << endl;
         }
      }

      void
      HeatSolver::Update()
      {
         mfem_error("HeatSolver::Update() is not implemented yet.");
      }

      void HeatSolver::UpdatedParameters()
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "\nUpdating FE space and Reassembling operators..." << flush;
         }

         // Update the operator
         op->Update();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "done." << endl;
         }
      }

      void
      HeatSolver::Step(real_t &t, real_t &dt, int step, bool provisional)
      {
         sw_solve.Start();

         ///<--- Check if the operator needs to be re-assembled
         op->Rebuild();

         ///<--- Update the operator
         SetTimeIntegrationCoefficients(step);

         ///<--- Set solution vector to the previous solution (without new bcs)
         T_gf->GetTrueDofs(*T);

         ///<--- Set the time for the boundary conditions, update the temperature gf and operator time
         op->ProjectDirichletBCS(t + dt, *T_gf);
         T_gf->GetTrueDofs(T_bcs); // Need another vector since T_final = T + dt * dT, and dT(ess_tdof) != 0
         ComputeDerivativeApproximation(T_bcs, dt);

         ///<--- Time advancing (Note time is updated by the ode_solver)
         ode_solver->Step(*T, t, dt);

         ///<--- Set bcs after solving
         for (int i = 0; i < ess_tdof_list.Size(); ++i)
         {
            (*T)[ess_tdof_list[i]] = T_bcs[ess_tdof_list[i]];
         }

         ///<--- Update the temperature gf with the new solution
         T_gf->SetFromTrueDofs(*T);

         //<--- Update time (if provisional, restore previous time and return)
         if (provisional)
         {
            t -= dt;
            return;
         }

         //<--- Update the time step history
         UpdateTimeStepHistory(*T);

         sw_solve.Stop();
      }

      void
      HeatSolver::UpdateTimeStepHistory(const Vector &new_solution)
      {
         Vector new_solution_restricted;
         new_solution_restricted.SetSize(ess_tdof_list.Size());
         new_solution.GetSubVector(ess_tdof_list, new_solution_restricted);
         T_prev.push_front(new_solution_restricted);
         int sz = T_prev.size();
         while (sz > time_scheme_order)
         {
            T_prev.pop_back();
            sz--;
         }
      }

      void
      HeatSolver::UpdateTimeStepHistory()
      {
         ///<--- Overload to update the history with the internally stored solution
         UpdateTimeStepHistory(*T);
      }

      void HeatSolver::SetTimeIntegrationCoefficients(int step)
      {

         ///<--- Maximum BDF order to use at current time step
         // step + 1 <= order <= time_scheme_order
         int bdf_order = std::min(step-1, time_scheme_order);

         ///<--- Set the coefficients for the BDF scheme
         // du/dt ~ (1/dt) * (  alpha un  +  ubdf  ),   with   ubdf = sum_{i=1}^{bdf_order} beta_i * u_{n-i}
         //
         // Based on the order, the coefficients are:
         //
         // order | alpha |                beta                 |
         //   1   |    1   | -1                                 |
         //   2   |   3/2  | -2 | 1/2                           |
         //   3   |  11/6  | -3 | 3/2 | -1/3                    |
         //   4   |  25/12 | -4 | 3   | -4/3 | 1/4              |
         //   5   | 137/60 | -5 | 5   | -10/3| 5/4 | -1/5       |
         //   6   | 147/60 | -6 | 15/2| -20/3| 15/4| -6/5 | 1/6 |

         if (step == 1 && bdf_order <= 1)
         {
            alpha = 1.0;
            beta[0] = -1.0;
         }
         else if (step > 2 && bdf_order == 2)
         {
            alpha = 1.5;
            beta[0] = -2.0;
            beta[1] = 0.5;
         }
         else if (step > 3 && bdf_order == 3)
         {
            alpha = 11.0 / 6.0;
            beta[0] = -3.0;
            beta[1] = 1.5;
            beta[2] = -1.0 / 3.0;
         }
         else if (step > 4 && bdf_order == 4)
         {
            alpha = 25.0 / 12.0;
            beta[0] = -4.0;
            beta[1] = 3.0;
            beta[2] = -4.0 / 3.0;
            beta[3] = 1.0 / 4.0;
         }
         else if (step > 5 && bdf_order == 5)
         {
            alpha = 137.0 / 60.0;
            beta[0] = -5.0;
            beta[1] = 5.0;
            beta[2] = -10.0 / 3.0;
            beta[3] = 5.0 / 4.0;
            beta[4] = -1.0 / 5.0;
         }
         else if (step > 6 && bdf_order == 6)
         {
            alpha = 147.0 / 60.0;
            beta[0] = -6.0;
            beta[1] = 15.0 / 2.0;
            beta[2] = -20.0 / 3.0;
            beta[3] = 15.0 / 4.0;
            beta[4] = -6.0 / 5.0;
            beta[5] = 1.0 / 6.0;
         }
      }

      void HeatSolver::ComputeDerivativeApproximation(const Vector &T, real_t dt) const
      {

         Vector &dT_approx = op->GetDerivativeApproximation();
         dT_approx = 0.0;

         for (int i = 0; i < ess_tdof_list.Size(); ++i)
         {
            const int dof = ess_tdof_list[i];
            dT_approx[dof] = alpha * T[dof]; // alpha * T_{n+1}

            // Sum over the previous solutions
            for (int j = 0; j < time_scheme_order; ++j)
            {
               const auto Tnmj = T_prev[j].Read();
               dT_approx[dof] += beta[j] * Tnmj[i]; // beta_j * T_{n-j}
            }

            dT_approx[dof] /= dt; // Divide by dt
         }
      }

      void HeatSolver::AddVolumetricTerm(Coefficient *coeff, Array<int> &attr, bool own)
      {
         op->AddVolumetricTerm(coeff, attr, own);

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

      void HeatSolver::AddVolumetricTerm(ScalarFuncT func, Array<int> &attr, bool own)
      {
         op->AddVolumetricTerm(func, attr, own);

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            mfem::out << "Adding  Volumetric heat term to domain attributes: ";
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

      void HeatSolver::AddVolumetricTerm(Coefficient *coeff, int &attr, bool own)
      {
         ///<--- Create array for attributes and mark given mesh boundary
         tmp_domain_attr = 0;
         tmp_domain_attr[attr - 1] = 1;

         ///<--- Add the volumetric term to the operator
         AddVolumetricTerm(coeff, tmp_domain_attr, own);
      }

      void HeatSolver::AddVolumetricTerm(ScalarFuncT func, int &attr, bool own)
      {
         ///<--- Create array for attributes and mark given mesh boundary
         tmp_domain_attr = 0;
         tmp_domain_attr[attr - 1] = 1;

         ///<--- Add the volumetric term to the operator
         AddVolumetricTerm(func, tmp_domain_attr, own);
      }

      void
      HeatSolver::RegisterParaviewFields(ParaViewDataCollection &paraview_dc_)
      {
         paraview_dc = &paraview_dc_;

         if (order > 1)
         {
            paraview_dc->SetHighOrderOutput(true);
            paraview_dc->SetLevelsOfDetail(order);
         }

         paraview_dc->RegisterField("T", T_gf);
      }

      void
      HeatSolver::RegisterVisItFields(VisItDataCollection &visit_dc_)
      {
         visit_dc = &visit_dc_;

         if (order > 1)
         {
            visit_dc->SetLevelsOfDetail(order);
         }

         visit_dc->RegisterField("T", T_gf);
      }

      void HeatSolver::AddParaviewField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(paraview_dc,
                     "Paraview data collection not initialized. Call RegisterParaviewFields first.");
         paraview_dc->RegisterField(field_name, gf);
      }

      void HeatSolver::AddVisItField(const std::string &field_name, ParGridFunction *gf)
      {
         MFEM_VERIFY(visit_dc,
                     "VisIt data collection not initialized. Call RegisterVisItFields first.");
         visit_dc->RegisterField(field_name, gf);
      }

      void
      HeatSolver::WriteFields(const int &it, const real_t &time)
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

      void
      HeatSolver::InitializeGLVis()
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Opening GLVis sockets." << endl;
         }

         socks["T"] = new socketstream;
         socks["T"]->precision(8);
      }

      void
      HeatSolver::DisplayToGLVis()
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Sending data to GLVis ..." << flush;
         }

         char vishost[] = "localhost";
         int visport = 19916;

         int Wx = 0, Wy = 0;                 // window position
         int Ww = 350, Wh = 350;             // window size
         int offx = Ww + 10; // window offsets

         VisualizeField(*socks["T"], vishost, visport,
                        *T_gf, "Temperature (T)", Wx, Wy, Ww, Wh);
         Wx += offx;

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << " done." << endl;
         }
      }

      std::vector<real_t> HeatSolver::GetTimingData()
      {
         std::vector<real_t> timing_data(3);

         timing_data[0] = sw_init.RealTime();
         timing_data[1] = sw_setup.RealTime();
         timing_data[2] = sw_solve.RealTime();

         real_t rt_max[3];
         MPI_Reduce(timing_data.data(), rt_max, 3, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            timing_data[0] = rt_max[0];
            timing_data[1] = rt_max[1];
            timing_data[2] = rt_max[2];
         }

         return timing_data;
      }

      void HeatSolver::PrintTimingData()
      {
         std::vector<real_t> timing_data = GetTimingData();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            mfem::out << "Timing: " << std::setw(8) << "init" << std::setw(8) << "setup" << std::setw(8) << "solve"
                      << std::setw(8) << "solve/setup" << std::setw(8) << "total/setup"
                      << "\n";

            mfem::out << std::setprecision(3) << std::setw(10) << timing_data[0] << std::setw(10) << timing_data[1]
                      << std::setw(10) << timing_data[2] << "\n";

            mfem::out << std::setprecision(3) << std::setw(10)
                      << timing_data[0] / timing_data[1] << std::setw(10)
                      << " " << std::setw(10)
                      << timing_data[2] / timing_data[1] << "\n";

            mfem::out << std::setprecision(8);
         }
      }

      // Print the Heat-ecm2 ascii logo to the given ostream
      void HeatSolver::display_banner(ostream &os)
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            os << "   __            __ \n"
                  "  / /  ___ ___ _/_/_\n"
                  " / _ \\/ -_) _ `/ __/\n"
                  "/_//_/\\__/\\_,_/\\__/ \n"
                  "                    \n"
               << endl
               << flush;
         }
      }

   } // namespace heat

} // namespace mfem

#endif // MFEM_USE_MPI
