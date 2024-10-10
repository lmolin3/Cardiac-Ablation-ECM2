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

#ifdef MFEM_USE_MPI

constexpr int max_bdf_order = 6;

using namespace std;
namespace mfem
{

   using namespace common;

   namespace heat
   {

      double CelsiusToKelvin(double Tc)
      {
         return Tc + 273.15;
      }

      double KelvinToCelsius(double Tk)
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

         display_banner(cout);

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "\nInitializing Heat solver... " << flush;
         }

         const int dim = pmesh->Dimension();

         // Define  parallel finite element spaces on the parallel
         // Here we use arbitrary order H1 for the Temperature.
         // Note: Gauss-Lobatto basis is used for H1 space to ensure LOR can be used
         H1FESpace = new H1_ParFESpace(pmesh.get(), order, pmesh->Dimension(), BasisType::GaussLobatto);
         VectorH1FESpace = new H1_ParFESpace(pmesh.get(), order, pmesh->Dimension(), BasisType::GaussLobatto, pmesh->Dimension());

         fes_truevsize = H1FESpace->GetTrueVSize();

         // Create the ParGridFunction and Vector for Temperature
         T = new Vector(fes_truevsize);
         T_gf = new ParGridFunction(H1FESpace);
         *T_gf = 0.0;

         // Create the ODE Solver (no initialization yet)
         ode_solver = CreateODESolver(ode_solver_type, *op);

         // Create the AdvectionReactionDiffusionOperator
         op = new AdvectionReactionDiffusionOperator(pmesh, *H1FESpace, bcs, Kappa, c, rho, advection, u, reaction);

         // Initialize ODESolver with operator
         ode_solver->Init(*op);

         tmp_domain_attr.SetSize(pmesh->attributes.Max());
         tmp_domain_attr = 0;

         // Initialize the BDF coefficients
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

         delete T;

         delete T_gf;

         delete op;

         delete ode_solver;

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
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "Number of H1      unknowns: " << size_h1 << endl;
         }
      }

      ODESolver *HeatSolver::CreateODESolver(int ode_solver_type, TimeDependentOperator &op)
      {
         // Define the ODESolver used to advance the heat solver
         std::string solver_name;
         ODESolver *ode_solver_ = nullptr;

         switch (ode_solver_type)
         {
         // Implicit L-stable methods
         case 1:
            ode_solver_ = new BackwardEulerSolver;
            time_scheme_order = 1;
            solver_name = "Backward Euler";
            break;
         case 2:
            ode_solver_ = new SDIRK23Solver(2);
            time_scheme_order = 2;
            solver_name = "SDIRK23";
            break;
         case 3:
            ode_solver_ = new SDIRK33Solver;
            time_scheme_order = 3;
            solver_name = "SDIRK33";
            break;
         // Implicit A-stable methods (not L-stable)
         case 4:
            ode_solver_ = new ImplicitMidpointSolver;
            time_scheme_order = 2;
            solver_name = "Implicit Midpoint";
            break;
         case 5:
            ode_solver_ = new SDIRK23Solver;
            time_scheme_order = 3;
            solver_name = "SDIRK23";
            break;
         case 6:
            ode_solver_ = new SDIRK34Solver;
            time_scheme_order = 4;
            solver_name = "SDIRK34";
            break;
         // Explicit methods
         case 7:
            ode_solver_ = new ForwardEulerSolver;
            time_scheme_order = 1;
            solver_name = "Forward Euler";
            break;
         case 8:
            ode_solver_ = new RK2Solver(0.5);
            time_scheme_order = 2;
            solver_name = "RK2";
            break; // midpoint method
         case 9:
            ode_solver_ = new RK3SSPSolver;
            time_scheme_order = 3;
            solver_name = "RK3 SSP";
            break;
         case 10:
            ode_solver_ = new RK4Solver;
            time_scheme_order = 4;
            solver_name = "RK4";
            break;
         default:
            mfem_error("Unknown ODE solver type \n");
         }

         MFEM_ASSERT(time_scheme_order <= max_bdf_order, "Invalid order of the time integrator. Maximum allowed is 6.");

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "\nUsing time integrator: " << solver_name << flush;
         }

         return ode_solver_;
      }

      void
      HeatSolver::Setup()
      {
         sw_setup.Start();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "\nSetting up Heat solver... " << flush;
         }

         // Set up the internal AdvectionReactionDiffusionOperator
         op->Setup();

         // Initialize every entry with zero solution (only stored essential_tdofs)
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

         // Retrieve the true dofs of the initial temperature
         T0_gf.GetTrueDofs(*T);

         // Push vector into the time history
         UpdateTimeStepHistory(*T);

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "...done." << endl;
         }
      }

      void
      HeatSolver::Update() // TODO: maybe for transient simulations we can add Update(double time) and update coeffs and bcs
      {
         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "\nUpdating FE space and Reassembling operators..." << flush;
         }

         // Inform the spaces that the mesh has changed
         // Note: we don't need to interpolate any GridFunctions on the new mesh
         // so we pass 'false' to skip creation of any transformation matrices.
         H1FESpace->Update(false);

         // Update the operator
         op->Update();

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << "done." << endl;
         }
      }

      void
      HeatSolver::Step(double &time, double dt, int step, bool UpdateHistory)
      {
         // Solve the system
         sw_solve.Start();

         // Update the operator
         op->SetTimeStep(dt);
         SetTimeIntegrationCoefficients(step);

         // Set solution vector to the previous solution
         T_gf->GetTrueDofs(*T);

         // Set the time for the boundary conditions, update the temperature gf and operator time
         op->ProjectDirichletBCS(time + dt, *T_gf);
         T_gf->GetTrueDofs(T_bcs); // Need another vector since T_final = T + dt * dT, and dT(ess_tdof) != 0
         ComputeDerivativeApproximation(T_bcs, dt);

         // Time advancing (Note time is updated by the ode_solver)
         ode_solver->Step(*T, time, dt);

         // Set bcs after solving
         for (int i = 0; i < ess_tdof_list.Size(); ++i)
         {
            (*T)[ess_tdof_list[i]] = T_bcs[ess_tdof_list[i]];
         }

         // Update the temperature gf with the new solution
         T_gf->SetFromTrueDofs(*T);

         // Update the time history
         if (UpdateHistory)
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
         // Overload to update the history with the internally stored solution
         UpdateTimeStepHistory(*T);
      }

      void HeatSolver::SetTimeIntegrationCoefficients(int step)
      {

         // Maximum BDF order to use at current time step
         // step + 1 <= order <= time_scheme_order
         int bdf_order = std::min(step, time_scheme_order);

         // Set the coefficients for the BDF scheme
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

         if (step == 1 && bdf_order == 1)
         {
            alpha = 1.0;
            beta[0] = -1.0;
         }
         else if (step >= 2 && bdf_order == 2)
         {
            alpha = 1.5;
            beta[0] = -2.0;
            beta[1] = 0.5;
         }
         else if (step >= 3 && bdf_order == 3)
         {
            alpha = 11.0 / 6.0;
            beta[0] = -3.0;
            beta[1] = 1.5;
            beta[2] = -1.0 / 3.0;
         }
         else if (step >= 4 && bdf_order == 4)
         {
            alpha = 25.0 / 12.0;
            beta[0] = -4.0;
            beta[1] = 3.0;
            beta[2] = -4.0 / 3.0;
            beta[3] = 1.0 / 4.0;
         }
         else if (step >= 5 && bdf_order == 5)
         {
            alpha = 137.0 / 60.0;
            beta[0] = -5.0;
            beta[1] = 5.0;
            beta[2] = -10.0 / 3.0;
            beta[3] = 5.0 / 4.0;
            beta[4] = -1.0 / 5.0;
         }
         else if (step >= 6 && bdf_order == 6)
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

      void HeatSolver::ComputeDerivativeApproximation(const Vector &T, double dt) const
      {

         Vector &dT_approx = op->GetDerivativeApproximation();
         dT_approx = 0.0;

         // std::cout << std::setprecision(10);
         // std::cout << "T_bc[0]: " << T[0] << endl;

         for (int i = 0; i < ess_tdof_list.Size(); ++i)
         {
            const int dof = ess_tdof_list[i];
            // out << "dof: " << dof << endl;

            dT_approx[dof] = alpha * T[dof]; // alpha * T_{n+1}
            // std::cout << std::setprecision(10);
            // std::cout << "T[dof]: " << T[dof] << endl;

            // Sum over the previous solutions
            for (int j = 0; j < time_scheme_order; ++j)
            {
               const auto Tnmj = T_prev[j].Read();
               dT_approx[dof] += beta[j] * Tnmj[i]; // beta_j * T_{n-j}
               // std::cout << " Tnmj[dof]: " << Tnmj[i] << endl;
               // out << " beta[j]: " << beta[j] << endl;
            }

            dT_approx[dof] /= dt; // Divide by dt
            // out << "dT_dt[dof]: " << dT_approx[dof] << endl;
            //  out << "\n" << flush;
         }
      }

      void HeatSolver::AddVolumetricTerm(Coefficient *coeff, Array<int> &attr)
      {
         op->AddVolumetricTerm(coeff, attr);

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

      void HeatSolver::AddVolumetricTerm(ScalarFuncT func, Array<int> &attr)
      {
         op->AddVolumetricTerm(func, attr);

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

      void HeatSolver::AddVolumetricTerm(Coefficient *coeff, int &attr)
      {
         // Create array for attributes and mark given mesh boundary
         tmp_domain_attr = 0;
         tmp_domain_attr[attr - 1] = 1;

         // Add the volumetric term to the operator
         AddVolumetricTerm(coeff, tmp_domain_attr);
      }

      void HeatSolver::AddVolumetricTerm(ScalarFuncT func, int &attr)
      {
         // Create array for attributes and mark given mesh boundary
         tmp_domain_attr = 0;
         tmp_domain_attr[attr - 1] = 1;

         // Add the volumetric term to the operator
         AddVolumetricTerm(func, tmp_domain_attr);
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
      HeatSolver::WriteFields(const int &it, const double &time)
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
         int offx = Ww + 10, offy = Wh + 45; // window offsets

         VisualizeField(*socks["T"], vishost, visport,
                        *T_gf, "Temperature (T)", Wx, Wy, Ww, Wh);
         Wx += offx;

         if (pmesh->GetMyRank() == 0 && verbose)
         {
            cout << " done." << endl;
         }
      }

      std::vector<double> HeatSolver::GetTimingData()
      {
         std::vector<double> timing_data(3);

         timing_data[0] = sw_init.RealTime();
         timing_data[1] = sw_setup.RealTime();
         timing_data[2] = sw_solve.RealTime();

         double rt_max[3];
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
         std::vector<double> timing_data = GetTimingData();

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
