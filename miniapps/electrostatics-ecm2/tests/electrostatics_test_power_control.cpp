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
//
//            -----------------------------------------------------
//            RF Miniapp:  Simple Electrostatics Simulation Code
//            -----------------------------------------------------
//
// This miniapp solves a simple 2D or 3D electrostatic problem (Quasi-static Maxwell).
//
//                            Div sigma Grad Phi = 0
//
// This specific test is designed to verify the power control algorithm.
// We imposed a (realistic) analytical temperature increase which modifies the
// conductivity distribution inside the domain. The solver should adjust the
// applied voltage at the Dirichlet boundaries to maintain a target power
// dissipation value.
//
// Sample runs:
//

#include <mfem.hpp>

#include "../lib/electrostatics_solver.hpp"
#include "../../common-ecm2/custom_coefficients.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip> // Include for std::setw

#include "../../common-ecm2/FilesystemHelper.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::electrostatics;
using namespace mfem::common_ecm2;

IdentityMatrixCoefficient *Id = NULL;

real_t CelsiusToKelvin(real_t Tc)
{
   return Tc + 273.15;
}

real_t KelvinToCelsius(real_t Tk)
{
   return Tk - 273.15;
}

int main(int argc, char *argv[])
{
   ////////////////////////////////////////////////////////////
   ////<--- 1. Initialize MPI and Hypre
   ////////////////////////////////////////////////////////////
   Mpi::Init(argc, argv);
   Hypre::Init();

   ////////////////////////////////////////////////////////////
   ////<--- 2. Parse command-line options.
   ////////////////////////////////////////////////////////////
   const char *mesh_file = "../multidomain/multidomain-hex.mesh";
   int order = 1;
   int serial_ref_levels = 1;
   int parallel_ref_levels = 0;
   int prec_type = 1;
   bool paraview = true;
   const char *outfolder = "./Output/Test/";
   bool pa = false;
   const char *device_config = "cpu";
   bool disable_hcurl_mass = false;
   real_t t_final = 5.0;
   real_t dt = 0.1;

   // Power control parameters
   real_t r = 10.0; // Rate of temperature increase (K/s) - increased for testing
   real_t target_power = 1.0;           // Target power dissipation (W)
   real_t power_tol = 1e-2;             // Relative tolerance for power control
   int max_power_iter = 20;             // Maximum iterations for power control

   Array<int> dbcs;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&disable_hcurl_mass, "-no-hcurl-mass", "--no-hcurl-mass",
                  "-with-hcurl-mass", "--with-hcurl-mass",
                  "Disable HCurl mass matrix assembly.");
   args.AddOption(&prec_type, "-prec", "--preconditioner",
                  "Preconditioner type (full assembly): 0 - BoomerAMG, 1 - LOR, \n"
                  "Preconditioner type (partial assembly): 0 - Jacobi smoother, 1 - LOR");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable Paraview visualization.");
   args.AddOption(&outfolder,
                  "-of",
                  "--output-folder",
                  "Output folder.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time for pseudo time-stepping.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step size for pseudo time-stepping.");
   args.AddOption(&r, "-r", "--temp-rate",
                  "Rate of temperature increase (K/s) for testing power control.");
   args.AddOption(&target_power, "-tp", "--target-power",
                  "Target power dissipation for power control (W).");
   args.AddOption(&power_tol, "-pt", "--power-tolerance",
                  "Relative tolerance for power control convergence.");
   args.AddOption(&max_power_iter, "-mpi", "--max-power-iter",
                  "Maximum iterations for power control.");
   args.ParseCheck();

   //    Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (Mpi::Root())
      device.Print();

   ////////////////////////////////////////////////////////////
   ////<--- 3. Read Mesh and create parallel
   ////////////////////////////////////////////////////////////
   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   StopWatch sw_initialization;
   sw_initialization.Start();
   if (Mpi::Root())
   {
      cout << "Starting initialization." << endl;
   }

   Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
   for (int l = 0; l < serial_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }

   // Extract submesh corresponding to the outer domain (tag 2)
   Array<int> box_domain_attributes(1);
   box_domain_attributes[0] = 2;
   Mesh *box_submesh = new SubMesh(SubMesh::CreateFromDomain(*serial_mesh, box_domain_attributes));
   auto pmesh = make_shared<ParMesh>(MPI_COMM_WORLD, *box_submesh);

   delete serial_mesh;
   delete box_submesh;

   int sdim = pmesh->SpaceDimension();

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = parallel_ref_levels;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }
   // Make sure tet-only meshes are marked for local refinement.
   pmesh->Finalize(true);

   /// 4. Set up conductivity coefficient   CHANGE
   int d = pmesh->Dimension();

   Coefficient *sigma_coeff = nullptr;

   real_t Tref = CelsiusToKelvin(37.0);
   // Define a temperature-dependent conductivity function (compatible with typical biological tissue behavior)
   // Geometry in cm, conductivity in S/cm, temperature in K
   // sigma increases by ~1.5% per degree increase
   real_t sigma_val = 0.005; // ~0.5 S/m typical for cardiac tissue
   auto sigma_func = [sigma_val, Tref](real_t T)
   {
      return sigma_val * (1.0 + 0.015 * (T - Tref));
   };
   sigma_coeff = new TemperatureDependentCoefficient(sigma_func);

   /// 5. Set up boundary conditions
   // Applied potential on central boundary, ground on outer boundaries, insulating on the rest

   // Create BCHandler and parse bcs
   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool verbose = true;
   if (Mpi::Root())
      mfem::out << "Creating BC handler..." << std::endl;

   BCHandler *bcs = new BCHandler(pmesh, verbose); // Boundary conditions handler

   // Boundary attribute IDs
   // Inner = 9, Lateral = 1, 2, 3, 4
   int inner_bdr = 9;
   
   Array<int> lateral_bdr(pmesh->bdr_attributes.Max());
   lateral_bdr = 0;
   lateral_bdr[0] = 1; 
   lateral_bdr[1] = 1;
   lateral_bdr[2] = 1;
   lateral_bdr[3] = 1;

   // Initial applied voltage (will be adjusted by power control)
   real_t initial_voltage = 1.0;

   // Create ConstantCoefficients for voltage BCs (we need to keep pointers to update them)
   // Note: These coefficients are owned by the BCHandler and will be deleted when BCHandler is deleted
   ConstantCoefficient *voltage_coeff = new ConstantCoefficient(initial_voltage);
   ConstantCoefficient *ground_coeff = new ConstantCoefficient(0.0);

   bcs->AddDirichletBC(voltage_coeff, inner_bdr);
   bcs->AddDirichletBC(ground_coeff, lateral_bdr);

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   /// 6. Create the Electrostatics Solver
   // Create the Electrostatic solver
   if (Mpi::Root())
      mfem::out << "Creating Electrostatics solver..." << std::endl;

   ElectrostaticsSolver RF_solver(pmesh, order, bcs, sigma_coeff);

   // Set Assembly Level (by default we use LEGACY assembly)
   if (pa)
      RF_solver.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   RF_solver.display_banner(std::cout);

   if (Mpi::Root())
   {
      if (!fs::is_directory(outfolder) || !fs::exists(outfolder))
      {                                     // Check if folder exists
         fs::create_directories(outfolder); // create folder
      }
   }

   if (Mpi::Root())
      mfem::out << "\nCreating DataCollection...";

   // Initialize Paraview visualization
   ParaViewDataCollection paraview_dc("RF_solver-Parallel", pmesh.get());
   paraview_dc.SetPrefixPath(outfolder);

   if (paraview)
   {
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetPrefixPath(outfolder);
      paraview_dc.SetLevelsOfDetail(order);
      RF_solver.RegisterParaviewFields(paraview_dc);
   }

   if (Mpi::Root())
      mfem::out << "done." << endl;

   sw_initialization.Stop();
   real_t my_rt[1], rt_max[1];
   my_rt[0] = sw_initialization.RealTime();
   MPI_Reduce(my_rt, rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   if (Mpi::Root())
   {
      cout << "Initialization done.  Elapsed time " << my_rt[0] << " s." << endl;
   }

   /// 7. Solve the problem
   if (Mpi::Root())
   {
      cout << "\nSolving... " << endl;
   }

   // Display the current number of DoFs in each finite element space
   RF_solver.PrintSizes();


   // Define the temperature increase function (analytical)
   ParFiniteElementSpace *fes = RF_solver.GetFESpace();
   
   ParGridFunction T_gf(fes);
   T_gf = 0.0;

   // We also define a grid function for the conductivity to output to Paraview
   ParGridFunction sigma_gf(fes);
   sigma_gf = 0.0;

   RF_solver.AddParaviewField("Temperature", &T_gf);
   RF_solver.AddParaviewField("Conductivity", &sigma_gf);

   // Setup solver and Assemble all forms
   int pl = 0;
   // RF_solver.EnablePA(pa);

   if (disable_hcurl_mass)
      RF_solver.DisableHCurlMass();

   // Define spatially uniform, time-dependent temperature increase
   // T(t) = T0 + r*t, where r is the rate of temperature increase (K/s)
   real_t T0 = CelsiusToKelvin(37.0);

   // Lambda function for time-dependent temperature: uniform in space, increasing linearly in time
   auto T_func = [T0, r](const Vector &x, real_t t)
   {
      return T0 + r * t;
   };

   // Create MFEM coefficient from the lambda function
   FunctionCoefficient *T_coeff = new FunctionCoefficient(T_func);

   ///<--- Initial Voltage Calibration ---<
   // Set initial temperature field (t=0)
   T_coeff->SetTime(0.0);
   T_gf.ProjectCoefficient(*T_coeff);

   if (Mpi::Root())
   {
      cout << "\n" << string(85, '=') << endl;
      cout << "Initial Voltage Calibration" << endl;
      cout << "Target Power: " << target_power << " W" << endl;
      cout << "Initial Temperature: " << T0 << " K (" << KelvinToCelsius(T0) << " °C)" << endl;
      cout << string(85, '=') << endl;
   }

   // Set the temperature grid function in the conductivity coefficient
   dynamic_cast<TemperatureDependentCoefficient *>(sigma_coeff)->SetGridFunction(&T_gf);
   RF_solver.Setup(prec_type, pl);

   // Calibrate initial voltage to achieve target power at initial temperature
   real_t current_voltage = initial_voltage;
   bool initial_converged = false;
   int initial_iter = 0;
   real_t initial_power = 0.0;
   real_t initial_rel_error = 0.0;

   for (initial_iter = 0; initial_iter < max_power_iter; initial_iter++)
   {
      ///<--- Update voltage BC
      voltage_coeff->constant = current_voltage;

      ///<--- Solve the system
      RF_solver.Solve();

      ///<--- Compute power dissipation
      initial_power = RF_solver.ElectricLosses();

      ///<--- Check convergence
      initial_rel_error = std::abs(initial_power - target_power) / target_power;

      if (initial_rel_error < 1e-5)
      {
         initial_converged = true;
         break;
      }

      ///<--- Update voltage using power scaling relationship (P ∝ V²)
      current_voltage = current_voltage * std::sqrt(target_power / initial_power);
      voltage_coeff->constant = current_voltage;
      RF_solver.Update(); // Update solver for next iteration
   }

   if (Mpi::Root())
   {
      cout << "\nCalibration Results:" << endl;
      cout << "  Iterations: " << (initial_iter + 1) << endl;
      cout << "  Calibrated Voltage: " << fixed << setprecision(6) << current_voltage << " V" << endl;
      cout << "  Achieved Power: " << scientific << setprecision(6) << initial_power << " W" << endl;
      cout << "  Relative Error: " << scientific << setprecision(2) << initial_rel_error << endl;
      if (!initial_converged)
      {
         cout << "  WARNING: Initial calibration did not converge!" << endl;
      }
      cout << string(85, '=') << endl;
   }

   // Pseudo time loop to test power control algorithm (the RF problem is steady state)
   real_t t = 0.0;

   if (Mpi::Root())
   {
      cout << "\nStarting Time-Dependent Power Control Test" << endl;
      cout << "Target Power: " << target_power << " W" << endl;
      cout << "Power Tolerance: " << power_tol * 100 << " %" << endl;
      cout << "Max Power Iterations: " << max_power_iter << endl;
      cout << string(85, '=') << endl;
      cout << "\n" << setw(8) << "Time"
           << setw(12) << "Temp (K)"
           << setw(12) << "Temp (°C)"
           << setw(14) << "Power (W)"
           << setw(14) << "Voltage (V)"
           << setw(12) << "PC Iters"
           << setw(13) << "Rel Error"
           << endl;
      cout << string(85, '-') << endl;
   }

   if (paraview)
   {
      RF_solver.WriteFields(0, 0.0);
   }

   bool last_step = false;
   for (int ti = 0; !last_step; ti++)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      ///<--- Update temperature field with current time
      T_coeff->SetTime(t);
      T_gf.ProjectCoefficient(*T_coeff);

      ///<--- Power Control Inner Loop 
      bool power_converged = false;
      int power_iter = 0;
      real_t computed_power = 0.0;
      real_t rel_error = 0.0;

      for (power_iter = 0; power_iter < max_power_iter; power_iter++)
      {
         ///<--- Flag that parameters have changed (for next assembly)
         RF_solver.Update();

         ///<--- Update voltage BC with current voltage value
         voltage_coeff->constant = current_voltage;

         ///<--- Check if power condition is already satisfied before solving
         if (power_iter == 0)
         {
            computed_power = RF_solver.ElectricLosses();
            rel_error = std::abs(computed_power - target_power) / target_power;
            if (rel_error < power_tol)
            {
               power_converged = true;
               break;
            }
            current_voltage = current_voltage * std::sqrt(target_power / computed_power);
            voltage_coeff->constant = current_voltage; // Update the voltage coefficient for the next iteration
         }

         ///<--- Solve the system
         RF_solver.Solve();

         ///<--- Compute power dissipation
         computed_power = RF_solver.ElectricLosses();

         ///<--- Check convergence
         rel_error = std::abs(computed_power - target_power) / target_power;

         if (rel_error < power_tol)
         {
            power_converged = true;
            break;
         }

         ///<--- Update voltage using power scaling relationship (P ∝ V²)
         //     V_new = V_old * sqrt(P_target / P_computed)
         current_voltage = current_voltage * std::sqrt(target_power / computed_power);
         voltage_coeff->constant = current_voltage; // Update the voltage coefficient for the next iteration
      }

      ///<--- Get average temperature for output
      real_t T_current = T0 + r * t;

      ///<--- Print results
      if (Mpi::Root())
      {
         cout << setw(8) << fixed << setprecision(2) << t
              << setw(12) << setprecision(3) << T_current
              << setw(12) << setprecision(3) << KelvinToCelsius(T_current)
              << setw(14) << scientific << setprecision(4) << computed_power
              << setw(14) << fixed << setprecision(6) << current_voltage
              << setw(12) << (power_iter + 1)
              << setw(13) << scientific << setprecision(2) << rel_error;

         if (!power_converged)
         {
            cout << " (NOT CONVERGED!)";
         }
         cout << endl;
      }

      ///<--- Increment time
      t += dt;

      ///<--- Write fields to disk for Paraview
      if (paraview)
      {
         RF_solver.WriteFields(ti+1, t);
      }

   }

   if (Mpi::Root())
   {
      cout << string(85, '=') << endl;
      cout << "Power Control Test Complete" << endl;
      cout << string(85, '=') << endl;
   }

   /// 8. Cleanup

   delete T_coeff;
   delete sigma_coeff;
   delete Id;

   return 0;
}
