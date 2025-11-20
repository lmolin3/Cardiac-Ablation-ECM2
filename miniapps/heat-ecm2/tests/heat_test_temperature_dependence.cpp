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
//            ------------------------------------------
//            Heat Miniapp:  Fourier Heat transfer in 2D
//            ------------------------------------------
//
// This example solves a 2D/3D Heat Transfer problem
//
//                            rho c du/dt = Div k Grad T + Q
//
// with and external volumetric heat source typical of Radiofrequency Ablation (RFA) applications,
// and Temperature-dependent material properties.
//
// Q_s: external heat source term (W/m^3), e.g. Joule heating from RF electrode  Q = sigma |E|^2 --> analytic in this case
// Q_s is analytic in this case; for the coupled EM-Heat problem, see multidomain-ecm2 miniapp.
//
// The temperature dependence of the material properties is defined using 
// the classes
// Here dependence is linear, but more complex dependencies can be provided changing the function.
//
// Sample runs:
//

#include "../lib/heat_solver.hpp"
#include "custom_coefficients.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include "FilesystemHelper.hpp"

#pragma GCC diagnostic ignored "-Wunused-variable"

using namespace std;
using namespace mfem;
using namespace mfem::heat;
using namespace mfem::common_ecm2;

IdentityMatrixCoefficient *Id = NULL;

constexpr real_t Sphere_Radius = 0.5;

real_t scale = 1.0e-2; // scaling factor from cm to m

real_t Qval = 1.0e7; // W/m^3
real_t HeatingSphere(const Vector &x, real_t t);

int main(int argc, char *argv[])
{
   /////////////////////////////////////////////////////////////////////////////
   //------     1. Initialize MPI and HYPRE.
   /////////////////////////////////////////////////////////////////////////////

   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   /////////////////////////////////////////////////////////////////////////////
   //------     2. Parse command-line options.
   /////////////////////////////////////////////////////////////////////////////

   // Mesh
   int sdim = 3;
   int element_type = 1; // 0 - Tri/Tet, 1 - Quad/Hex
   int serial_ref_levels = 1;
   int parallel_ref_levels = 0;
   // Finite element space parameters
   int order = 3;
   bool pa = false; // Enable partial assembly
   // Time integrator
   int ode_solver_type = 21;
   real_t t_final = 100;
   real_t dt = 1.0e-2;
   // Problem
   bool temperature_dependent_properties = false;
   // Postprocessing
   bool visit = false;
   bool paraview = true;
   int save_freq = 1; // Save fields every 'save_freq' time steps
   const char *outfolder = "";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&element_type, "-et", "--element-type",
                  "Element type: 0 - triangles/tetrahedra, 1 - quadrilaterals/hexahedra.");
   args.AddOption(&sdim, "-d", "--dimension",
                  "Problem dimension: 2 - 2D, 3 - 3D.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&Qval, "-Q", "--volumetric-heat-source",
                  "Volumetric heat source (W/m^3).");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&temperature_dependent_properties, "-tdp", "--temp-dep-prop",
                  "-no-tdp", "--no-temp-dep-prop",
                  "Enable or disable temperature-dependent material properties.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&save_freq, "-sf", "--save-freq",
                  "Save fields every 'save_freq' time steps.");
   args.AddOption(&outfolder, "-of", "--out-folder",
                  "Output folder.");
   args.ParseCheck();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   //------     3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   StopWatch sw_initialization;
   sw_initialization.Start();
   if (Mpi::Root())
   {
      mfem::out << "Starting initialization." << endl;
   }

   //<--- Load serial mesh
   Mesh *mesh = nullptr;
   Element::Type type;

   switch (sdim)
   {
   case 2:
   {
      type = (element_type == 0) ? Element::TRIANGLE : Element::QUADRILATERAL;
      mesh = new Mesh(Mesh::MakeCartesian2D(4, 4, type, true, 4, 4, true));
   }
   break;
   case 3:
   {
      type = (element_type == 0) ? Element::TETRAHEDRON : Element::HEXAHEDRON;
      mesh = new Mesh(Mesh::MakeCartesian3D(4, 4, 2, type, 4, 4, 1, true));
   }
   break;
   default:
      mfem_error("Unknown dimension");
   }

   // Center the mesh at the origin
   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   *nodes -= 2.0; 
   *nodes *= scale; // scale to meters

   //<--- Refine the serial mesh on all processors to increase the resolution. I
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   //<--- Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution.
   auto pmesh = make_shared<ParMesh>(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int l = 0; l < parallel_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   //------     4. Set up coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Id = new IdentityMatrixCoefficient(sdim); // needed for conductivity definition

   MatrixCoefficient *Kappa = nullptr; // Conductivity
   Coefficient *c = nullptr;           // Heat Capacity
   Coefficient *rho = nullptr;         // Density

   // Physical parameters
   real_t c_val = 3600.0;   // J/kgK (typical tissue)
   real_t rho_val = 1060.0; // kg/m^3 (typical tissue)
   real_t k_val = 0.5;      // W/mK (typical tissue)

   if (!temperature_dependent_properties)
   {
      Kappa = new ScalarMatrixProductCoefficient(k_val, *Id);
      c = new ConstantCoefficient(c_val);
   }
   else
   {
      // Temperature-dependent (linear) material properties
      real_t Tref = CelsiusToKelvin(37.0);
      auto k_func = [k_val, Tref](real_t T, DenseMatrix &K)
      {
         int dim = K.Height();
         K = 0.0;
         for (int d = 0; d < dim; d++)
         {
            K(d, d) = k_val * (1.0 - 0.0005 * (T - Tref));
         }
      };
      Kappa = new TemperatureDependentMatrixCoefficient(sdim, k_func);

      auto c_func = [c_val, Tref](real_t T)
      {
         return c_val * (1.0 - 0.0042 * (T - Tref));
      };
      c = new TemperatureDependentCoefficient(c_func);
   }

   rho = new ConstantCoefficient(rho_val);


   ///////////////////////////////////////////////////////////////////////////////////////////////
   //------     5. Set up boundary conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create BCHandler and parse bcs
   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool verbose = true;
   BCHandler *bcs = new BCHandler(pmesh, verbose); // Boundary conditions handler

   // Apply boundary conditions on all external boundaries:
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   pmesh->MarkExternalBoundaries(ess_bdr);

   if (sdim == 3)
   {
      // Unmark top boundary
      ess_bdr[5] = 0;
   }

   // Add dirichlet bc T = T_body on all boundaries
   real_t T_body = CelsiusToKelvin(37.0);
   bcs->AddDirichletBC(T_body, ess_bdr);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   //------     6. Create the Heat Solver
   ///////////////////////////////////////////////////////////////////////////////////////////////

   //<--- Create the Heat solver
   HeatSolver Heat(pmesh, order, bcs, Kappa, c, rho, ode_solver_type);

   // Get reference to the temperature vector and gridfunction internal to Heat
   Vector &T = Heat.GetTemperature();
   ParGridFunction &T_gf = Heat.GetTemperatureGf();

   if (temperature_dependent_properties)
   {
      // Provide the temperature GridFunction to the temperature-dependent coefficients
      dynamic_cast<TemperatureDependentMatrixCoefficient *>(Kappa)->SetGridFunction(&T_gf);
      dynamic_cast<TemperatureDependentCoefficient *>(c)->SetGridFunction(&T_gf);
   }

   ParFiniteElementSpace *fes = Heat.GetFESpace();

   // Create a ParGridFunction to hold computed heat capacity (just for visualization)
   ParGridFunction c_gf(fes); c_gf = 0.0;

   // Display the current number of DoFs in each finite element space
   Heat.PrintSizes();

   //<--- Add volumetric heat source terms Q
   int domain_attr = 1;

   // External heat source term Q_s
   Heat.AddVolumetricTerm(HeatingSphere, domain_attr);

   //<--- Create output directory if it does not exist
   if (!fs::is_directory(outfolder) || !fs::exists(outfolder))
   {                                     // Check if folder exists
      fs::create_directories(outfolder); // create folder
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Heat-Parallel", pmesh.get());
   if (visit)
   {
      visit_dc.SetPrefixPath(outfolder);
      Heat.RegisterVisItFields(visit_dc);
   }

   // Initialize Paraview visualization
   ParaViewDataCollection paraview_dc("Heat-Parallel", pmesh.get());
   if (paraview)
   {
      paraview_dc.SetPrefixPath(outfolder);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetCompressionLevel(9);
      Heat.RegisterParaviewFields(paraview_dc);
   }

   paraview_dc.RegisterField("HeatCapacity", &c_gf);

   sw_initialization.Stop();
   real_t my_rt[1], rt_max[1];
   my_rt[0] = sw_initialization.RealTime();
   MPI_Reduce(my_rt, rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   if (Mpi::Root())
   {
      mfem::out << "Initialization done.  Elapsed time " << my_rt[0] << " s." << endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   //------     7. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Heat.EnablePA(pa);
   Heat.Setup();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   //------     8. Perform time-integration
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      mfem::out << "\n Time integration... " << endl;
   }

   real_t t = 0.0;

   // Set the initial temperature
   real_t T0val = CelsiusToKelvin(37.0);
   ConstantCoefficient T0(T0val);
   T_gf.ProjectCoefficient(T0);
   Heat.SetInitialTemperature(T_gf);

   c_gf.ProjectCoefficient(*c); // just for visualization

   // Write fields to disk for VisIt
   if (visit || paraview)
   {
      Heat.WriteFields(0, t);
   }

   real_t total_solve_time = 0.0;
   int iterations = 0;

   bool last_step = false;
   for (int step = 1; !last_step; step++)
   {
      if (Mpi::Root())
      {
         mfem::out << "\nSolving: step " << step << ", t = " << t << ", dt = " << dt << endl;
      }

      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      ///<--- Flag that parameters have changed (for next assembly)
      if (temperature_dependent_properties)
      {
         Heat.UpdatedParameters();
      }

      ///<--- Update the temperature-dependent heat capacity coefficient 
      c_gf.ProjectCoefficient(*c); // just for visualization
   
      ///<--- Step the heat solver
      Heat.Step(t, dt, step);

      ///<--- Write fields to disk for VisIt
      if ((visit || paraview) && (step % save_freq == 0))
      {
         Heat.WriteFields(step, t);
      }

      // Compute the total solve time
      std::vector<real_t> timing_data = Heat.GetTimingData();
      if (Mpi::Root())
      {
         total_solve_time += timing_data[2]; // Add the solve time to the total
         iterations = step;
      }
   }

   std::vector<real_t> timing_data = Heat.GetTimingData();

   if (Mpi::Root())
   {
      // Print timing summary
      mfem::out << std::endl;
      mfem::out << "-----------------------------------------------" << std::endl;
      mfem::out << "Timing Summary" << std::endl;
      mfem::out << "-----------------------------------------------" << std::endl;

      // Helper lambda to format time with appropriate units
      auto format_time = [](real_t time_s) -> std::string
      {
         if (time_s < 0.1)
         {
            return std::to_string(time_s * 1000.0) + " ms";
         }
         else
         {
            return std::to_string(time_s) + " s ";
         }
      };

      real_t t_init = timing_data[0];
      real_t t_setup = timing_data[1];
      real_t t_solve_total = total_solve_time;
      real_t t_solve_avg = total_solve_time / iterations;

      mfem::out << std::fixed << std::setprecision(3);
      mfem::out << std::setw(30) << std::left << "Initialization time:"
                << std::setw(15) << std::right << format_time(t_init) << std::endl;
      mfem::out << std::setw(30) << std::left << "Setup time:"
                << std::setw(15) << std::right << format_time(t_setup) << std::endl;
      mfem::out << std::setw(30) << std::left << "Total solution time:"
                << std::setw(15) << std::right << format_time(t_solve_total) << std::endl;
      mfem::out << std::setw(30) << std::left << "Average solution time:"
                << std::setw(15) << std::right << format_time(t_solve_avg) << std::endl;
      mfem::out << "-----------------------------------------------" << std::endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   //------     9. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Delete the MatrixCoefficient objects at the end of main
   delete Kappa;
   delete c;
   delete rho;

   delete Id;

   return 0;
}

// Sphere center is (0,0,-1) in 3D and (0,0) in 2D
real_t HeatingSphere(const Vector &x, real_t t)
{
   real_t Q = 0.0;
   real_t r = 0.0;
   real_t center_z = (x.Size() == 3) ? (x[2] + 1.0*scale) : 0.0;
   if (x.Size() == 3)
   {
      r = sqrt(x[0] * x[0] + x[1] * x[1] + center_z * center_z);
   }
   else if (x.Size() == 2)
   {
      r = sqrt(x[0] * x[0] + x[1] * x[1]);
   }

   // Smooth transition: width controls how "soft" the boundary is
   real_t width = 0.05*scale; // adjust for desired smoothness
   real_t transition = 0.5 * (1.0 - tanh((r - Sphere_Radius*scale) / width));
   Q = Qval * transition;

   return Q;
}

