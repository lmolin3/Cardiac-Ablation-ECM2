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
// This example solves a 2D Heat Transfer problem
//
//                            rho c du/dt = Div k Grad T + Q
//
// rho and c are piecewise coefficients,
// k is a piecewise matrix coefficient (isotropic conductivity)
//
// Boundary conditions consist in:
// * Dirichlet: temperature
// * Neumann: heat flux
// * Robin: convective bc
//
// We discretize the temperature with H1 finite elements.
//
// Temperature in Kelvin, Conductivity in W/mK
//
// Problems:
//
// 1. 2D conduction in rectangular plate
// 2. 2D conduction in rectangular plate with time-dependent boundary condition
// 3. 3D Sphere heating
// 4. 3D Sphere with volumetric source and convective cooling
//
//
// Sample runs:
//
// 1. 2D conduction in rectangular plate 
//    mpirun -np 4 ./heat_test -p 1 -o 1 -et 0 -rs 0 -rp 0 -ode 1 -tf 0.5 -dt 1.0e-2 --paraview -sf 10
//
// 2. 2D conduction in rectangular plate with time-dependent boundary condition
//    mpirun -np 4 ./heat_test -p 2 -o 1 -et 0 -rs 0 -rp 0 -ode 1 -tf 0.5 -dt 1.0e-2 --paraview -sf 10
//
// 3. 3D Sphere heating 
//   mpirun -np 4 ./heat_test -p 3 -o 1 -rs 0 -rp 0 -ode 1 -tf 10.0 -dt 1.0e-2 -ht 300.0 -Tamb 100.0 --paraview -sf 10
//
// 4. 3D Sphere with volumetric source and convective cooling
//   mpirun -np 4 ./heat_test -p 4 -o 1 -rs 0 -rp 0 -ode 1 -tf 10.0 -dt 1.0e-2 -ht 300.0 -Tamb 0.0 -Q 1e7 --paraview -sf 10
//

#include "lib/heat_solver.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/stat.h> // Include for mkdir

using namespace std;
using namespace mfem;
using namespace mfem::heat;

IdentityMatrixCoefficient *Id = NULL;

constexpr double Sphere_Radius = 0.1;
double Qval = 1.0e6; // W/m^3

double HeatingSphere(const Vector &x, double t);
double T_side(const Vector &x, double t);

int main(int argc, char *argv[])
{
   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mpi::Init(argc, argv);
   Hypre::Init();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options.
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Problem
   int problem = 1;
   double h = 100.0;       // W/m^2K Heat transfer coefficient (problem 3/5)
   double T_amb_c = 100.0; // Ambient temperature (°C) (problem 3/5)
   // Mesh
   int element_type = 0; // 0 - Tri, 1 - Quad
   int order = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool pa = false; // Enable partial assembly
   // Time integrator
   int ode_solver_type = -1;
   double t_final = 100;
   double dt = 1.0e-2;
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
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                  "Enable or disable partial assembly.");   
   args.AddOption(&h, "-ht", "--heat-transfer-coefficient",
                  "Heat transfer coefficient (W/m^2K).");
   args.AddOption(&T_amb_c, "-Tamb", "--ambient-temperature",
                  "Ambient temperature (°C).");
   args.AddOption(&Qval, "-Q", "--volumetric-heat-source",
                  "Volumetric heat source (W/m^3).");
   args.AddOption(&ode_solver_type, "-ode", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   4 - Implicit Midpoint, 5 - SDIRK23, 6 - SDIRK34,\n\t"
                  "\t   7 - Forward Euler, 8 - RK2, 9 - RK3 SSP, 10 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem: 1 - 2D conduction in rectangular plate, 2 - 2D conduction in a disk, 3 - 3D Sphere heating, 4 - 3D Cylinder heating.");
   args.AddOption(&save_freq, "-sf", "--save-freq",
                  "Save fields every 'save_freq' time steps.");
   args.AddOption(&outfolder, "-of", "--out-folder",
                  "Output folder.");

   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   // Convert ambient temperature to Kelvin
   double T_amb = CelsiusToKelvin(T_amb_c);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   Mesh *mesh = nullptr;
   Element::Type type;

   switch (problem)
   {
   case 1: // 2D conduction in rectangular plate L = 0.83m, H = 0.833m
   case 2:
   {
      type = (element_type == 0) ? Element::TRIANGLE : Element::QUADRILATERAL;
      mesh = new Mesh(Mesh::MakeCartesian2D(5, 5, type, 0.83, 0.833, true));
      if (Mpi::Root())
      {
         cout << "Solving Problem 1: 2D conduction in rectangular plate \n"
              << endl;
      }
   }
   break;
   case 3: // 3D Sphere heating
   case 4:
   {
      // Load mesh from file
      mesh = new Mesh("../../data/sphere.msh");
      if (Mpi::Root())
      {
         cout << "Solving Problem 3: 3D Sphere heating \n"
              << endl;
      }
   }
   break;
   default:
      mfem_error("Unknown problem");
   }

   int sdim = mesh->SpaceDimension();

   StopWatch sw_initialization;
   sw_initialization.Start();
   if (Mpi::Root())
   {
      cout << "Starting initialization." << endl;
   }

   // Ensure that quad and hex meshes are treated as non-conforming.
   mesh->EnsureNCMesh();

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. NURBS meshes are
   // refined at least twice, as they are typically coarse.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   auto pmesh = make_shared<ParMesh>(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = parallel_ref_levels;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }
   // Make sure tet-only meshes are marked for local refinement.
   pmesh->Finalize(true);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Set up coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Id = new IdentityMatrixCoefficient(sdim);
   Array<int> attr(0);
   attr.Append(1);

   double kval, cval, rhoval;

   switch (problem)
   {
   case 1: // 2D conduction in rectangular plate
   case 2:
   {
      kval = 81.0;  // W/mK
      cval = 1.0;   // J/kgK
      rhoval = 1.0; // kg/m^3
   }
   break;
   case 3: // 3D Sphere heating
   case 4:
   {
      kval = 50.0;     // W/mK
      cval = 5000.0;    // J/kgK
      rhoval = 8000.0; // kg/m^3
   }
   break;
   default:
   {
      mfem_error("Unknown problem");
   }
   }

   // Conductivity
   Array<MatrixCoefficient *> coefs_k(0);
   coefs_k.Append(new ScalarMatrixProductCoefficient(kval, *Id));
   PWMatrixCoefficient *Kappa = new PWMatrixCoefficient(sdim, attr, coefs_k);

   // Heat Capacity
   Array<Coefficient *> coefs_c(0);
   coefs_c.Append(new ConstantCoefficient(cval));
   PWCoefficient *c = new PWCoefficient(attr, coefs_c);

   // Density
   Array<Coefficient *> coefs_rho(0);
   coefs_rho.Append(new ConstantCoefficient(rhoval));
   PWCoefficient *rho = new PWCoefficient(attr, coefs_rho);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Set up boundary conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create BCHandler and parse bcs
   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool verbose = true;
   BCHandler *bcs = new BCHandler(pmesh, verbose); // Boundary conditions handler

   switch (problem)
   {
   case 1: // 2D conduction in rectangular plate (bottom = 1, right = 2, top = 3, left = 4)
   {
      int top = 3;
      Array<int> dbcs(pmesh->bdr_attributes.Max());
      dbcs = 1;
      dbcs[top - 1] = 0;

      double T2 = CelsiusToKelvin(20.0);  // Top side
      double T1 = CelsiusToKelvin(100.0); // Bottom, Right, Left sides

      bcs->AddDirichletBC(T2, top);
      bcs->AddDirichletBC(T1, dbcs);
   }
   break;
   case 2: // 2D conduction in rectangular plate time dependent
   {
      int top = 3;
      Array<int> dbcs(pmesh->bdr_attributes.Max());
      dbcs = 1;
      dbcs[top - 1] = 0;

      double T2 = CelsiusToKelvin(20.0); // Top side

      bcs->AddDirichletBC(T2, top);
      bcs->AddDirichletBC(T_side, dbcs);
   }
   break;
   case 3: // 3D Sphere heating
   {
      int outer_surface = 1;

      bcs->AddRobinBC(h, T_amb, outer_surface);
   }
   break;
   case 4: // 3D Sphere heating
   {
      int outer_surface = 1;

      bcs->AddRobinBC(h, T_amb, outer_surface);
   }
   break;
   default:
   {
      mfem_error("Unknown problem");
   }
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Create the Heat Solver
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create the Heat solver
   HeatSolver Heat(pmesh, order, bcs, Kappa, c, rho, ode_solver_type);

   // Display the current number of DoFs in each finite element space
   Heat.PrintSizes();

   // Adding volumetric heat source
   switch (problem)
   {
   case 1:
   case 2:
   case 3:
      break;
   case 4:
   {
      // Add volumetric heat source
      int domain_attr = 1;
      Heat.AddVolumetricTerm(HeatingSphere, domain_attr);
   }
   break;
   default:
   {
      mfem_error("Unknown problem");
   }
   }

   // Create output folder
   // 1. 2D conduction in rectangular plate
   // 2. 2D conduction in a disk
   // 3. 3D Sphere heating
   // 4. 3D Cylinder heating
   if (outfolder[0] == '\0')
   {

      switch (problem)
      {
      case 1:
      {
         outfolder = "./Output/2D_RectangularPlate/";
      }
      break;
      case 2:
      {
         outfolder = "./Output/2D_RectangularPlate_TimeDependent/";
      }
      break;
      case 3:
      {
         outfolder = "./Output/3D_Sphere/";
      }
      break;
      case 6:
      {
         outfolder = "./Output/3D_Sphere_VolumetricHeat/";
      }
      break;
      default:
      {
         mfem_error("Unknown problem");
      }
      }
   }

   if ((mkdir(outfolder, 0777) == -1) && Mpi::Root())
   {
      mfem::err << "Error :  " << strerror(errno) << std::endl;
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

   sw_initialization.Stop();
   double my_rt[1], rt_max[1];
   my_rt[0] = sw_initialization.RealTime();
   MPI_Reduce(my_rt, rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   if (Mpi::Root())
   {
      cout << "Initialization done.  Elapsed time " << my_rt[0] << " s." << endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Heat.EnablePA(pa);
   Heat.Setup();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Perform time-integration (looping over the time iterations, step, with a
   //     time-step dt).
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      cout << "\n Time integration... " << endl;
   }

   double t = 0.0;

   // Get reference to the temperature vector and gridfunction internal to Heat
   Vector &T = Heat.GetTemperature();
   ParGridFunction &T_gf = Heat.GetTemperatureGf();

   // Set the initial temperature
   double T0val;
   switch (problem)
   {
   case 1: // 2D conduction in rectangular plate
   case 2:
   {
      T0val = CelsiusToKelvin(20.0);
   }
   break;
   case 3: // 3D Sphere heating
   case 4:
   {
      T0val = CelsiusToKelvin(20.0);
   }
   break;
   default:
   {
      mfem_error("Unknown problem");
   }
   }

   ConstantCoefficient T0(T0val);
   T_gf.ProjectCoefficient(T0);
   Heat.SetInitialTemperature(T_gf);

   // Write fields to disk for VisIt
   if (visit || paraview)
   {
      Heat.WriteFields(0, t);
   }

   double total_solve_time = 0.0;
   int iterations = 0;

   bool last_step = false;
   for (int step = 1; !last_step; step++)
   {
      if (Mpi::Root())
      {
         cout << "\nSolving: step " << step << ", t = " << t << ", dt = " << dt << endl;
      }

      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      Heat.Step(t, dt, step);

      // Write fields to disk for VisIt
      if ((visit || paraview) && (step % save_freq == 0))
      {
         Heat.WriteFields(step, t);
      }

      // Compute the total solve time
      std::vector<double> timing_data = Heat.GetTimingData();
      if (Mpi::Root())
      {
         total_solve_time += timing_data[2]; // Add the solve time to the total
         iterations = step;
      }
   }

   std::vector<double> timing_data = Heat.GetTimingData();

   if (Mpi::Root())
   {
      double average_solve_time = total_solve_time / iterations;

      mfem::out << "Timing (s): " << std::setw(11) << "Init" << std::setw(13) << "Setup" << std::setw(18) << "Solve (tot)"
                << std::setw(18) << "Solve (avg)"
                << "\n";

      mfem::out << std::setprecision(3) << std::setw(25) << timing_data[0] << std::setw(11) << timing_data[1]
                << std::setw(15) << total_solve_time << std::setw(20)
                << average_solve_time << "\n";

      mfem::out << std::setprecision(8);
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Delete the MatrixCoefficient objects at the end of main
   for (int i = 0; i < coefs_k.Size(); i++)
   {
      delete coefs_k[i];
   }

   for (int i = 0; i < coefs_c.Size(); i++)
   {
      delete coefs_c[i];
   }

   for (int i = 0; i < coefs_rho.Size(); i++)
   {
      delete coefs_rho[i];
   }

   delete Kappa;
   delete c;
   delete rho;

   delete Id;

   return 0;
}

double T_side(const Vector &x, double t)
{
   double T = 0.0;
   if (x.Size() == 2)
   {
      // T = sin(kappa x) sin(kappa y) + beta t
      T = (20 + 2.0 * t);
   }

   T = CelsiusToKelvin(T);

   return T > 373.15 ? 373.15 : T;
}

double HeatingSphere(const Vector &x, double t)
{
   double Q = 0.0;
   if (x.Size() == 3)
   {
      double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Q = r < Sphere_Radius / 4.0 ? Qval : 0.0; // W/m^2
   }
   else if (x.Size() == 2)
   {
      double r = sqrt(x[0] * x[0] + x[1] * x[1]);
      Q = r < Sphere_Radius / 4.0 ? Qval : 0.0; // W/m^2
   }

   return Q;
}