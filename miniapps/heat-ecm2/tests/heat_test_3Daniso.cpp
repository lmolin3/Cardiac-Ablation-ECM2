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
// This example solves a 3D Heat Transfer problem with aniostropic conductivity in a beam geometry.
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
//
// Sample runs:
//
// 1. 2D conduction in rectangular plate
//    mpirun -np 10 ./heat_test_3Daniso -et 1 -p 1 -o 4 -rs 0 -rp 0 -ar 1.0 -ht 10000 -Tamb 500 -ode 4 -tf 1e5 -dt 1.0e2 --paraview -sf 10
//
// Test for partial assembly
//
// mpirun -np 4 ./heat_test_3Daniso --partial-assembly -et 1 -p 1 -o 6 -rs 0 -rp 0 -ar 1.0 -ht 10000 -Tamb 500 -ode 7 -tf 1e4 -dt 1.0e0 --paraview -sf 100
// mpirun -np 4 ./heat_test_3Daniso --no-partial-assembly -et 1 -p 1 -o 6 -rs 0 -rp 0 -ar 1.0 -ht 10000 -Tamb 500 -ode 7 -tf 1e4 -dt 1.0e0 --paraview -sf 100
//


#include "../lib/heat_solver.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/stat.h> // Include for mkdir

using namespace std;
using namespace mfem;
using namespace mfem::heat;

IdentityMatrixCoefficient *Id = NULL;

int left_bdry = 1;
int right_bdry = 2;
int side_bdry = 3;

real_t HeatingSphere(const Vector &x, real_t t);
real_t T_fun(const Vector &x, real_t t);

// Conductivity Matrix
void EulerAngles(const Vector &x, Vector &e);
std::function<void(const Vector &, DenseMatrix &)> ConductivityMatrix(const Vector &d);

// Volumetric heat
constexpr real_t Sphere_Radius = 0.5;
real_t Qval = 1.0e6; // Volumetric heat source (W/m^3)

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
   real_t h = 100.0;         // W/m^2K Heat transfer coefficient (problem 3/5)
   real_t T_amb_c = 100.0;   // Ambient temperature (°C) (problem 3/5)
   real_t aniso_ratio = 2.0; // Anisotropic ratio
   // Mesh
   int elem_type = 0; // 0 tet, 1 hex
   int order = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool pa = false; // Enable partial assembly
   // Time integrator
   int ode_solver_type = -1;
   real_t t_final = 100;
   real_t dt = 1.0e-2;
   // Postprocessing
   bool visit = false;
   bool paraview = true;
   int save_freq = 1; // Save fields every 'save_freq' time steps
   const char *outfolder = "";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&elem_type, "-et", "--elem-type",
                  "Element type: 0 - tet, 1 - hex.");
   args.AddOption(&h, "-ht", "--heat-transfer-coefficient",
                  "Heat transfer coefficient (W/m^2K).");
   args.AddOption(&T_amb_c, "-Tamb", "--ambient-temperature",
                  "Ambient temperature (°C).");
   args.AddOption(&Qval, "-Q", "--volumetric-heat-source",
                  "Volumetric heat source (W/m^3).");
   args.AddOption(&aniso_ratio, "-ar", "--anisotropic-ratio",
                  "Anisotropic ratio.");
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
   real_t T_amb = CelsiusToKelvin(T_amb_c);

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
   {
      // Load mesh from file
      if (elem_type == 0)
      {
         mesh = new Mesh("../../data/beam-tet.mesh");
      }
      else
      {
         mesh = new Mesh("../../data/beam-hex.mesh");
      }

      if (Mpi::Root())
      {
         cout << "Loading beam mesh \n"
              << endl;
      }
   }
   break;
   case 2:
   {
      mfem_error("Still not implemented");
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
   attr.Append(2);

   real_t kval, cval, rhoval;

   switch (problem)
   {
   case 1: // 2D conduction in rectangular plate
   case 2:
   {
      kval = 50.0;    // W/mK
      cval = 500.0;   // J/kgK
      rhoval = 800.0; // kg/m^3
   }
   break;
   default:
   {
      mfem_error("Unknown problem");
   }
   }

   // Heat Capacity
   Array<Coefficient *> coefs_c(0);
   coefs_c.Append(new ConstantCoefficient(cval));
   coefs_c.Append(new ConstantCoefficient(cval));
   PWCoefficient *c = new PWCoefficient(attr, coefs_c);

   // Density
   Array<Coefficient *> coefs_rho(0);
   coefs_rho.Append(new ConstantCoefficient(rhoval));
   coefs_rho.Append(new ConstantCoefficient(rhoval));
   PWCoefficient *rho = new PWCoefficient(attr, coefs_rho);

   // Conductivity
   Array<MatrixCoefficient *> coefs_k(0);

   for (int i = 0; i < attr.Size(); i++)
   {

      auto kFunc = [kval, aniso_ratio](const Vector &x, DenseMatrix &s)
      {
         s.SetSize(3);
         s = 0.0;
         real_t sx = aniso_ratio * kval;
         real_t sy = kval;
         real_t sz = kval;
         s(0, 0) = sx;
         s(1, 1) = sy;
         s(2, 2) = sz;
      };

      coefs_k.Append(new MatrixFunctionCoefficient(3, kFunc));
   };

   PWMatrixCoefficient *Kappa = new PWMatrixCoefficient(sdim, attr, coefs_k);

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

      // real_t Tside = CelsiusToKelvin(20.0); // Bottom, Right, Left sides
      real_t Tend = CelsiusToKelvin(20.0); // Bottom, Right, Left sides

      bcs->AddDirichletBC(Tend, right_bdry);
      // bcs->AddDirichletBC(T_fun, left_bdry);
      bcs->AddRobinBC(h, T_amb, left_bdry);
   }
   break;
   case 2: // 2D conduction in rectangular plate time dependent
   {
      mfem_error("Still not implemented");
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
      break;
   case 2:
   {
      mfem_error("Still not implemented");
   }
   break;
   default:
   {
      mfem_error("Unknown problem");
   }
   }

   // Create output folder
   if (outfolder[0] == '\0')
   {

      switch (problem)
      {
      case 1:
      {
         outfolder = "./Output/3D_Beam/";
      }
      break;
      case 2:
      {
         mfem_error("Still not implemented");
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
   real_t my_rt[1], rt_max[1];
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

   real_t t = 0.0;

   // Get reference to the temperature vector and gridfunction internal to Heat
   Vector &T = Heat.GetTemperature();
   ParGridFunction &T_gf = Heat.GetTemperatureGf();

   // Set the initial temperature
   real_t T0val;
   switch (problem)
   {
   case 1: // 2D conduction in rectangular plate
   case 2:
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

   StopWatch sw_solve;
   sw_solve.Start();

   real_t total_solve_time = 0.0;
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

      iterations++;
   }

   sw_solve.Stop();
   my_rt[0] = sw_solve.RealTime();
   MPI_Reduce(my_rt, rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   std::vector<real_t> timing_data = Heat.GetTimingData();

   if (Mpi::Root())
   {
      real_t average_solve_time = my_rt[0] / iterations;

      mfem::out << "Timing (s): " << std::setw(11) << "Init" << std::setw(13) << "Setup" << std::setw(18) << "Solve (tot)"
                << std::setw(18) << "Solve (avg)"
                << "\n";

      mfem::out << std::setprecision(3) << std::setw(25) << timing_data[0] << std::setw(11) << timing_data[1]
                << std::setw(15) << my_rt[0] << std::setw(20)
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

real_t T_fun(const Vector &x, real_t t)
{
   real_t T = 0.0;
   if (x.Size() == 3)
   {
      // T = sin(kappa x) sin(kappa y) + beta t
      T = (20 + 2.0 * t);
   }

   T = CelsiusToKelvin(T);

   return T > 373.15 ? 373.15 : T;
}

real_t HeatingSphere(const Vector &x, real_t t)
{
   real_t Q = 0.0;
   if (x.Size() == 3)
   {
      real_t r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Q = r < Sphere_Radius / 4.0 ? Qval : 0.0; // W/m^2
   }
   else if (x.Size() == 2)
   {
      real_t r = sqrt(x[0] * x[0] + x[1] * x[1]);
      Q = r < Sphere_Radius / 4.0 ? Qval : 0.0; // W/m^2
   }

   return Q;
}

std::function<void(const Vector &, DenseMatrix &)> ConductivityMatrix(const Vector &d)
{
   return [d](const Vector &x, DenseMatrix &m)
   {
      // Define dimension of problem
      const int dim = x.Size();

      // Compute Euler angles
      Vector e(3);
      EulerAngles(x, e);
      real_t e1 = e(0);
      real_t e2 = e(1);
      real_t e3 = e(2);

      // Compute rotated matrix
      if (dim == 3)
      {
         // Compute cosine and sine of the angles e1, e2, e3
         const real_t c1 = cos(e1);
         const real_t s1 = sin(e1);
         const real_t c2 = cos(e2);
         const real_t s2 = sin(e2);
         const real_t c3 = cos(e3);
         const real_t s3 = sin(e3);

         // Fill the rotation matrix R with the Euler angles.
         DenseMatrix R(3, 3);
         R(0, 0) = c1 * c3 - c2 * s1 * s3;
         R(0, 1) = -c1 * s3 - c2 * c3 * s1;
         R(0, 2) = s1 * s2;
         R(1, 0) = c3 * s1 + c1 * c2 * s3;
         R(1, 1) = c1 * c2 * c3 - s1 * s3;
         R(1, 2) = -c1 * s2;
         R(2, 0) = s2 * s3;
         R(2, 1) = c3 * s2;
         R(2, 2) = c2;

         // Multiply the rotation matrix R with the diffusivity vector.
         Vector l(3);
         l(0) = d[0];
         l(1) = d[1];
         l(2) = d[2];

         // Compute m = R^t diag(l) R
         R.Transpose();
         MultADBt(R, l, R, m);
      }
      else if (dim == 2)
      {
         const real_t c1 = cos(e1);
         const real_t s1 = sin(e1);
         DenseMatrix Rt(2, 2);
         Rt(0, 0) = c1;
         Rt(0, 1) = s1;
         Rt(1, 0) = -s1;
         Rt(1, 1) = c1;
         Vector l(2);
         l(0) = d[0];
         l(1) = d[1];
         MultADAt(Rt, l, m);
      }
      else
      {
         m(0, 0) = d[0];
      }
   };
}

void EulerAngles(const Vector &x, Vector &e)
{
   const int dim = x.Size();

   e(0) = 0.0 * M_PI / 180.0; // convert to radians
   e(1) = 0.0;
   if (dim == 3)
   {
      e(2) = 0.0;
   }
}