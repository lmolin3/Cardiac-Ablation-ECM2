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
// Navier MMS example
//
// A manufactured solution is defined as
//
// u = [pi * sin(t) * sin(pi * x)^2 * sin(2 * pi * y),
//      -(pi * sin(t) * sin(2 * pi * x)) * sin(pi * y)^2].
//
// p = cos(pi * x) * sin(t) * sin(pi * y)
//
// The solution is used to compute the symbolic forcing term (right hand side),
// of the equation. Then the numerical solution is computed and compared to the
// exact manufactured solution to determine the error.
//
// Boundary markers for the square mesh are:
// Bottom=1, Right=2, Top=3, Left=4
//
// Run with:
// mpirun -np 4 ./navier_ecm2_mms_rk -ou 2 -op 1 --splitting-type 0 --rk-solver-type 0 -dt 1e-3 -tf 1e-2 -kv 0.01 -v -pv -f 3  -o ./Output/MMS/f3/ 
//

#include "lib/navierstokes_operator.hpp"
#include "lib/navierstokes_srk.hpp"
#include "bc/navierstokes_bchandler.hpp"
#include "../../common-ecm2/utils.hpp"
#include <fstream>
#include <sys/stat.h> // Include for mkdir

#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979
#endif

using namespace mfem;
using navier::BCHandler;

struct s_NavierContext // Navier Stokes params
{
   // FE
   int uorder = 2;
   int porder = 2;

   // rk
   int splitting_type = 0; // 0 = IMEX, 1 = Implicit, 2 = Explicit (TODO: implement explicit)
   int rk_solver_type = 0; // 0 = BEFE, 1 = IMEX Midpoint, 2 = DIRK_2_3_2, 3 = DIRK_2_2_2, 4 = DIRK_4_4_3

   double kinvis = 0.01;
   double a = 2.0;
   double dt = 1e-3;
   double t_final = 10 * dt;
   double gamma = 1.0;
   bool verbose = true;
   bool paraview = false;
   bool checkres = false;
   const char *outfolder = "./Output/MMS/Test/";
   int fun = 1;
   int bdf = 3;
   int bcs = 0; // 0 = FullyDirichlet, 1 = FullyNeumann, 2 = Mixed
} NS_ctx;

struct s_MeshContext // mesh
{
   int n = 10;
   int dim = 2;
   int elem = 0;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
} Mesh_ctx;

// Forward declarations of functions
void vel1(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = M_PI * sin(t) * pow(sin(M_PI * xi), 2.0) * sin(2.0 * M_PI * yi);
   u(1) = -(M_PI * sin(t) * sin(2.0 * M_PI * xi) * pow(sin(M_PI * yi), 2.0));
}

double p1(const Vector &x, double t)
{
   double xi = x(0);
   double yi = x(1);

   return sin(t) * cos(M_PI * xi) * sin(M_PI * yi);
}

void accel1(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = M_PI * pow(sin(M_PI * xi), 2.0) * sin(2 * M_PI * yi) * cos(t)                                                     // dudt
          - 2.0 * NS_ctx.kinvis * pow(M_PI, 3.0) * sin(2.0 * M_PI * yi) * sin(t) * (2 * cos(2.0 * M_PI * xi) - 1.0)         // - nu lap u
          - M_PI * sin(M_PI * xi) * sin(M_PI * yi) * sin(t)                                                                 // grad p
          + 4.0 * pow(M_PI, 3.0) * cos(M_PI * xi) * pow(sin(M_PI * xi), 3.0) * pow(sin(M_PI * yi), 2.0) * pow(sin(t), 2.0); // u grad u

   u(1) = -M_PI * pow(sin(M_PI * yi), 2.0) * sin(2 * M_PI * xi) * cos(t)                                                    // dudt
          + 2.0 * NS_ctx.kinvis * pow(M_PI, 3.0) * sin(2.0 * M_PI * xi) * sin(t) * (2 * cos(2.0 * M_PI * yi) - 1.0)         // - nu lap u
          + M_PI * cos(M_PI * xi) * cos(M_PI * yi) * sin(t)                                                                 // grad p
          + 4.0 * pow(M_PI, 3.0) * cos(M_PI * yi) * pow(sin(M_PI * yi), 3.0) * pow(sin(M_PI * xi), 2.0) * pow(sin(t), 2.0); // u grad u
}

// MMS2
void vel2(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = sin(xi) * sin(yi + t);
   u(1) = cos(xi) * cos(yi + t);
}

double p2(const Vector &x, double t)
{
   double xi = x(0);
   double yi = x(1);

   return cos(xi) * sin(yi + t);
}

void accel2(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = cos(t + yi) * sin(xi)               // dudt
          + 2.0 * NS_ctx.kinvis * sin(t + yi) // - nu lap u
          - sin(t + yi) * sin(xi)             // grad p
          + sin(2.0 * xi) / 2.0;              // u grad u

   u(1) = cos(t + yi) * cos(xi)                         // dudt
          + 2.0 * NS_ctx.kinvis * cos(t + yi) * cos(xi) // - nu lap u
          - sin(t + yi) * cos(xi)                       // grad p
          - sin(2.0 * t + 2.0 * yi) / 2.0;              // u grad u
}

// Kim & Moin
void vel3(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) = -cos(NS_ctx.a * M_PI * xi) * sin(NS_ctx.a * M_PI * yi) * std::exp(-2.0 * pow(NS_ctx.a, 2.0) * pow(M_PI, 2.0) * NS_ctx.kinvis * t);
   u(1) = sin(NS_ctx.a * M_PI * xi) * cos(NS_ctx.a * M_PI * yi) * std::exp(-2.0 * pow(NS_ctx.a, 2.0) * pow(M_PI, 2.0) * NS_ctx.kinvis * t);
}

double p3(const Vector &x, double t)
{
   double xi = x(0);
   double yi = x(1);

   return -1.0 / 4.0 * (cos(2.0 * NS_ctx.a * M_PI * xi) + cos(2.0 * NS_ctx.a * M_PI * yi)) * std::exp(-4.0 * pow(NS_ctx.a, 2.0) * pow(M_PI, 2.0) * NS_ctx.kinvis * t);
}

void accel3(const Vector &x, double t, Vector &u)
{
   u = 0.0;
}

/*void accel3(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   u(0) =   2.0*std::exp(-2.0*t)*cos(xi)*sin(yi)                    // dudt
          - 2.0 * NS_ctx.kinvis * std::exp(-2.0*t)*cos(xi)*sin(yi)  // - nu lap u
          + (sin(2.0*xi) * std::exp(-2.0*t))/2.0                    // grad p
          - (sin(2.0*xi) * std::exp(-4.0*t))/2.0;                   // u grad u

   u(1) =  -2.0*std::exp(-2.0*t)*cos(yi)*sin(xi)                    // dudt
          + 2.0 * NS_ctx.kinvis * std::exp(-2.0*t)*cos(yi)*sin(xi)  // - nu lap u
          + (sin(2.0*yi) * std::exp(-2.0*t))/2.0                    // grad p
          - (sin(2.0*yi) * std::exp(-4.0*t))/2.0;                   // u grad u
} */

Mesh CreateMesh()
{
   Element::Type type;
   switch (Mesh_ctx.elem)
   {
   case 0: // quad
      type = (Mesh_ctx.dim == 2) ? Element::QUADRILATERAL : Element::HEXAHEDRON;
      break;
   case 1: // tri
      type = (Mesh_ctx.dim == 2) ? Element::TRIANGLE : Element::TETRAHEDRON;
      break;
   }

   Mesh mesh;
   switch (Mesh_ctx.dim)
   {
   case 2: // 2d
      mesh = Mesh::MakeCartesian2D(Mesh_ctx.n, Mesh_ctx.n, type, true);
      break;
   case 3: // 3d
      mesh = Mesh::MakeCartesian3D(Mesh_ctx.n, Mesh_ctx.n, Mesh_ctx.n, type, true);
      break;
   }
   mesh.EnsureNodes();

   if (NS_ctx.fun == 1)
   {
      GridFunction *nodes = mesh.GetNodes();
      *nodes *= 2.0;
      *nodes -= 1.0;
   }

   for (int l = 0; l < Mesh_ctx.ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }   

   return mesh;
}




int main(int argc, char *argv[])
{

   //
   /// 1. Initialize MPI and Hypre.
   //

   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();




   //
   /// 2. Parse command-line options.
   //
   OptionsParser args(argc, argv);
   args.AddOption(&NS_ctx.uorder,
                  "-ou",
                  "--order-velocity",
                  "Order (degree) of the finite elements for velocity.");
   args.AddOption(&NS_ctx.porder,
                  "-op",
                  "--order-pressure",
                  "Order (degree) of the finite elements for pressure.");
   args.AddOption(&NS_ctx.splitting_type,
                  "-st", "--splitting-type",
                  "Type of splitting (0: IMEX, 1: Implicit, 2: Explicit)");
   args.AddOption(&NS_ctx.rk_solver_type,
                  "-rk", "--rk-solver-type",
                  "Type of Runge-Kutta solver (0: BEFE, 1: IMEX Midpoint, 2: DIRK_2_3_2, 3: DIRK_2_2_2, 4: DIRK_4_4_3)");
   args.AddOption(&NS_ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&NS_ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&NS_ctx.kinvis, "-kv", "--kinematic-viscosity", "Kinematic Viscosity.");
   args.AddOption(&NS_ctx.verbose,
                  "-v",
                  "--verbose",
                  "-no-v",
                  "--no-verbose",
                  "Enable verbosity.");
   args.AddOption(&NS_ctx.paraview,
                  "-pv",
                  "--paraview",
                  "-no-pv",
                  "--no-paraview",
                  "Enable or disable Paraview output.");
   args.AddOption(&NS_ctx.checkres,
                  "-cr",
                  "--checkresult",
                  "-no-cr",
                  "--no-checkresult",
                  "Enable or disable checking of the result. Returns -1 on failure.");
   args.AddOption(&NS_ctx.outfolder,
                  "-o",
                  "--output-folder",
                  "Output folder.");
   args.AddOption(&NS_ctx.fun, "-f", "--test-function",
                  "Analytic function to test");
   args.AddOption(&Mesh_ctx.dim,
                  "-d",
                  "--dimension",
                  "Dimension of the problem (2 = 2d, 3 = 3d)");
   args.AddOption(&Mesh_ctx.elem,
                  "-e",
                  "--element-type",
                  "Type of elements used (0: Quad/Hex, 1: Tri/Tet)");
   args.AddOption(&Mesh_ctx.n,
                  "-n",
                  "--num-elements",
                  "Number of elements in uniform mesh.");
   args.AddOption(&Mesh_ctx.ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&Mesh_ctx.par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.ParseCheck();





   //
   /// 3. Read the (serial) mesh from the given mesh file on all processors.
   //
   auto serial_mesh = CreateMesh();




   //
   /// 4. Define a parallel mesh by a partitioning of the serial mesh.
   // Refine this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   //
   auto pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   {
      for (int l = 0; l < Mesh_ctx.par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }




   //
   /// 5. Setup the SRK Solver:
   //  The solver requires the following objects:
   //   - BCHandler for boundary conditions (not necessarily populated with bcs)
   //   - NavierStokesOperator (different splitting types: IMEX, Implicit, Explicit)
   //   - NavierStokesSRKSolver for the time-stepping (different time stepping methods: BEFE, IMEX Midpoint, DIRK_2_3_2, DIRK_2_2_2, DIRK_4_4_3)
   //

   const int dim = pmesh->Dimension();
   H1_FECollection *ufec = new H1_FECollection(NS_ctx.uorder, dim);
   H1_FECollection *pfec = new H1_FECollection(NS_ctx.porder);
   ParFiniteElementSpace *ufes = new ParFiniteElementSpace(pmesh.get(), ufec, dim);
   ParFiniteElementSpace *pfes = new ParFiniteElementSpace(pmesh.get(), pfec, 1);

   HYPRE_BigInt global_velocity_tdofs = ufes->GlobalTrueVSize();
   HYPRE_BigInt global_pressure_tdofs = pfes->GlobalTrueVSize();
   if (Mpi::Root())
   {
      printf("# Velocity dofs: %d\n", global_velocity_tdofs);
      printf("# Pressure dofs: %d\n", global_pressure_tdofs);
   }

   // Create the objects for the solver
   auto bcs = std::make_shared<BCHandler>(pmesh, NS_ctx.verbose); // Boundary conditions handler
   NavierStokesOperator *navierOperator = nullptr;                // NavierStokesOperator
   NavierStokesSRKSolver *navierSolver = nullptr;                 // NavierStokesSRKSolver

   // NavierStokesOperator
   switch (NS_ctx.splitting_type)
   {
   case 0: // IMEX
   {
      navierOperator = new NavierStokesOperatorIMEX(pmesh, ufes, pfes, NS_ctx.kinvis, bcs, NS_ctx.verbose);
      break;
   }
   case 1: // Implicit
   {
      navierOperator = new NavierStokesOperatorImplicit(pmesh, ufes, pfes, NS_ctx.kinvis, bcs, NS_ctx.verbose);
      break;
   }
   }

   SolverParams params_p(1e-8, 1e-10, 1000, 0); // rtol, atol, maxiter, print-level
   SolverParams params_m(1e-8, 1e-10, 1000, 0); // rtol, atol, maxiter, print-level
   int pc_type = 3; // 0 = Pressure Mass, 1 = Pressure Laplacian, 2 = PCD, 3 = Cahouet-Chabard, 4 = Approximate inverse
   navierOperator->SetSolvers(params_p, params_m, pc_type);

   // Segregated Runge Kutta solver (SRK)
   bool ownOp = true; // The SRK solver will own the operator and delete it when done.
   navierSolver = new NavierStokesSRKSolver(navierOperator, NS_ctx.rk_solver_type, ownOp);




   //
   /// 6. Set initial condition and boundary conditions
   //

   // Set the initial condition.
   VectorFunctionCoefficient *uex_coeff = nullptr;
   VectorFunctionCoefficient *accelex_coeff = nullptr;
   FunctionCoefficient *pex_coeff = nullptr;

   switch (NS_ctx.fun)
   {
   case 1:
   {
      uex_coeff = new VectorFunctionCoefficient(pmesh->Dimension(), vel1);
      pex_coeff = new FunctionCoefficient(p1);
      accelex_coeff = new VectorFunctionCoefficient(pmesh->Dimension(), accel1);
      break;
   }
   case 2:
   {
      uex_coeff = new VectorFunctionCoefficient(pmesh->Dimension(), vel2);
      pex_coeff = new FunctionCoefficient(p2);
      accelex_coeff = new VectorFunctionCoefficient(pmesh->Dimension(), accel2);
      break;
   }
   case 3:
   {
      uex_coeff = new VectorFunctionCoefficient(pmesh->Dimension(), vel3);
      pex_coeff = new FunctionCoefficient(p3);
      accelex_coeff = new VectorFunctionCoefficient(pmesh->Dimension(), accel3);
      break;
   }
   default:
      break;
   }

   double t = 0.0;
   uex_coeff->SetTime(t);
   pex_coeff->SetTime(t);
   accelex_coeff->SetTime(t);

   ParGridFunction u_gf(ufes), uex_gf(ufes), p_gf(pfes), pex_gf(pfes), rhs_gf(ufes);

   BlockVector X(navierOperator->GetOffsets());
   X = 0.0;
   u_gf.ProjectCoefficient(*uex_coeff);
   p_gf.ProjectCoefficient(*pex_coeff);
   uex_gf.ProjectCoefficient(*uex_coeff);
   pex_gf.ProjectCoefficient(*pex_coeff);

   u_gf.ParallelProject(X.GetBlock(0));
   p_gf.ParallelProject(X.GetBlock(1));

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   // Bottom=1, Right=2, Top=3, Left=4
   int bottom_attr = 1;
   int right_attr = 2;
   int top_attr = 3;
   int left_attr = 4;


   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 1;
   bcs->AddVelDirichletBC(uex_coeff, ess_attr);

   // Add acceleration term to the NavierStokesOperator
   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1;
   navierOperator->AddAccelTerm(accelex_coeff, domain_attr);

   // Setup the NavierStokesOperator
   navierOperator->Setup(NS_ctx.dt);

   ParGridFunction uerr_gf(ufes), perr_gf(pfes);
   uerr_gf = 0.0;
   perr_gf = 0.0;

   // Creating output directory if not existent
   ParaViewDataCollection *paraview_dc = nullptr;

   auto save_callback = [&](int cycle, double t)
   {
      paraview_dc->SetCycle(cycle);
      paraview_dc->SetTime(t);

      for (int i = 0; i < uerr_gf.Size(); i++)
      {
         uerr_gf(i) = abs(u_gf(i) - uex_gf(i));
      }

      for (int i = 0; i < perr_gf.Size(); i++)
      {
         perr_gf(i) = abs(p_gf(i) - pex_gf(i));
      }

      paraview_dc->Save();
   };

   if (NS_ctx.paraview)
   {
      if ((mkdir(NS_ctx.outfolder, 0777) == -1) && (pmesh->GetMyRank() == 0))
      {
         mfem::err << "Error :  " << strerror(errno) << std::endl;
      }

      paraview_dc = new ParaViewDataCollection("Results-Paraview", pmesh.get());
      paraview_dc->SetPrefixPath(NS_ctx.outfolder);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->RegisterField("pressure", &p_gf);
      paraview_dc->RegisterField("velocity", &u_gf);
      paraview_dc->RegisterField("exact_rhs", &rhs_gf);
      paraview_dc->RegisterField("exact_pressure", &pex_gf);
      paraview_dc->RegisterField("exact_velocity", &uex_gf);
      paraview_dc->RegisterField("error_pressure", &perr_gf);
      paraview_dc->RegisterField("error_velocity", &uerr_gf);

      uex_coeff->SetTime(t);
      pex_coeff->SetTime(t);
      accelex_coeff->SetTime(t);

      rhs_gf.ProjectCoefficient(*accelex_coeff);
      uex_gf.ProjectCoefficient(*uex_coeff);
      pex_gf.ProjectCoefficient(*pex_coeff);

      save_callback(0, t);
   }




   //
   /// 7. Solve unsteady problem
   //
   double dt = NS_ctx.dt;
   bool last_step = false;

   for (int step = 0; !last_step; ++step)
   {
      // Check if the final time has been reached
      if (t + dt >= NS_ctx.t_final - dt / 2)
      {
         last_step = true;
      }

      if (Mpi::Root())
      {
         std::cout << "Step " << std::left << std::setw(5) << step
                   << std::setprecision(2) << std::scientific
                   << " t = " << t
                   << " dt = " << dt
                   << "\n"
                   << std::endl;
      }

      // Solve current step
      navierSolver->Step(X, t, dt);

      // Extract the velocity and pressure from the solution
      u_gf.Distribute(X.GetBlock(0));
      p_gf.Distribute(X.GetBlock(1));

      // Compute errors
      uex_coeff->SetTime(t);
      uex_gf.ProjectCoefficient(*uex_coeff);
      double vel_l2_err = u_gf.ComputeL2Error(*uex_coeff);

      pex_coeff->SetTime(t);
      pex_gf.ProjectCoefficient(*pex_coeff);
      double pres_l2_err = p_gf.ComputeL2Error(*pex_coeff);

      // Compute incompressiblity error
      double div_u_l2 = 0.0;
      {
         double div_u_l2_local = 0.0;
         Vector grad(2);
         for (int e = 0; e < pmesh->GetNE(); e++)
         {
            auto fe = ufes->GetFE(e);
            auto T = ufes->GetElementTransformation(e);
            int intorder = 2 * fe->GetOrder();
            const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);
            for (int qp = 0; qp < ir->GetNPoints(); qp++)
            {
               const IntegrationPoint &ip = ir->IntPoint(qp);
               T->SetIntPoint(&ip);
               u_gf.GetGradient(*T, grad);
               div_u_l2_local += ip.weight * (grad(0) + grad(1));
            }
         }
         MPI_Allreduce(&div_u_l2_local,
                       &div_u_l2,
                       1,
                       MPI_DOUBLE,
                       MPI_SUM,
                       MPI_COMM_WORLD);
      }

      if (Mpi::Root())
      {
         printf("u_l2err = %.5E\np_l2err = %.5E\ndiv(u) = %.5E\n", vel_l2_err,
                pres_l2_err, div_u_l2);
      }

      save_callback(step, t);

      if (Mpi::Root())
      {
         std::cout << std::endl;
      }
   }




   // Free the used memory
   delete paraview_dc;
   delete uex_coeff;
   delete pex_coeff;
   delete accelex_coeff;
   delete navierSolver;

   delete ufec;
   delete pfec;
   delete ufes;
   delete pfes;

   return 0;
}
