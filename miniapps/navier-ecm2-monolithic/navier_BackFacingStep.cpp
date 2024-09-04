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
// Flow around cylinder in 2D/3D
//
// The problem domain is set up like this
//
//                                  no slip
//              |\    + ---------------------------------- +
// Parabolic -->  |   |                                    |
//  inflow      |/    + ------ +                           |  Traction free (outflow)
//                             |                           |
//                             + ------------------------- +
//                                      no slip
//
// Mesh attributes for 2D/3D are:
// inflow = 1, outflow = 2, wall = 3
//
// Run with:
// mpirun -np 4 ./navier_ecm2_backfacingstep -d 2 -rs 0 -rp 0 -ou 2 -op 1 -dt 1e-3 -tf 1e-1 -tp 1e-2 -kv 0.01 -u 1.0 --gamma 1.0 --verbose --paraview --output-folder ./Output/BackFacingStep/2D/Test
//

   
#include "navier_unsteady_solver.hpp"
//#include "utils.hpp"
#include <fstream>
#include <sys/stat.h>  // Include for mkdir

#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979
#endif

using namespace mfem;

struct s_MeshContext // mesh
{
   int dim = 2;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int D = 1.0;  // Channel diameter
} Mesh_ctx;


struct s_NavierContext // Navier Stokes params
{
   int uorder = 2;
   int porder = 1;
   double kinvis = 0.01;
   double Umax =  1.0;
   double dt = 1e-3;
   double t_final = 10 * dt;
   double preloadT = 0.1 * t_final;
   double gamma = 1.0;
   bool verbose = true;
   bool paraview = false;
   const char *outfolder = "./Output/BackFacingStep/2D/Test/";
   bool ExportData = false;
   int bdf = 3;
} NS_ctx;


// Forward declarations of functions
void inflow(const Vector &x, double t, Vector &u);
void noSlip(const Vector &x, double t, Vector &u);
double pZero(const Vector &x, double t);

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
   SolverParams sParams(1e-6, 1e-8, 1000, 0); // rtol, atol, maxiter, print-level


   OptionsParser args(argc, argv);
   args.AddOption(&NS_ctx.uorder,
                  "-ou",
                  "--order-velocity",
                  "Order (degree) of the finite elements for velocity.");
   args.AddOption(&NS_ctx.porder,
                  "-op",
                  "--order-pressure",
                  "Order (degree) of the finite elements for pressure.");
   args.AddOption(&NS_ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&NS_ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&NS_ctx.preloadT, "-tp", "--preload-time", "Preload time.");
   args.AddOption(&NS_ctx.kinvis, "-kv", "--kinematic-viscosity", "Kinematic Viscosity.");
   args.AddOption(&NS_ctx.Umax, "-u", "--inflow-velocity",
                   "Inflow velocity");
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
    args.AddOption(&NS_ctx.outfolder,
                   "-o",
                   "--output-folder",
                   "Output folder.");
   args.AddOption(&NS_ctx.ExportData,
                  "-ed",
                  "--export-data",
                  "-no-ed",
                  "--no-export-data",
                  "Enable or disable output of matrices/vectors (debug only, false by default).");
    args.AddOption(&NS_ctx.gamma,
                   "-g",
                   "--gamma",
                   "Relaxation parameter");
    args.AddOption(&NS_ctx.bdf,
                   "-bdf",
                   "--bdf-order",
                   "Maximum bdf order (1<=bdf<=3)");

    args.AddOption(&Mesh_ctx.dim,
                   "-d",
                   "--dimension",
                   "Dimension of the problem (2 = 2d, 3 = 3d)");
   args.AddOption(&Mesh_ctx.ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&Mesh_ctx.par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }


   //
   /// 3. Read the (serial) mesh from the given mesh file on all processors.
   //
   Mesh mesh;

   switch (Mesh_ctx.dim)
   {
   case 2:
   {
      mesh = Mesh::LoadFromFile("./Mesh/back_facing_step_2D.msh");
      break;
   }
   case 3:
   {
      mesh = Mesh::LoadFromFile("./Mesh/back_facing_step_3D.msh");
      break;
   }
   default:
      break;
   }

   for (int l = 0; l < Mesh_ctx.ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }


   //
   /// 4. Define a parallel mesh by a partitioning of the serial mesh.
   // Refine this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   //
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < Mesh_ctx.par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }


   //
   /// 5. Create the flow solver.
   //
   NavierUnsteadySolver naviersolver(pmesh, NS_ctx.uorder, NS_ctx.porder, NS_ctx.kinvis, NS_ctx.verbose);

   naviersolver.SetSolver(sParams);
   naviersolver.SetMaxBDFOrder(NS_ctx.bdf);


   QuantitiesOfInterest qt(pmesh);


#ifdef MFEM_DEBUG
   naviersolver.SetExportData( NS_ctx.ExportData ); // Export matrices/vectors 
#endif

   //
   /// 6. Set initial condition and boundary conditions
   //

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.

   if (NS_ctx.verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Kinematic viscosity: " << NS_ctx.kinvis << std::endl;
      mfem::out << "Max velocity: " << NS_ctx.Umax << std::endl;
      mfem::out << std::endl;
   }

   int inflow_attr = 1;
   int outflow_attr = 2;
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 1;                    // Mark walls/cylinder for no-slip condition
   ess_attr[inflow_attr - 1]  = 0;
   ess_attr[outflow_attr - 1] = 0;

   // Inflow
   naviersolver.AddVelDirichletBC(inflow,inflow_attr);

   // No Slip
   naviersolver.AddVelDirichletBC(noSlip,ess_attr);



   // initial condition
   //naviersolver.SetInitialConditionVel( *u_in );
   //naviersolver.SetInitialConditionPrevVel( *u_in );

   //
   /// 7. Setup
   //

   double t = 0.0;
   double dt = NS_ctx.dt;
   bool last_step = false;

   naviersolver.SetOutputFolder(NS_ctx.outfolder);
   naviersolver.Setup(dt);
   naviersolver.SetGamma(NS_ctx.gamma);
   ParGridFunction u_gf(naviersolver.GetUFes());
   ParGridFunction p_gf(naviersolver.GetPFes());

   // Creating output directory if not existent
   ParaViewDataCollection *paraview_dc = nullptr;

   
   if( NS_ctx.paraview )
   {
      if ( (mkdir(NS_ctx.outfolder, 0777) == -1) && (pmesh->GetMyRank() == 0) ) {mfem::err << "Error :  " << strerror(errno) << std::endl;}

      paraview_dc = new ParaViewDataCollection("Results-Paraview", pmesh);
      paraview_dc->SetPrefixPath(NS_ctx.outfolder);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->RegisterField("pressure",&p_gf);
      paraview_dc->RegisterField("velocity",&u_gf);
      u_gf = naviersolver.GetVelocity();
      p_gf = naviersolver.GetPressure();

      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(t);
      paraview_dc->Save();
   }   


   //
   /// 8. Solve unsteady problem
   //

   double CFL = 0.0;
   
   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= NS_ctx.t_final - dt / 2)
      {
         last_step = true;
      }

      naviersolver.Step(t, dt, step);

      //CFL = qt.ComputeCFL(u_gf,NS_ctx.dt);

      // Compare against exact solution of velocity and pressure.
      u_gf = naviersolver.GetVelocity();
      p_gf = naviersolver.GetPressure();

      if( NS_ctx.paraview )
      {
         paraview_dc->SetCycle(step+1);
         paraview_dc->SetTime(t);
         paraview_dc->Save();
      }

   }

   naviersolver.PrintTimingData();

   delete pmesh;
   delete paraview_dc; 

   return 0;
}


void inflow(const Vector &x, double t, Vector &u)
{

   const int dim = x.Size();
   double xi = x[0];
   double yi = x[1];

   u = 0.0;

   u(1) = 0.0;

   // Preload
   double preload = 0.0;
   if( t < NS_ctx.preloadT )
   {
      preload = 0.5* (1.0 - cos(M_PI*t/NS_ctx.preloadT));   
   }
   else
   {
      preload = 1.0;
   }

   if( dim == 3)
   {
      double zi = x[2];
      u(0) = preload * NS_ctx.Umax * 32.0 / Mesh_ctx.D *( 1.0 - yi/Mesh_ctx.D) * ( 1.0 - 2.0 * yi / Mesh_ctx.D) * ( zi/Mesh_ctx.D - 1.0) * zi;
      u(2) = 0.0;
   }
   else
   {
      u(0) = preload * 8.0 * NS_ctx.Umax * ( 1.0 - yi/Mesh_ctx.D ) * ( 2.0 * yi/Mesh_ctx.D - 1.0 );
   }

}

void noSlip(const Vector &x, double t, Vector &u)
{
   u = 0.0;
}

double pZero(const Vector &x, double t)
{
   return 0.0;
}