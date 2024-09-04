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
// Navier Lid driven cavity 
//
// The problem domain is set up like this
//
//                 u = (1,0)
//            + --> --> --> --> +
//            |                 |
//            |                 |
// u=(0,0)    |                 |     u=(0,0)
//            |                 |
//            |                 |
//            |                 |
//            +-----------------+
//                  u=(0,0)
//
// and Dirichlet boundary conditions are applied for the velocity on every
// boundary.
//
//
// Run with:
// mpirun -np 4 ./navier_ecm2_liddriven -d 2 -rs 0 -rp 0 -ou 2 -op 1 -dt 1e-3 -tf 1e-1 -re 100.0 --gamma 1.0 --verbose --paraview --output-folder ./Output/LidDriven/Test/
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


struct s_NavierContext // Navier Stokes params
{
   int uorder = 2;
   int porder = 1;
   double R = 1.0;           
   double kinvis = 1.0;
   double re = 100;
   double dt = 1e-3;
   double t_final = 10 * dt;
   double gamma = 1.0;
   bool verbose = true;
   bool paraview = false;
   const char *outfolder = "./Output/Poiseulle/Test/";
   bool ExportData = false;
   int bdf = 3;
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
void inflow(const Vector &x, double t, Vector &u);
void noSlip(const Vector &x, double t, Vector &u);
double pZero(const Vector &x, double t);

int main(int argc, char *argv[])
{

   //
   /// 1. Initialize MPI and Hypre.
   //
   Mpi::Init(argc, argv);
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
   args.AddOption(&NS_ctx.re, "-re", "--reynolds",
                   "Reynolds number");
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
   Element::Type type;
   switch (Mesh_ctx.elem)
   {
   case 0: // quad
      type = (Mesh_ctx.dim == 2) ? Element::QUADRILATERAL: Element::HEXAHEDRON;
      break;
   case 1: // tri
      type = (Mesh_ctx.dim == 2) ? Element::TRIANGLE: Element::TETRAHEDRON;
      break;
   }

   Mesh mesh;
   switch (Mesh_ctx.dim)
   {
   case 2: // 2d
      mesh = Mesh::MakeCartesian2D(Mesh_ctx.n,Mesh_ctx.n,type,true);
      break;
   case 3: // 3d
      mesh = Mesh::MakeCartesian3D(Mesh_ctx.n,Mesh_ctx.n,Mesh_ctx.n,type,true);
      break;
   }
   mesh.EnsureNodes();


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
   NS_ctx.kinvis = 1.0/NS_ctx.re;

   if (NS_ctx.verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Reynolds: " << NS_ctx.re << std::endl;
      mfem::out << "Kinematic viscosity: " << NS_ctx.kinvis << std::endl;
   }

   NavierUnsteadySolver naviersolver(pmesh, NS_ctx.uorder, NS_ctx.porder, NS_ctx.kinvis, NS_ctx.verbose);

   naviersolver.SetSolvers(sParams,sParams,sParams,sParams);
   naviersolver.SetMaxBDFOrder(NS_ctx.bdf);

#ifdef MFEM_DEBUG
   naviersolver.SetExportData( NS_ctx.ExportData ); // Export matrices/vectors 
#endif

   //
   /// 6. Set initial condition and boundary conditions
   //

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
    int inflow_attr = (Mesh_ctx.dim == 2) ? 3: 6; // for cube the top boundary is 6, for square it's 3
    Array<int> ess_attr(pmesh->bdr_attributes.Max());
    ess_attr = 1;
    ess_attr[inflow_attr - 1] = 0;

    // Inflow
    naviersolver.AddVelDirichletBC(inflow,inflow_attr);

    // No Slip
    naviersolver.AddVelDirichletBC(noSlip,ess_attr);


   //
   /// 7. Setup
   //

   double t = 0.0;
   double dt = NS_ctx.dt;
   bool last_step = false;

   naviersolver.SetOutputFolder(NS_ctx.outfolder);
   naviersolver.Setup(NS_ctx.dt);
   naviersolver.SetGamma(NS_ctx.gamma);
   ParGridFunction u_gf(naviersolver.GetUFes());
   ParGridFunction p_gf(naviersolver.GetPFes());

   ParGridFunction u_pred_gf(naviersolver.GetUFes());
   ParGridFunction p_pred_gf(naviersolver.GetPFes());

   // Creating output directory if not existent
   ParaViewDataCollection *paraview_dc = nullptr;

   
   if( NS_ctx.paraview )
   {
      if ( (mkdir(NS_ctx.outfolder, 0777) == -1) && (pmesh->GetMyRank() == 0) ) {mfem::err << "Error :  " << strerror(errno) << std::endl;}

      paraview_dc = new ParaViewDataCollection("Results-Paraview", pmesh);
      paraview_dc->SetPrefixPath(NS_ctx.outfolder);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->RegisterField("predicted_velocity",&u_pred_gf);
      paraview_dc->RegisterField("predicted_pressure",&p_pred_gf);
      paraview_dc->RegisterField("corrected_pressure",&p_gf);
      paraview_dc->RegisterField("corrected_velocity",&u_gf);
      u_gf = naviersolver.GetVelocity();
      p_gf = naviersolver.GetPressure();
      u_pred_gf = naviersolver.GetPredictedVelocity();
      p_pred_gf = naviersolver.GetPredictedPressure();

      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(t);
      paraview_dc->Save();
   }   


   //
   /// 8. Solve unsteady problem
   //
   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= NS_ctx.t_final - dt / 2)
      {
         last_step = true;
      }

      naviersolver.Step(t, dt, step);

      // Compare against exact solution of velocity and pressure.
      u_gf = naviersolver.GetVelocity();
      p_gf = naviersolver.GetPressure();
      u_pred_gf = naviersolver.GetPredictedVelocity();
      p_pred_gf = naviersolver.GetPredictedPressure();

      if( NS_ctx.paraview )
      {
         paraview_dc->SetCycle(step+1);
         paraview_dc->SetTime(t);
         paraview_dc->Save();
      }

      if (Mpi::Root())
      {
         printf("\n%11s %11s\n", "Time", "dt");
         printf("%.5E %.5E\n", t, dt);
         fflush(stdout);
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

   u = 0.0;
   u(0) = 1.0;
   u(1) = 0.0;

   if( dim == 3)
   {
      u(2) = 0;
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