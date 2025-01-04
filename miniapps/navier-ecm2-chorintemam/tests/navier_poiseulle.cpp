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
// Navier Poiseulle example
//
// The problem domain is set up like this
//
//                            no slip
//             |\     + ----------------------------------+
//             | \    |                                   |
// Parabolic -->  \   |                                   |   Traction free (outflow)
//  inflow   -->  /   |                                   |
//             | /    |                                   |
//             |/     + ----------------------------------+
//                            no slip
//
// Inflow profile: u_in = 1/4kinvis dpdx (R^2 - r^2)    (Poiseulle flow)
// 
// Mesh attributes for 2D are:
// inflow = 1, outflow = 2, cylinder = 3, wall = 4
//
// Mesh attributes for 3D are:
// inflow = 1, outflow = 2, sphere = 3, wall = 4
//
//
// Run with:
// mpirun -np 4 ./navier-poiseulle -d 2 -rs 0 -ou 2 -op 1 -dt 1e-3 -tf 1e-1 -kv 1.0 -dp 1.0 --gamma 1.0 --verbose --paraview -of ./Output/Poiseulle/
//

   
#include "lib/navier_solver.hpp"
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
   real_t R = 1.0;           
   real_t kinvis = 1.0;
   real_t dpdx = 4.0;
   real_t dt = 1e-3;
   real_t t_final = 10 * dt;
   real_t gamma = 1.0;
   bool verbose = true;
   bool paraview = false;
   const char *outfolder = "./Output/Poiseulle/Test/";
   bool ExportData = false;
   int bdf = 3;
   bool yosida = false;
} NS_ctx;

struct s_MeshContext // mesh
{
   int dim = 2;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
} Mesh_ctx;


// Forward declarations of functions
void inflow(const Vector &x, real_t t, Vector &u);
void noSlip(const Vector &x, real_t t, Vector &u);
real_t pZero(const Vector &x, real_t t);

int main(int argc, char *argv[])
{

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options.
   ///////////////////////////////////////////////////////////////////////////////////////////////

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
   args.AddOption(&NS_ctx.kinvis, "-kv", "--kinematic-viscosity", "Kinematic Viscosity.");
   args.AddOption(&NS_ctx.dpdx, "-dp", "--pressure-drop", "Pressure difference driving flow.");
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
                   "-of",
                   "--output-folder",
                   "Output folder.");
    args.AddOption(&NS_ctx.gamma,
                   "-g",
                   "--gamma",
                   "Relaxation parameter");
    args.AddOption(&NS_ctx.bdf,
                   "-bdf",
                   "--bdf-order",
                   "Maximum bdf order (1<=bdf<=3)");
    args.AddOption(&NS_ctx.yosida,
                   "-y",
                   "--yosida",
                   "-ct",
                   "--chorin-temam",
                   "Use Yosida or Chorin-Temam splitting.");
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


   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Read Mesh and create parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mesh mesh;

   switch (Mesh_ctx.dim)
   {
   case 2:
   {
      mesh = Mesh::LoadFromFile("./Mesh/channel_2D.msh");
      break;
   }
   case 3:
   {
      mesh = Mesh::LoadFromFile("./Mesh/channel_3D.msh");
      break;
   }
   default:
      break;
   }
   //mesh.EnsureNodes();


   for (int l = 0; l < Mesh_ctx.ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }


   auto pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, mesh);

   mesh.Clear();
   {
      for (int l = 0; l < Mesh_ctx.par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create the NS Solver and BCHandler
   ///////////////////////////////////////////////////////////////////////////////////////////////
   
   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool verbose = false;
   navier::BCHandler *bcs = new navier::BCHandler(pmesh, verbose); // Boundary conditions handler
   navier::NavierUnsteadySolver naviersolver(pmesh, bcs, NS_ctx.kinvis, NS_ctx.uorder, NS_ctx.porder, NS_ctx.verbose, NS_ctx.yosida);

   naviersolver.SetSolvers(sParams,sParams,sParams,sParams);
   naviersolver.SetMaxBDFOrder(NS_ctx.bdf);
   naviersolver.SetGamma(NS_ctx.gamma);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Set up boundary conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////


   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   int inflow_attr = 1;
   int outflow_attr = 2;
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 1;                    // Mark walls/cylinder for no-slip condition
   ess_attr[inflow_attr - 1]  = 0;
   ess_attr[outflow_attr - 1] = 0;

   // Inflow
   VectorFunctionCoefficient *u_in = new VectorFunctionCoefficient(pmesh->Dimension(), inflow);
   bcs->AddVelDirichletBC(u_in,inflow_attr);

   // Outflow
   FunctionCoefficient *p_out = new FunctionCoefficient(pZero);
   bcs->AddPresDirichletBC(p_out,outflow_attr);

   // No Slip
   bcs->AddVelDirichletBC(noSlip,ess_attr);


   // Initial condition
   //naviersolver.SetInitialConditionVel( *u_in );
   //naviersolver.SetInitialConditionPrevVel( *u_in );

   ParGridFunction *u_gf = naviersolver.GetVelocity();
   ParGridFunction *p_gf = naviersolver.GetPressure();

   // Creating output directory if not existent
   ParaViewDataCollection *paraview_dc = nullptr;
   
   if( NS_ctx.paraview )
   {
      if ( (mkdir(NS_ctx.outfolder, 0777) == -1) && (pmesh->GetMyRank() == 0) ) {mfem::err << "Error :  " << strerror(errno) << std::endl;}

      paraview_dc = new ParaViewDataCollection("Results-Paraview", pmesh.get());
      paraview_dc->SetPrefixPath(NS_ctx.outfolder);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetCompressionLevel(9);
      naviersolver.RegisterParaviewFields(*paraview_dc);

      naviersolver.WriteFields(0, 0.0);

   }   

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   naviersolver.Setup(NS_ctx.dt);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Solve unsteady problem
   ///////////////////////////////////////////////////////////////////////////////////////////////
 
   real_t t = 0.0;
   real_t dt = NS_ctx.dt;
   bool last_step = false;


   for (int step = 1; !last_step; ++step)
   {
      if (t + dt >= NS_ctx.t_final - dt / 2)
      {
         last_step = true;
      }

      naviersolver.Step(t, dt, step);

      if( NS_ctx.paraview )
      {
         naviersolver.WriteFields(step, t);
      }

      if (Mpi::Root())
      {
         printf("\n%11s %11s\n", "Time", "dt");
         printf("%.5E %.5E\n", t, dt);
         fflush(stdout);
      }
   }

   naviersolver.PrintTimingData();

   delete u_in;
   delete paraview_dc; 

   return 0;
}



void inflow(const Vector &x, real_t t, Vector &u)
{

   const int dim = x.Size();

   real_t xi = x[0];
   real_t yi = x[1];

   u = 0.0;

   if( dim == 3)
   {
      real_t zi = x[2];
      real_t r = yi*yi + zi*zi;
      // NYI
   }
   else
   {
      real_t r = yi;
      u(0) = 1.0 / (4.0 * NS_ctx.kinvis ) * NS_ctx.dpdx * ( NS_ctx.R*NS_ctx.R - r*r);   // 1/ 4kinvis dpdx (R^2 - r^2)
   }

}

void noSlip(const Vector &x, real_t t, Vector &u)
{
   u = 0.0;
}

real_t pZero(const Vector &x, real_t t)
{
   return 0.0;
}