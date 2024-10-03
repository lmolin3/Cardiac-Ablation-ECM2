// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// This miniapp aims to demonstrate how to solve two PDEs, that represent
// different physics, on the same domain. MFEM's SubMesh interface is used to
// compute on and transfer between the spaces of predefined parts of the domain.
// For the sake of simplicity, the spaces on each domain are using the same
// order H1 finite elements. This does not mean that the approach is limited to
// this configuration.
//
// A 3D domain comprised of an outer box with a cylinder shaped inside is used.
//
// A heat equation is described on the outer box domain
//
//                  dT/dt = κΔT         in outer box
//                      T = T_out       on outside wall
//                   ∇T•n = 0           on inside (cylinder) wall
//
// with temperature T and coefficient κ (non-physical in this example).
//
// A heat equation equation is described inside the cylinder domain
//
//                  dT/dt = κΔT          in inner cylinder
//                      T = T_wall       on cylinder wall (obtained from heat equation)
//                   ∇T•n = 0            else
//
// with temperature T, coefficients κ, α and prescribed velocity profile b.
//
// To couple the solutions of both equations, a segregated solve with one way
// coupling approach is used. The heat equation of the outer box is solved from
// the timestep T_box(t) to T_box(t+dt). Then for the convection-diffusion
// equation T_wall is set to T_box(t+dt) and the equation is solved for T(t+dt)
// which results in a first-order one way coupling.

#include "mfem.hpp"
#include "lib/heat_solver.hpp"

#include <fstream>
#include <iostream>
#include <memory>

using namespace mfem;

IdentityMatrixCoefficient *Id = NULL;

int main(int argc, char *argv[])
{

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mpi::Init();
   Hypre::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options.
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // FE
   int order = 1;
   bool pa = false; // Enable partial assembly
   // Mesh
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   // Time integrator
   int ode_solver_type = 1;
   real_t t_final = 5.0;
   real_t dt = 1.0e-5;
   // Postprocessing
   bool visit = false;
   bool paraview = true;
   int save_freq = 1; // Save fields every 'save_freq' time steps
   const char *outfolder = "";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&ode_solver_type, "-ode", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   4 - Implicit Midpoint, 5 - SDIRK23, 6 - SDIRK34,\n\t"
                  "\t   7 - Forward Euler, 8 - RK2, 9 - RK3 SSP, 10 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&save_freq, "-sf", "--save-freq",
                  "Save fields every 'save_freq' time steps.");
   args.AddOption(&outfolder, "-of", "--out-folder",
                  "Output folder.");
   args.ParseCheck();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mesh *serial_mesh = new Mesh("multidomain-hex.mesh");
   int sdim = serial_mesh->SpaceDimension();
   ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;

   parent_mesh.UniformRefinement();

   // Create the sub-domains for the cylinder and the outer block
   Array<int> cylinder_domain_attributes(1);
   cylinder_domain_attributes[0] = 1;

   Array<int> box_domain_attributes(1);
   box_domain_attributes[0] = 2;

   auto cylinder_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, cylinder_domain_attributes));

   auto block_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, box_domain_attributes));

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Set up coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   auto Id = new IdentityMatrixCoefficient(sdim);

   Array<int> attr_cyl(0), attr_block(0);
   attr_cyl.Append(1), attr_block.Append(2);

   double kval_cyl, cval_cyl, rhoval_cyl;
   kval_cyl = 1.0;   // W/mK
   cval_cyl = 1.0;   // J/kgK
   rhoval_cyl = 1.0; // kg/m^3

   double kval_block, cval_block, rhoval_block;
   kval_block = 1.0;   // W/mK
   cval_block = 1.0;   // J/kgK
   rhoval_block = 1.0; // kg/m^3

   // Conductivity
   Array<MatrixCoefficient *> coefs_k_cyl(0);
   coefs_k_cyl.Append(new ScalarMatrixProductCoefficient(kval_cyl, *Id));
   PWMatrixCoefficient *Kappa_cyl = new PWMatrixCoefficient(sdim, attr_cyl, coefs_k_cyl);

   Array<MatrixCoefficient *> coefs_k_block(0);
   coefs_k_block.Append(new ScalarMatrixProductCoefficient(kval_block, *Id));
   PWMatrixCoefficient *Kappa_block = new PWMatrixCoefficient(sdim, attr_block, coefs_k_block);

   // Heat Capacity
   Array<Coefficient *> coefs_c_cyl(0);
   coefs_c_cyl.Append(new ConstantCoefficient(cval_cyl));
   PWCoefficient *c_cyl = new PWCoefficient(attr_cyl, coefs_c_cyl);

   Array<Coefficient *> coefs_c_block(0);
   coefs_c_block.Append(new ConstantCoefficient(cval_block));
   PWCoefficient *c_block = new PWCoefficient(attr_block, coefs_c_block);

   // Density
   Array<Coefficient *> coefs_rho_cyl(0);
   coefs_rho_cyl.Append(new ConstantCoefficient(rhoval_cyl));
   PWCoefficient *rho_cyl = new PWCoefficient(attr_cyl, coefs_rho_cyl);

   Array<Coefficient *> coefs_rho_block(0);
   coefs_rho_block.Append(new ConstantCoefficient(rhoval_block));
   PWCoefficient *rho_block = new PWCoefficient(attr_block, coefs_rho_block);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create BC Handler (not populated yet)
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool verbose = true;

   heat::BCHandler *bcs_cyl = new heat::BCHandler(cylinder_submesh, verbose); // Boundary conditions handler for cylinder
   heat::BCHandler *bcs_block = new heat::BCHandler(block_submesh, verbose);  // Boundary conditions handler for block

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Create the Heat Solver
   ///////////////////////////////////////////////////////////////////////////////////////////////

   heat::HeatSolver Heat_Cylinder(cylinder_submesh, order, bcs_cyl, Kappa_cyl, c_cyl, rho_cyl, ode_solver_type);
   heat::HeatSolver Heat_Block(block_submesh, order, bcs_block, Kappa_block, c_block, rho_block, ode_solver_type);

   ParGridFunction *temperature_cylinder_gf = Heat_Cylinder.GetTemperatureGfPtr();
   ParGridFunction *temperature_block_gf = Heat_Block.GetTemperatureGfPtr();

   // Create the transfer map needed in the time integration loop
   auto temperature_block_to_cylinder_map = ParSubMesh::CreateTransferMap(
       *temperature_block_gf,
       *temperature_cylinder_gf);

   // Setup ouput
   ParaViewDataCollection paraview_dc_cylinder("Heat-Cylinder", cylinder_submesh.get());
   ParaViewDataCollection paraview_dc_block("Heat-Block", block_submesh.get());
   if (paraview)
   {
      paraview_dc_cylinder.SetPrefixPath(outfolder);
      paraview_dc_cylinder.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_cylinder.SetCompressionLevel(9);
      Heat_Cylinder.RegisterParaviewFields(paraview_dc_cylinder);

      paraview_dc_block.SetPrefixPath(outfolder);
      paraview_dc_block.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_block.SetCompressionLevel(9);
      Heat_Block.RegisterParaviewFields(paraview_dc_block);
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Populate BC Handler
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Outer Box:
   // - T = T_out on the outside wall
   // - ∇T•n = 0 on the inside (cylinder) wall and top/bottom surfaces

   Array<int> block_wall_attributes(block_submesh->bdr_attributes.Max());
   block_wall_attributes = 0;
   block_wall_attributes[0] = 1;
   block_wall_attributes[1] = 1;
   block_wall_attributes[2] = 1;
   block_wall_attributes[3] = 1;

   double T_out = 1.0;
   bcs_block->AddDirichletBC(T_out, block_wall_attributes);

   // Cylinder:
   // - T = T_wall on the cylinder wall (obtained from heat equation on box)
   // - ∇T•n = 0 else
   //

   Array<int> inner_cylinder_wall_attributes(
       cylinder_submesh->bdr_attributes.Max());
   inner_cylinder_wall_attributes = 0;
   inner_cylinder_wall_attributes[8] = 1;

   ParGridFunction *temperature_cylinder_wall_gf = new ParGridFunction(Heat_Cylinder.GetFESpace()); // Check if creates deep copy or not -->  want deep copy since we need to step Heat_Cylinder using T_prev, and store T_new in another gf
   GridFunctionCoefficient *T_wall = new GridFunctionCoefficient(temperature_cylinder_wall_gf);
   bcs_cyl->AddDirichletBC(T_wall, inner_cylinder_wall_attributes);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Heat_Cylinder.EnablePA(pa);
   Heat_Cylinder.Setup();

   Heat_Block.EnablePA(pa);
   Heat_Block.Setup();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Perform time-integration (looping over the time iterations, step, with a
   //     time-step dt).
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ConstantCoefficient T0(0.0);
   temperature_block_gf->ProjectCoefficient(T0);
   Heat_Block.SetInitialTemperature(*temperature_block_gf);

   temperature_cylinder_gf->ProjectCoefficient(T0);
   Heat_Cylinder.SetInitialTemperature(*temperature_cylinder_gf);

   real_t t = 0.0;
   bool last_step = false;

   // Write fields to disk for VisIt
   if (paraview)
   {
      Heat_Block.WriteFields(0, t);
      Heat_Cylinder.WriteFields(0, t);
   }

   for (int step = 1; !last_step; step++)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      // Advance the diffusion equation on the outer block to the next time step
      Heat_Block.Step(t, dt, step);
      t -= dt; // Reset t to same time step, since t is incremented in the Step function

      // Transfer the solution from the inner surface of the outer block to
      // the cylinder outer surface to act as a boundary condition.
      temperature_block_to_cylinder_map.Transfer(*temperature_block_gf,
                                                 *temperature_cylinder_wall_gf);

      // Advance the convection-diffusion equation on the outer block to the
      // next time step
      Heat_Cylinder.Step(t, dt, step);

      // Write fields to disk for Paraview
      if (paraview && (step % save_freq == 0))
      {
         Heat_Block.WriteFields(step, t);
         Heat_Cylinder.WriteFields(step, t);
      }
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   delete temperature_cylinder_wall_gf;

   delete Kappa_cyl;
   delete c_cyl;
   delete rho_cyl;

   delete Kappa_block;
   delete c_block;
   delete rho_block;

   // Delete the MatrixCoefficient objects at the end of main
   for (int i = 0; i < coefs_k_block.Size(); i++)
   {
      delete coefs_k_block[i];
   }

   for (int i = 0; i < coefs_c_block.Size(); i++)
   {
      delete coefs_c_block[i];
   }

   for (int i = 0; i < coefs_rho_block.Size(); i++)
   {
      delete coefs_rho_block[i];
   }

   for (int i = 0; i < coefs_k_cyl.Size(); i++)
   {
      delete coefs_k_cyl[i];
   }

   for (int i = 0; i < coefs_c_cyl.Size(); i++)
   {
      delete coefs_c_cyl[i];
   }

   for (int i = 0; i < coefs_rho_cyl.Size(); i++)
   {
      delete coefs_rho_cyl[i];
   }

   delete Id;

   return 0;
}
