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
//                  dT/Sim_ctx.dt = κΔT         in outer box
//                      T = T_out       on outside wall
//                   ∇T•n = 0           on inside (cylinder) wall
//
// with temperature T and coefficient κ (non-physical in this example).
//
// A heat equation equation is described inside the cylinder domain
//
//                  dT/Sim_ctx.dt = κΔT          in inner cylinder
//                      T = T_wall       on cylinder wall (obtained from heat equation)
//                   ∇T•n = 0            else
//
// with temperature T, coefficients κ, α and prescribed velocity profile b.
//
// To couple the solutions of both equations, a segregated solve with one way
// coupling approach is used. The heat equation of the outer box is solved from
// the timestep T_box(t) to T_box(t+Sim_ctx.dt). Then for the convection-diffusion
// equation T_wall is set to T_box(t+Sim_ctx.dt) and the equation is solved for T(t+Sim_ctx.dt)
// which results in a first-order one way coupling.

// MFEM library
#include "mfem.hpp"

// Multiphysics modules
#include "lib/heat_solver.hpp"

// Physical and Domain-Decomposition parameters
#include "../contexts.hpp"

// Utils

// Output
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

   OptionsParser args(argc, argv);
   args.AddOption(&Heat_ctx.order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&Heat_ctx.pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&Heat_ctx.ode_solver_type, "-ode", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   4 - Implicit Midpoint, 5 - SDIRK23, 6 - SDIRK34,\n\t"
                  "\t   7 - Forward Euler, 8 - RK2, 9 - RK3 SSP, 10 - RK4.");
   args.AddOption(&Sim_ctx.t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&Sim_ctx.dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&Sim_ctx.paraview, "-paraview", "-paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable Paraview visualization.");
   args.AddOption(&Sim_ctx.save_freq, "-sf", "--save-freq",
                  "Save fields every 'save_freq' time steps.");
   args.AddOption(&Sim_ctx.outfolder, "-of", "--out-folder",
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

   Array<int> attr_cyl(0), attr_solid(0);
   attr_cyl.Append(1), attr_solid.Append(2);

   // Overwrite the default values
   Heat_ctx.k_cylinder = 1.0;   // W/mK
   Heat_ctx.c_cylinder = 1.0;   // J/kgK
   Heat_ctx.rho_cylinder = 1.0; // kg/m^3

   Heat_ctx.k_solid = 1.0;   // W/mK
   Heat_ctx.c_solid = 1.0;   // J/kgK
   Heat_ctx.rho_solid = 1.0; // kg/m^3

   // Conductivity
   Array<MatrixCoefficient *> coefs_k_cyl(0);
   coefs_k_cyl.Append(new ScalarMatrixProductCoefficient(Heat_ctx.k_cylinder, *Id));
   PWMatrixCoefficient *Kappa_cyl = new PWMatrixCoefficient(sdim, attr_cyl, coefs_k_cyl);

   Array<MatrixCoefficient *> coefs_k_solid(0);
   coefs_k_solid.Append(new ScalarMatrixProductCoefficient(Heat_ctx.k_solid, *Id));
   PWMatrixCoefficient *Kappa_solid = new PWMatrixCoefficient(sdim, attr_solid, coefs_k_solid);

   // Heat Capacity
   Array<Coefficient *> coefs_c_cyl(0);
   coefs_c_cyl.Append(new ConstantCoefficient(Heat_ctx.c_cylinder));
   PWCoefficient *c_cyl = new PWCoefficient(attr_cyl, coefs_c_cyl);

   Array<Coefficient *> coefs_c_solid(0);
   coefs_c_solid.Append(new ConstantCoefficient(Heat_ctx.c_solid));
   PWCoefficient *c_solid = new PWCoefficient(attr_solid, coefs_c_solid);

   // Density
   Array<Coefficient *> coefs_rho_cyl(0);
   coefs_rho_cyl.Append(new ConstantCoefficient(Heat_ctx.rho_cylinder));
   PWCoefficient *rho_cyl = new PWCoefficient(attr_cyl, coefs_rho_cyl);

   Array<Coefficient *> coefs_rho_solid(0);
   coefs_rho_solid.Append(new ConstantCoefficient(Heat_ctx.rho_solid));
   PWCoefficient *rho_solid = new PWCoefficient(attr_solid, coefs_rho_solid);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create BC Handler (not populated yet)
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool verbose = true;

   heat::BCHandler *bcs_cyl = new heat::BCHandler(cylinder_submesh, verbose); // Boundary conditions handler for cylinder
   heat::BCHandler *bcs_solid = new heat::BCHandler(block_submesh, verbose);  // Boundary conditions handler for block

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Create the Heat Solver
   ///////////////////////////////////////////////////////////////////////////////////////////////

   heat::HeatSolver Heat_Cylinder(cylinder_submesh, Heat_ctx.order, bcs_cyl, Kappa_cyl, c_cyl, rho_cyl, Heat_ctx.ode_solver_type);
   heat::HeatSolver Heat_solid(block_submesh, Heat_ctx.order, bcs_solid, Kappa_solid, c_solid, rho_solid, Heat_ctx.ode_solver_type);

   ParGridFunction *temperature_cylinder_gf = Heat_Cylinder.GetTemperatureGfPtr();
   ParGridFunction *temperature_solid_gf = Heat_solid.GetTemperatureGfPtr();

   // Create the transfer map needed in the time integration loop
   auto temperature_solid_to_cylinder_map = ParSubMesh::CreateTransferMap(
       *temperature_solid_gf,
       *temperature_cylinder_gf);

   // Setup ouput
   ParaViewDataCollection paraview_dc_cylinder("Heat-Cylinder", cylinder_submesh.get());
   ParaViewDataCollection paraview_dc_solid("Heat-Block", block_submesh.get());
   if (Sim_ctx.paraview)
   {
      paraview_dc_cylinder.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_cylinder.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_cylinder.SetCompressionLevel(9);
      Heat_Cylinder.RegisterParaviewFields(paraview_dc_cylinder);

      paraview_dc_solid.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_solid.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_solid.SetCompressionLevel(9);
      Heat_solid.RegisterParaviewFields(paraview_dc_solid);
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

   real_t T_out = 1.0;
   bcs_solid->AddDirichletBC(T_out, block_wall_attributes);

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

   Heat_Cylinder.EnablePA(Heat_ctx.pa);
   Heat_Cylinder.Setup();

   Heat_solid.EnablePA(Heat_ctx.pa);
   Heat_solid.Setup();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Perform time-integration (looping over the time iterations, step, with a
   //     time-step Sim_ctx.dt).
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ConstantCoefficient T0(0.0);
   temperature_solid_gf->ProjectCoefficient(T0);
   Heat_solid.SetInitialTemperature(*temperature_solid_gf);

   temperature_cylinder_gf->ProjectCoefficient(T0);
   Heat_Cylinder.SetInitialTemperature(*temperature_cylinder_gf);

   real_t t = 0.0;
   bool last_step = false;

   // Write fields to disk for VisIt
   if (Sim_ctx.paraview)
   {
      Heat_solid.WriteFields(0, t);
      Heat_Cylinder.WriteFields(0, t);
   }

   for (int step = 1; !last_step; step++)
   {
      if (t + Sim_ctx.dt >= Sim_ctx.t_final - Sim_ctx.dt / 2)
      {
         last_step = true;
      }

      // Advance the diffusion equation on the outer block to the next time step
      Heat_solid.Step(t, Sim_ctx.dt, step);
      t -= Sim_ctx.dt; // Reset t to same time step, since t is incremented in the Step function

      // Transfer the solution from the inner surface of the outer block to
      // the cylinder outer surface to act as a boundary condition.
      temperature_solid_to_cylinder_map.Transfer(*temperature_solid_gf,
                                                 *temperature_cylinder_wall_gf);

      // Advance the convection-diffusion equation on the outer block to the
      // next time step
      Heat_Cylinder.Step(t, Sim_ctx.dt, step);

      // Write fields to disk for Paraview
      if (Sim_ctx.paraview && (step % Sim_ctx.save_freq == 0))
      {
         Heat_solid.WriteFields(step, t);
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

   delete Kappa_solid;
   delete c_solid;
   delete rho_solid;

   // Delete the MatrixCoefficient objects at the end of main
   for (int i = 0; i < coefs_k_solid.Size(); i++)
   {
      delete coefs_k_solid[i];
   }

   for (int i = 0; i < coefs_c_solid.Size(); i++)
   {
      delete coefs_c_solid[i];
   }

   for (int i = 0; i < coefs_rho_solid.Size(); i++)
   {
      delete coefs_rho_solid[i];
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
