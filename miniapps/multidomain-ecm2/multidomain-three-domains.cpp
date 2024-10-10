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
// A 3D domain comprised of three domains:
// - solid box (S)
// - fluid domain (F)
// - solid cylinder (C) (embedded in the fluid domain)
//
// A diffusion equation is described in the cylinder
//
//                 rho c dT/dt = κΔT          in cylinder
//                           Q = Qval       inside sphere
//
// An advection-diffusion equation is described inside the fluid domain
//
//                 rho c dT/dt = κΔT - α • u ∇T      in fluid
//
// A reaction-diffusion equation is described in the solid box:
//
//                rho c dT/dt = κΔT - β T      in box
//
// with temperature T, coefficients κ, α, β and prescribed velocity profile u.
//
// To couple the solutions of both equations, a segregated solve with two way
// coupling (Neumann-Dirichlet) approach is used to solve the timestep tn tn+dt
//
// C -> S -> F
//
//
// Sample run:
//

#include "mfem.hpp"
#include "lib/heat_solver.hpp"

#include <fstream>
#include <sstream>
#include <sys/stat.h> // Include for mkdir
#include <iostream>
#include <memory>

using namespace mfem;

IdentityMatrixCoefficient *Id = NULL;

// Forward declaration
void TransferQoIToDest(const std::vector<int> &elem_idx, const ParFiniteElementSpace &fes_grad, const Vector &grad_vec, ParGridFunction &grad_gf, MatrixCoefficient *K);
void print_matrix(const DenseMatrix &A);
void saveConvergenceSubiter(const Array<real_t> &convergence_subiter, const std::string &outfolder, int step);

// Volumetric heat source in the sphere
constexpr double Sphere_Radius = 0.2;
double Qval = 1e3; // W/m^3
double HeatingSphere(const Vector &x, double t);

// Advection velocity profile
void velocity_profile(const Vector &x, Vector &b);
double viscosity = 1.0;
double dPdx = 1.0;
constexpr double R = 0.25; // cylinder mesh radius

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
   // Physics
   double kval_cyl = 1.0;   // W/mK
   double kval_solid = 1.0; // W/mK
   double kval_fluid = 1.0; // W/mK
   double alpha = 1.0;      // Advection coefficient
   double reaction = 0.0;   // Reaction term
   // Test selection
   int test = 1; // 0 - Box heating, 1 - Cylinder heating
   // Mesh
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   // Time integrator
   int ode_solver_type = 1;
   real_t t_final = 1.0;
   real_t dt = 1.0e-2;
   // Domain decomposition
   real_t omega = 0.5; // Relaxation parameter
   // Postprocessing
   bool visit = false;
   bool paraview = true;
   int save_freq = 1; // Save fields every 'save_freq' time steps
   const char *outfolder = "";

   OptionsParser args(argc, argv);
   // Test
   args.AddOption(&test, "-p", "--problem",
                  "Test selection: 0 - Box heating, 1 - Cylinder heating.");
   // FE
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                  "Enable or disable partial assembly.");
   // Mesh
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   // Time integrator
   args.AddOption(&ode_solver_type, "-ode", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   4 - Implicit Midpoint, 5 - SDIRK23, 6 - SDIRK34,\n\t"
                  "\t   7 - Forward Euler, 8 - RK2, 9 - RK3 SSP, 10 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   // Domain decomposition
   args.AddOption(&omega, "-omega", "--relaxation-parameter",
                  "Relaxation parameter.");
   // Physics
   args.AddOption(&kval_cyl, "-kc", "--k-cylinder",
                  "Thermal conductivity of the cylinder (W/mK).");
   args.AddOption(&kval_solid, "-kb", "--k-solid",
                  "Thermal conductivity of the solid (W/mK).");
   args.AddOption(&kval_fluid, "-kf", "--k-fluid",
                  "Thermal conductivity of the fluid (W/mK).");
   args.AddOption(&dPdx, "-dPdx", "--pressure-drop",
                  "Pressure drop forvelocity profile.");
   args.AddOption(&alpha, "-alpha", "--advection-coefficient",
                  "Advection coefficient.");
   args.AddOption(&reaction, "-beta", "--reaction-coefficient",
                  "Reaction coefficient.");
   args.AddOption(&Qval, "-Q", "--volumetric-heat-source",
                  "Volumetric heat source (W/m^3).");
   // Postprocessing
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

   Mesh *serial_mesh = new Mesh("../../data/three-domains.msh");
   int sdim = serial_mesh->SpaceDimension();

   for (int l = 0; l < serial_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }

   ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;

   for (int l = 0; l < parallel_ref_levels; l++)
   {
      parent_mesh.UniformRefinement();
   }

   parent_mesh.EnsureNodes();

   // Create the sub-domains for the cylinder, solid and fluid domains
   AttributeSets &attr_sets = parent_mesh.attribute_sets;
   AttributeSets &bdr_attr_sets = parent_mesh.bdr_attribute_sets;

   Array<int> solid_domain_attributes = attr_sets.GetAttributeSet("Solid");
   Array<int> fluid_domain_attributes = attr_sets.GetAttributeSet("Fluid");
   Array<int> cylinder_domain_attributes = attr_sets.GetAttributeSet("Cylinder");

   auto solid_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, solid_domain_attributes));

   auto fluid_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, fluid_domain_attributes));

   auto cylinder_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, cylinder_domain_attributes));

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Set up coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   auto Id = new IdentityMatrixCoefficient(sdim);

   Array<int> attr_solid(0), attr_fluid(0), attr_cyl(0);
   attr_solid.Append(1), attr_fluid.Append(2), attr_cyl.Append(3);

   double cval_cyl, rhoval_cyl;
   cval_cyl = 1.0;   // J/kgK
   rhoval_cyl = 1.0; // kg/m^3

   double cval_solid, rhoval_solid;
   cval_solid = 1.0;   // J/kgK
   rhoval_solid = 1.0; // kg/m^3

   double cval_fluid, rhoval_fluid;
   cval_fluid = 1.0;   // J/kgK
   rhoval_fluid = 1.0; // kg/m^3

   // Conductivity
   // NOTE: if using PWMatrixCoefficient you need to create one for the boundary too
   auto *Kappa_cyl = new ScalarMatrixProductCoefficient(kval_cyl, *Id);
   auto *Kappa_fluid = new ScalarMatrixProductCoefficient(kval_fluid, *Id);
   auto *Kappa_solid = new ScalarMatrixProductCoefficient(kval_solid, *Id);

   // Heat Capacity
   auto *c_cyl = new ConstantCoefficient(cval_cyl);
   auto *c_fluid = new ConstantCoefficient(cval_fluid);
   auto *c_solid = new ConstantCoefficient(cval_solid);

   // Density
   auto *rho_cyl = new ConstantCoefficient(rhoval_cyl);
   auto *rho_fluid = new ConstantCoefficient(rhoval_fluid);
   auto *rho_solid = new ConstantCoefficient(rhoval_solid);

   // Velocity profile for advectio term in cylinder α∇•(q T)
   VectorCoefficient *q = new VectorFunctionCoefficient(sdim, velocity_profile);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create BC Handler (not populated yet)
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool bc_verbose = true;

   heat::BCHandler *bcs_cyl = new heat::BCHandler(cylinder_submesh, bc_verbose); // Boundary conditions handler for cylinder
   heat::BCHandler *bcs_solid = new heat::BCHandler(solid_submesh, bc_verbose);  // Boundary conditions handler for solid
   heat::BCHandler *bcs_fluid = new heat::BCHandler(fluid_submesh, bc_verbose);  // Boundary conditions handler for fluid

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Create the Heat Solver
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Solvers
   bool solv_verbose = false;
   heat::HeatSolver Heat_Cylinder(cylinder_submesh, order, bcs_cyl, Kappa_cyl, c_cyl, rho_cyl, alpha, q, reaction, ode_solver_type, solv_verbose);
   heat::HeatSolver Heat_Solid(solid_submesh, order, bcs_solid, Kappa_solid, c_solid, rho_solid, ode_solver_type, solv_verbose);
   heat::HeatSolver Heat_Fluid(fluid_submesh, order, bcs_fluid, Kappa_fluid, c_fluid, rho_fluid, alpha, q, reaction, ode_solver_type, solv_verbose);

   // Grid functions in domain (inside solver)
   ParGridFunction *temperature_cylinder_gf = Heat_Cylinder.GetTemperatureGfPtr();
   ParGridFunction *temperature_solid_gf = Heat_Solid.GetTemperatureGfPtr();
   ParGridFunction *temperature_fluid_gf = Heat_Fluid.GetTemperatureGfPtr();

   // Grid functions for interface transfer --> need it for gridfunction coefficients

   ParGridFunction *temperature_fc_cylinder = new ParGridFunction(Heat_Cylinder.GetFESpace());
   ParGridFunction *temperature_fc_fluid = new ParGridFunction(Heat_Fluid.GetFESpace());
   ParGridFunction *temperature_sc_solid = new ParGridFunction(Heat_Solid.GetFESpace());
   ParGridFunction *temperature_sc_cylinder = new ParGridFunction(Heat_Cylinder.GetFESpace());
   // ParGridFunction *temperature_fs_solid = new ParGridFunction(Heat_Solid.GetFESpace());
   // ParGridFunction *temperature_fs_fluid = new ParGridFunction(Heat_Fluid.GetFESpace());
   ParGridFunction *temperature_fluid_fsc = new ParGridFunction(Heat_Fluid.GetFESpace());

   ParGridFunction *heatFlux_fs_solid = new ParGridFunction(Heat_Solid.GetVectorFESpace());
   ParGridFunction *heatFlux_fs_fluid = new ParGridFunction(Heat_Fluid.GetVectorFESpace());
   ParGridFunction *heatFlux_fc_cylinder = new ParGridFunction(Heat_Cylinder.GetVectorFESpace());
   ParGridFunction *heatFlux_fc_fluid = new ParGridFunction(Heat_Fluid.GetVectorFESpace());
   ParGridFunction *heatFlux_sc_solid = new ParGridFunction(Heat_Solid.GetVectorFESpace());
   ParGridFunction *heatFlux_sc_cylinder = new ParGridFunction(Heat_Cylinder.GetVectorFESpace());

   // Grid functions and coefficients for error computation
   ParGridFunction *temperature_solid_prev_gf = new ParGridFunction(Heat_Solid.GetFESpace());
   ParGridFunction *temperature_fluid_prev_gf = new ParGridFunction(Heat_Fluid.GetFESpace());
   ParGridFunction *temperature_cylinder_prev_gf = new ParGridFunction(Heat_Cylinder.GetFESpace());
   GridFunctionCoefficient temperature_solid_prev_coeff(temperature_solid_prev_gf);
   GridFunctionCoefficient temperature_fluid_prev_coeff(temperature_fluid_prev_gf);
   GridFunctionCoefficient temperature_cylinder_prev_coeff(temperature_cylinder_prev_gf);

   // Grid functions for visualization
   ParGridFunction *velocity_gf = new ParGridFunction(Heat_Fluid.GetVectorFESpace());

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Populate BC Handler
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Extract boundary attributes
   Array<int> fluid_cylinder_interface = bdr_attr_sets.GetAttributeSetMarker("Cylinder-Fluid");
   Array<int> fluid_solid_interface = bdr_attr_sets.GetAttributeSetMarker("Solid-Fluid");
   Array<int> solid_cylinder_interface = bdr_attr_sets.GetAttributeSetMarker("Cylinder-Solid");

   Array<int> fluid_lateral_attr = bdr_attr_sets.GetAttributeSetMarker("Fluid Lateral");
   Array<int> fluid_top_attr = bdr_attr_sets.GetAttributeSetMarker("Fluid Top");
   Array<int> cylinder_top_attr = bdr_attr_sets.GetAttributeSetMarker("Cylinder Top");
   Array<int> solid_lateral_attr = bdr_attr_sets.GetAttributeSetMarker("Solid Lateral");
   Array<int> solid_bottom_attr = bdr_attr_sets.GetAttributeSetMarker("Solid Bottom");

   double T_out = 0.0;

   // Solid:
   // - T = T_out on bottom/lateral walls
   // - ks ∇Ts•n = ks ∇Tc•n   on  Γsc
   // - ks ∇Ts•n = ks ∇Tf•n   on  Γfs

   VectorGridFunctionCoefficient *heatFlux_fs_solid_coeff = new VectorGridFunctionCoefficient(heatFlux_fs_solid);
   VectorGridFunctionCoefficient *heatFlux_sc_solid_coeff = new VectorGridFunctionCoefficient(heatFlux_sc_solid);

   bcs_solid->AddNeumannVectorBC(heatFlux_fs_solid_coeff, fluid_solid_interface);
   bcs_solid->AddNeumannVectorBC(heatFlux_sc_solid_coeff, solid_cylinder_interface);
   bcs_solid->AddDirichletBC(T_out, solid_lateral_attr);
   bcs_solid->AddDirichletBC(T_out, solid_bottom_attr);

   // Cylinder:
   // - T = T_out on top wall
   // - kc ∇Tc•n = kc ∇Ts•n   on  Γsc
   // - kc ∇Tc•n = kc ∇Tf•n   on  Γfc

   VectorGridFunctionCoefficient *heatFlux_fc_cylinder_coeff = new VectorGridFunctionCoefficient(heatFlux_fc_cylinder);
   VectorGridFunctionCoefficient *heatFlux_sc_cylinder_coeff = new VectorGridFunctionCoefficient(heatFlux_sc_cylinder);

   bcs_cyl->AddNeumannVectorBC(heatFlux_fc_cylinder_coeff, fluid_cylinder_interface);
   bcs_cyl->AddNeumannVectorBC(heatFlux_sc_cylinder_coeff, solid_cylinder_interface);
   bcs_cyl->AddDirichletBC(T_out, cylinder_top_attr);

   Heat_Cylinder.AddVolumetricTerm(HeatingSphere, cylinder_domain_attributes);

   // Fluid:
   // - T = T_out on top/lateral walls
   // - T = Ts   on  Γfs
   // - T = Tc   on  Γfc

   GridFunctionCoefficient *temperature_fluid_fsc_coeff = new GridFunctionCoefficient(temperature_fluid_fsc);
   bcs_fluid->AddDirichletBC(temperature_fluid_fsc_coeff, fluid_solid_interface);
   bcs_fluid->AddDirichletBC(temperature_fluid_fsc_coeff, fluid_solid_interface);
   bcs_fluid->AddDirichletBC(T_out, fluid_lateral_attr);
   bcs_fluid->AddDirichletBC(T_out, fluid_top_attr);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Setup interface transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create the submesh transfer map
   auto tm_solid_to_fluid = ParSubMesh::CreateTransferMap(*temperature_solid_gf, *temperature_fluid_gf);
   auto tm_cylinder_to_fluid = ParSubMesh::CreateTransferMap(*temperature_cylinder_gf, *temperature_fluid_gf);

   // Setup GSLIB for gradient transfer:
   // 1. Find points on the destination mesh
   // 2. Setup GSLIB finder on the source mesh
   // 3. Define QoI (gradient of the temperature field) on the source meshes (cylinder, solid, fluid)

   // Solid (S) --> Cylinder (D)
   std::vector<int> sc_cylinder_element_idx;
   Vector sc_cylinder_element_coords;
   ecm2_utils::ComputeBdrQuadraturePointsCoords(solid_cylinder_interface, *Heat_Cylinder.GetFESpace(), sc_cylinder_element_idx, sc_cylinder_element_coords);

   FindPointsGSLIB finder_solid_to_cylinder(MPI_COMM_WORLD);
   finder_solid_to_cylinder.Setup(*solid_submesh);
   finder_solid_to_cylinder.FindPoints(sc_cylinder_element_coords, Ordering::byVDIM);

   // Cylinder (S) --> Solid (D)
   std::vector<int> sc_solid_element_idx;
   Vector sc_solid_element_coords;
   ecm2_utils::ComputeBdrQuadraturePointsCoords(solid_cylinder_interface, *Heat_Solid.GetFESpace(), sc_solid_element_idx, sc_solid_element_coords);

   FindPointsGSLIB finder_cylinder_to_solid(MPI_COMM_WORLD);
   finder_cylinder_to_solid.Setup(*cylinder_submesh);
   finder_cylinder_to_solid.FindPoints(sc_solid_element_coords, Ordering::byVDIM);

   // Fluid (S) --> Cylinder (D)
   std::vector<int> fc_cylinder_element_idx;
   Vector fc_cylinder_element_coords;
   ecm2_utils::ComputeBdrQuadraturePointsCoords(fluid_cylinder_interface, *Heat_Cylinder.GetFESpace(), fc_cylinder_element_idx, fc_cylinder_element_coords);

   FindPointsGSLIB finder_fluid_to_cylinder(MPI_COMM_WORLD);
   finder_fluid_to_cylinder.Setup(*fluid_submesh);
   finder_fluid_to_cylinder.FindPoints(fc_cylinder_element_coords, Ordering::byVDIM);

   // Fluid (S) --> Solid (D)
   std::vector<int> fs_solid_element_idx;
   Vector fs_solid_element_coords;
   ecm2_utils::ComputeBdrQuadraturePointsCoords(fluid_solid_interface, *Heat_Solid.GetFESpace(), fs_solid_element_idx, fs_solid_element_coords);

   FindPointsGSLIB finder_fluid_to_solid(MPI_COMM_WORLD);
   finder_fluid_to_solid.Setup(*fluid_submesh);
   finder_fluid_to_solid.FindPoints(fs_solid_element_coords, Ordering::byVDIM);

   // 3. Define QoI (gradient of the temperature field) on the source meshes (cylinder, solid, fluid)
   int qoi_size_on_qp = sdim;
   Vector qoi_src, qoi_dst; // QoI vector, used to store qoi_src in qoi_func and in call to GSLIB interpolator

   // Define lamdbas to compute gradient of the temperature field
   auto grad_temperature_cyl = [&](ElementTransformation &Tr, int pt_idx, int num_pts)
   {
      Vector gradloc(qoi_src.GetData() + pt_idx * sdim, sdim);
      temperature_cylinder_gf->GetGradient(Tr, gradloc);
   };

   auto grad_temperature_solid = [&](ElementTransformation &Tr, int pt_idx, int num_pts)
   {
      Vector gradloc(qoi_src.GetData() + pt_idx * sdim, sdim);
      temperature_solid_gf->GetGradient(Tr, gradloc);
   };

   auto grad_temperature_fluid = [&](ElementTransformation &Tr, int pt_idx, int num_pts)
   {
      Vector gradloc(qoi_src.GetData() + pt_idx * sdim, sdim);
      temperature_fluid_gf->GetGradient(Tr, gradloc);
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Heat_Cylinder.EnablePA(pa);
   Heat_Cylinder.Setup();

   Heat_Solid.EnablePA(pa);
   Heat_Solid.Setup();

   Heat_Fluid.EnablePA(pa);
   Heat_Fluid.Setup();

   // Setup ouput
   ParaViewDataCollection paraview_dc_cylinder("Heat-Cylinder", cylinder_submesh.get());
   ParaViewDataCollection paraview_dc_solid("Heat-Solid", solid_submesh.get());
   ParaViewDataCollection paraview_dc_fluid("Heat-Fluid", fluid_submesh.get());
   if (paraview)
   {
      paraview_dc_cylinder.SetPrefixPath(outfolder);
      paraview_dc_cylinder.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_cylinder.SetCompressionLevel(9);
      Heat_Cylinder.RegisterParaviewFields(paraview_dc_cylinder);

      paraview_dc_solid.SetPrefixPath(outfolder);
      paraview_dc_solid.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_solid.SetCompressionLevel(9);
      Heat_Solid.RegisterParaviewFields(paraview_dc_solid);

      paraview_dc_fluid.SetPrefixPath(outfolder);
      paraview_dc_fluid.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_fluid.SetCompressionLevel(9);
      Heat_Fluid.RegisterParaviewFields(paraview_dc_fluid);
      Heat_Fluid.AddParaviewField("velocity", velocity_gf);
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Perform time-integration (looping over the time iterations, step, with a
   //     time-step dt).
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ConstantCoefficient T0(0.0);
   temperature_solid_gf->ProjectCoefficient(T0);
   Heat_Solid.SetInitialTemperature(*temperature_solid_gf);

   temperature_cylinder_gf->ProjectCoefficient(T0);
   Heat_Cylinder.SetInitialTemperature(*temperature_cylinder_gf);

   temperature_fluid_gf->ProjectCoefficient(T0);
   Heat_Fluid.SetInitialTemperature(*temperature_fluid_gf);

   velocity_gf->ProjectCoefficient(*q);

   real_t t = 0.0;
   bool last_step = false;

   // Write fields to disk for VisIt
   if (paraview)
   {
      Heat_Solid.WriteFields(0, t);
      Heat_Cylinder.WriteFields(0, t);
      Heat_Fluid.WriteFields(0, t);
   }

   return 0;

   bool converged = false;
   double tol = 1.0e-4;
   int max_iter = 100;

   // Vectors for error computation and relaxation
   Vector temperature_solid(temperature_solid_gf->Size()); // Relaxation
   Vector temperature_cylinder(temperature_cylinder_gf->Size());
   Vector temperature_fluid(temperature_fluid_gf->Size());

   Vector temperature_solid_prev(temperature_solid_gf->Size()); // Subiteration (convergence/relaxation)

   Vector temperature_solid_tn(*temperature_solid_gf->GetTrueDofs()); // Initial temperature in outer time loop
   Vector temperature_cylinder_tn(*temperature_cylinder_gf->GetTrueDofs());
   Vector temperature_fluid_tn(*temperature_fluid_gf->GetTrueDofs());

   int cyl_dofs = Heat_Cylinder.GetProblemSize();
   int solid_dofs = Heat_Solid.GetProblemSize();
   int fluid_dofs = Heat_Fluid.GetProblemSize();

   if (Mpi::Root())
   {
      out << " Cylinder dofs: " << cyl_dofs << std::endl;
      out << " Solid dofs: " << solid_dofs << std::endl;
      out << " Fluid dofs: " << fluid_dofs << std::endl;
   }

   if (Mpi::Root())
   {
      out << "----------------------------------------------------------------------------------------"
          << std::endl;
      out << std::left << std::setw(16) << "Step" << std::setw(16) << "Time" << std::setw(16) << "dt" << std::setw(16) << "Sub-iterations" << std::endl;
      out << "----------------------------------------------------------------------------------------"
          << std::endl;
   }

   // Outer loop for time integration
   int num_steps = (int)(t_final / dt);
   Array2D<real_t> convergence(num_steps, 3);
   Array<real_t> convergence_subiter;
   for (int step = 1; !last_step; step++)
   {
      if (Mpi::Root())
      {
         mfem::out << std::left << std::setw(16) << step << std::setw(16) << t << std::setw(16) << dt << std::setw(16);
      }

      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      temperature_solid_tn = *temperature_solid_gf->GetTrueDofs();
      temperature_cylinder_tn = *temperature_cylinder_gf->GetTrueDofs();
      temperature_fluid_tn = *temperature_fluid_gf->GetTrueDofs();

      // Inner loop for the segregated solve
      int iter = 0;
      double norm_diff = 2 * tol;
      // double dt_iter = dt / 10;
      // double t_iter = 0;

      while (!converged && iter <= max_iter)
      {
         /*

   // Store the previous temperature on the box wall to check convergence
   temperature_solid_prev = *(temperature_solid_gf->GetTrueDofs());
   temperature_wall_solid_prev_gf->SetFromTrueDofs(temperature_solid_prev);

   // Transfer k ∇T_wall from cylinder -> solid
   {
      ecm2_utils::GSLIBInterpolate(finder, *Heat_Cylinder.GetFESpace(), qoi_func, qoi_src, qoi_dst, qoi_size_on_qp);
      TransferQoIToDest(solid_element_idx, Heat_Solid.GetVectorFESpace(), qoi_dst, *k_grad_temperature_wall_gf, Kappa_solid_bdr);
   }

   // Advance the diffusion equation on the outer solid to the next time step
   temperature_solid_gf->SetFromTrueDofs(temperature_solid_tn);
   Heat_Solid.Step(t, dt, step, false);
   temperature_solid_gf->GetTrueDofs(temperature_solid);
   t -= dt; // Reset t to same time step, since t is incremented in the Step function

   // Relaxation
   // T_wall(j+1) = ω * T_solid,j+1 + (1 - ω) * T_solid,j
   if (iter > 0)
   {
      temperature_solid *= omega;
      temperature_solid.Add(1 - omega, temperature_solid_prev);
      temperature_solid_gf->SetFromTrueDofs(temperature_solid);
   }

   // Transfer temperature from solid -> cylinder
   temperature_solid_to_cylinder_map.Transfer(*temperature_solid_gf,
                                              *temperature_wall_cylinder_gf);

   // Advance the convection-diffusion equation on the outer solid to the
   // next time step
   temperature_cylinder_gf->SetFromTrueDofs(temperature_cylinder_tn);
   Heat_Cylinder.Step(t, dt, step, false);
   t -= dt; // Reset t to same time step, since t is incremented in the Step function

   // Check convergence
   //  || T_wall(j+1) - T_wall(j) || < tol
   norm_diff = temperature_solid_gf->ComputeL2Error(temperature_wall_solid_prev_coeff);
   converged = norm_diff < tol;

   iter++;

   // Output of subiterations   --> TODO: Create a new paraview collection for saving subiterations data (there's no copy constructor for ParaViewDataCollection)
   // t_iter += dt_iter;
   // Heat_Solid.WriteFields(iter, t_iter);
   // Heat_Cylinder.WriteFields(iter, t_iter);
   if (Mpi::Root())
   {
      convergence_subiter.Append(norm_diff);
   }
}

if (Mpi::Root())
{
   convergence(step - 1, 0) = t;
   convergence(step - 1, 1) = iter;
   convergence(step - 1, 2) = norm_diff;
   saveConvergenceSubiter(convergence_subiter, outfolder, step);
   convergence_subiter.DeleteAll();
}

// Reset the convergence flag and time for the next iteration
if (Mpi::Root())
{
   mfem::out << iter << std::endl;
}

if (iter > max_iter)
{
   if (Mpi::Root())
      mfem::out << "Warning: Maximum number of iterations reached. Error: " << norm_diff << " << Aborting!" << std::endl;
   break;
}

// Update time step history
Heat_Solid.UpdateTimeStepHistory();
Heat_Cylinder.UpdateTimeStepHistory();

temperature_solid_tn = *temperature_solid_gf->GetTrueDofs();
temperature_cylinder_tn = *temperature_cylinder_gf->GetTrueDofs();

t += dt;
converged = false;

// Output of time steps
if (paraview && (step % save_freq == 0))
{
   Heat_Solid.WriteFields(step, t);
   Heat_Cylinder.WriteFields(step, t);
*/
      }
   }

   // Save convergence data
   if (Mpi::Root())
   {
      std::string outputFilePath = std::string(outfolder) + "/convergence" + "/convergence.txt";
      std::ofstream outFile(outputFilePath);
      if (outFile.is_open())
      {
         convergence.Save(outFile);
         outFile.close();
      }
      else
      {
         std::cerr << "Unable to open file: " << std::string(outfolder) + "/convergence.txt" << std::endl;
      }
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   delete Kappa_cyl;
   delete Kappa_fluid;
   delete Kappa_solid;
   delete c_cyl;
   delete c_fluid;
   delete c_solid;
   delete rho_cyl;
   delete rho_fluid;
   delete rho_solid;
   delete q;
   delete Id;
   delete temperature_fc_cylinder;
   delete temperature_fc_fluid;
   delete temperature_sc_solid;
   delete temperature_sc_cylinder;
   delete temperature_fluid_fsc;
   delete heatFlux_fs_solid;
   delete heatFlux_fs_fluid;
   delete heatFlux_fc_cylinder;
   delete heatFlux_fc_fluid;
   delete heatFlux_sc_solid;
   delete heatFlux_sc_cylinder;
   delete temperature_solid_prev_gf;
   delete temperature_fluid_prev_gf;
   delete temperature_cylinder_prev_gf;
   delete velocity_gf;

   finder_solid_to_cylinder.FreeData();
   finder_fluid_to_cylinder.FreeData();
   finder_fluid_to_solid.FreeData();
   finder_cylinder_to_solid.FreeData();

   return 0;
}

void TransferQoIToDest(const std::vector<int> &elem_idx, const ParFiniteElementSpace &fes_grad, const Vector &grad_vec, ParGridFunction &grad_gf, MatrixCoefficient *K)
{

   int sdim = fes_grad.GetMesh()->SpaceDimension();
   int vdim = fes_grad.GetVDim();

   const IntegrationRule &ir_face = (fes_grad.GetBE(elem_idx[0]))->GetNodes();

   int dof, idx, be_idx, qp_idx;
   Vector grad_loc(vdim), k_grad_loc(vdim), loc_values;
   DenseMatrix Kmat(vdim);
   for (int be = 0; be < elem_idx.size(); be++) // iterate over each BE on interface boundary and construct FE value from quadrature point
   {
      Array<int> vdofs;
      be_idx = elem_idx[be];
      fes_grad.GetBdrElementVDofs(be_idx, vdofs);
      const FiniteElement *fe = fes_grad.GetBE(be_idx);
      ElementTransformation *Tr = fes_grad.GetBdrElementTransformation(be_idx);
      dof = fe->GetDof();
      loc_values.SetSize(dof * vdim);
      auto ordering = fes_grad.GetOrdering();
      for (int qp = 0; qp < dof; qp++)
      {
         qp_idx = be * dof + qp;
         // Evaluate diffusion tensor K at the quadrature point
         const IntegrationPoint &ip = ir_face.IntPoint(qp);
         K->Eval(Kmat, *Tr, ip);
         // if (Mpi::Root())
         //    print_matrix(Kmat);
         grad_loc = Vector(grad_vec.GetData() + qp_idx * vdim, vdim);
         Kmat.Mult(grad_loc, k_grad_loc);
         // k_grad_loc *= -1.0;
         for (int d = 0; d < vdim; d++)
         {
            idx = ordering == Ordering::byVDIM ? qp * sdim + d : dof * d + qp;
            loc_values(idx) = k_grad_loc(d);
         }
      }
      grad_gf.SetSubVector(vdofs, loc_values);
   }
}

void print_matrix(const DenseMatrix &A)
{
   std::cout << std::scientific;
   std::cout << "{";
   for (int i = 0; i < A.NumRows(); i++)
   {
      std::cout << "{";
      for (int j = 0; j < A.NumCols(); j++)
      {
         std::cout << A(i, j);
         if (j < A.NumCols() - 1)
         {
            std::cout << ", ";
         }
      }
      if (i < A.NumRows() - 1)
      {
         std::cout << "}, ";
      }
      else
      {
         std::cout << "}";
      }
   }
   std::cout << "}\n";
   std::cout << std::fixed;
   std::cout << std::endl
             << std::flush; // Debugging print
}

void saveConvergenceSubiter(const Array<real_t> &convergence_subiter, const std::string &outfolder, int step)
{
   // Create the output folder path
   std::string outputFolder = outfolder + "/convergence";

   // Ensure the directory exists
   if ((mkdir(outputFolder.c_str(), 0777) == -1) && Mpi::Root())
   {
      // check error
   }

   // Construct the filename
   std::ostringstream filename;
   filename << outputFolder << "/step_" << step << ".txt";

   // Open the file and save the data
   std::ofstream outFile(filename.str());
   if (outFile.is_open())
   {
      convergence_subiter.Save(outFile);
      outFile.close();
   }
   else
   {
      std::cerr << "Unable to open file: " << filename.str() << std::endl;
   }
}

void velocity_profile(const Vector &c, Vector &q)
{
   real_t A = 1.0;
   real_t x = c(0);
   real_t y = c(1);
   real_t r = sqrt(pow(x, 2.0) + pow(y, 2.0));

   q(0) = 0.0;
   q(1) = 0.0;

   if (std::abs(r) >= R - 1e-8)
   {
      q(2) = 0.0;
   }
   else
   {
      // q(2) = A * exp(-(pow(x, 2.0) / 2.0 + pow(y, 2.0) / 2.0));
      q(2) = 1 / (4 * viscosity) * dPdx * (R * R - r * r);
   }
}

double HeatingSphere(const Vector &x, double t)
{
   double Q = 0.0;
   double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
   Q = r < Sphere_Radius ? Qval : 0.0; // W/m^2

   return Q;
}