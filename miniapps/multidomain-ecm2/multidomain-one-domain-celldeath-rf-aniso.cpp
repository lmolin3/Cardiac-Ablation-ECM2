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
//
// Sample run:
// mpirun -np 4 ./multidomain-one-domain-celldeath-rf-aniso -hex -pa_heat -pa_rf -oh 3 -or 4 -ode 1 -tf 1.0 -dt 0.01 -paraview -of ./Output/Solid/oh3_or4


#include "mfem.hpp"
#include "lib/heat_solver.hpp"
#include "lib/celldeath_solver.hpp"
#include "lib/electrostatics_solver.hpp"

#include <fstream>
#include <sstream>
#include <sys/stat.h> // Include for mkdir
#include <iostream>
#include <memory>

using namespace mfem;

IdentityMatrixCoefficient *Id = NULL;

// Forward declaration
void print_matrix(const DenseMatrix &A);

static real_t T_solid = 37; // 37.0;   // Body temperature

static real_t phi_applied = 10;
static real_t phi_gnd = 0.0;

real_t mesh_scale = 1e2; // cm 

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
   int order_heat = 1;
   int order_rf = 1;
   int order_celldeath = -1;
   bool pa_heat = false; // Enable partial assembly for HeatSolver
   bool pa_rf = false;   // Enable partial assembly for ElectrostaticsSolver
   int celldeath_solver_type = 0; // 0: Eigen, 1: GoTran
   // Mesh
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool hex = false;
   // Time integrator
   int ode_solver_type = 1;
   real_t t_final = 1.0;
   real_t dt = 1.0e-2;
   // Postprocessing
   bool print_timing = false;
   bool visit = false;
   bool paraview = true;
   int save_freq = 1; // Save fields every 'save_freq' time steps
   const char *outfolder = "./Output/Test";
   bool save_convergence = false;

   OptionsParser args(argc, argv);
   // FE
   args.AddOption(&order_heat, "-oh", "--order-heat",
                  "Finite element order for heat transfer (polynomial degree).");
   args.AddOption(&order_rf, "-or", "--order-rf",
                  "Finite element order for RF problem (polynomial degree).");
   args.AddOption(&order_celldeath, "-oc", "--order-celldeath",
                  "Finite element order for cell death (polynomial degree).");
   args.AddOption(&pa_heat, "-pa_heat", "--partial-assembly-heat", "-no-pa_heat", "--no-partial-assembly-heat",
                  "Enable or disable partial assembly.");
   args.AddOption(&pa_rf, "-pa_rf", "--partial-assembly-rf", "-no-pa_rf", "--no-partial-assembly-rf",
                  "Enable or disable partial assembly for RF problem.");
   args.AddOption(&celldeath_solver_type, "-cdt", "--celldeath-solver-type",
                  "Cell-death solver type: 0 - Eigen, 1 - GoTran.");
   // Mesh
   args.AddOption(&hex, "-hex", "--hex-mesh", "-tet", "--tet-mesh",
                  "Use hexahedral mesh.");
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
   // Physics
   args.AddOption(&phi_applied, "-phi", "--applied-potential",
                  "Applied potential.");
   // Postprocessing
   args.AddOption(&print_timing, "-pt", "--print-timing", "-no-pt", "--no-print-timing",
                  "Print timing data.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&save_freq, "-sf", "--save-freq",
                  "Save fields every 'save_freq' time steps.");
   args.AddOption(&outfolder, "-of", "--out-folder",
                  "Output folder.");

   args.ParseCheck();

   // Determine order for cell death problem
   if (order_celldeath < 0)
   {
      order_celldeath = order_heat;
   }

   // Convert temperature to kelvin
   T_solid = heat::CelsiusToKelvin(T_solid);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nLoading mesh... \033[0m";

   // Load serial mesh
   Mesh *serial_mesh = nullptr;
   if (hex)
   { // Load Hex mesh (NETCDF required)
#ifdef MFEM_USE_NETCDF
      serial_mesh = new Mesh("../../data/three-domains.e");
#else
      MFEM_ABORT("MFEM is not built with NetCDF support!");
#endif
   }
   else
   {
      serial_mesh = new Mesh("../../data/three-domains.msh");
   }

   int sdim = serial_mesh->SpaceDimension();

   for (int l = 0; l < serial_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   // Generate mesh partitioning

   if (Mpi::Root())
      mfem::out << "Generating partitioning and creating parallel mesh... \033[0m";

   // Partition type:
   // 0) METIS_PartGraphRecursive (sorted neighbor lists)
   // 1) METIS_PartGraphKway      (sorted neighbor lists) (default)
   // 2) METIS_PartGraphVKway     (sorted neighbor lists)
   // 3) METIS_PartGraphRecursive
   // 4) METIS_PartGraphKway
   // 5) METIS_PartGraphVKway
   int partition_type = 1;
   int np = num_procs;
   int *partitioning = serial_mesh->GeneratePartitioning(np, partition_type);

   // Create parallel mesh
   ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh, partitioning, partition_type);
   // ExportMeshwithPartitioning(outfolder, *serial_mesh, partitioning);
   delete[] partitioning;
   delete serial_mesh;

   for (int l = 0; l < parallel_ref_levels; l++)
   {
      parent_mesh.UniformRefinement();
   }

   parent_mesh.EnsureNodes();

   if (Mpi::Root())
   {
      mfem::out << "\033[34mdone." << std::endl;
      mfem::out << "Creating sub-meshes... \033[0m";
   }

   // Create the sub-domains for the cylinder, solid and fluid domains
   AttributeSets &attr_sets = parent_mesh.attribute_sets;
   AttributeSets &bdr_attr_sets = parent_mesh.bdr_attribute_sets;

   Array<int> solid_domain_attribute;
   Array<int> fluid_domain_attribute;
   Array<int> cylinder_domain_attribute;

   if (hex)
   {
      solid_domain_attribute.SetSize(1);
      fluid_domain_attribute.SetSize(1);
      cylinder_domain_attribute.SetSize(1);
      solid_domain_attribute = 3;
      fluid_domain_attribute = 1;
      cylinder_domain_attribute = 2;
   }
   else
   {
      solid_domain_attribute = attr_sets.GetAttributeSet("Solid");
      fluid_domain_attribute = attr_sets.GetAttributeSet("Fluid");
      cylinder_domain_attribute = attr_sets.GetAttributeSet("Cylinder");
   }

   auto solid_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, solid_domain_attribute));

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Set up coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up coefficients... \033[0m" << std::endl;

   auto Id = new IdentityMatrixCoefficient(sdim);

   // Heat Transfer
   if (Mpi::Root())
      mfem::out << "\033[0mHeat transfer problem... \033[0m";

   real_t cval_solid, rhoval_solid, kval_solid;
   cval_solid = 3017;                            // J/kgK --> cm
   rhoval_solid = 1076/std::pow(mesh_scale,3) ;  // kg/m^3 --> cm
   kval_solid = 0.518/mesh_scale;                // W/mK --> cm

   // Conductivity
   // NOTE: if using PWMatrixCoefficient you need to create one for the boundary too
   auto *Kappa_solid = new ScalarMatrixProductCoefficient(kval_solid, *Id);

   // Heat Capacity
   auto *c_solid = new ConstantCoefficient(cval_solid);

   // Density
   auto *rho_solid = new ConstantCoefficient(rhoval_solid);

   if (Mpi::Root())
      mfem::out << "\033[0mdone." << std::endl;

   // Cell Death Problem
   if (Mpi::Root())
      mfem::out << "\033[0mCell death problem... \033[0m";
   real_t A1 = 3.68*1e30;
   real_t A2 = 5.68*1e3;
   real_t A3 = 2.58*1e5;
   real_t deltaE1 = 210*1e3;
   real_t deltaE2 = 38.6*1e3;
   real_t deltaE3 = 47.2*1e3;
   if (Mpi::Root())
      mfem::out << "\033[0mdone." << std::endl;

   // RF Problem
   if (Mpi::Root())
      mfem::out << "\033[0mRF problem... \033[0m";

   real_t sigma_solid = 0.54/mesh_scale; // S/m --> cm

   // Conductivity
   // NOTE: if using PWMatrixCoefficient you need to create one for the boundary too
   auto *Sigma_solid = new ScalarMatrixProductCoefficient(sigma_solid, *Id);

   if (Mpi::Root())
      mfem::out << "\033[0mdone." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create BC Handler (not populated yet)
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nCreating BCHandlers and Solvers... \033[0m";

   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool bc_verbose = true;

   // Heat Transfer
   heat::BCHandler *heat_bcs_solid = new heat::BCHandler(solid_submesh, bc_verbose); // Boundary conditions handler for solid

   // RF
   electrostatics::BCHandler *rf_bcs_solid = new electrostatics::BCHandler(solid_submesh, bc_verbose); // Boundary conditions handler for solid

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Create the Solvers
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Solvers
   bool solv_verbose = false;
   heat::HeatSolver Heat_Solid(solid_submesh, order_heat, heat_bcs_solid, Kappa_solid, c_solid, rho_solid, ode_solver_type, solv_verbose);

   electrostatics::ElectrostaticsSolver RF_Solid(solid_submesh, order_rf, rf_bcs_solid, Sigma_solid, solv_verbose);

   // Grid functions in domain (inside solver)
   ParGridFunction *temperature_solid_gf = Heat_Solid.GetTemperatureGfPtr();
   ParGridFunction *phi_solid_gf = RF_Solid.GetPotentialGfPtr();

   // Cell Death solver (needs pointer to temperature grid function)
   celldeath::CellDeathSolver *CellDeath_Solid = nullptr;
   if (celldeath_solver_type == 0)
      CellDeath_Solid = new celldeath::CellDeathSolverEigen(solid_submesh, order_celldeath, temperature_solid_gf, A1, A2, A3, deltaE1, deltaE2, deltaE3);
   else if (celldeath_solver_type == 1)
      CellDeath_Solid = new celldeath::CellDeathSolverGotran(solid_submesh, order_celldeath, temperature_solid_gf, A1, A2, A3, deltaE1, deltaE2, deltaE3);
   else
      MFEM_ABORT("Invalid cell death solver type.");

   // Finite element spaces
   ParFiniteElementSpace *rf_fes_l2_solid = RF_Solid.GetL2FESpace();

   // Auxiliary grid functions
   ParGridFunction *JouleHeating_gf = new ParGridFunction(rf_fes_l2_solid);
   *JouleHeating_gf = 0.0;
   Coefficient *JouleHeating_coeff = RF_Solid.GetJouleHeatingCoefficient();

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Populate BC Handler
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Array<int> fluid_solid_interface;
   Array<int> solid_cylinder_interface;

   Array<int> solid_lateral_attr;
   Array<int> solid_bottom_attr;

   if (hex)
   {
      // Extract boundary attributes
      fluid_solid_interface.SetSize(1);
      fluid_solid_interface = 3;
      solid_cylinder_interface.SetSize(1);
      solid_cylinder_interface = 4;

      solid_lateral_attr.SetSize(1);
      solid_lateral_attr = 2;
      solid_bottom_attr.SetSize(1);
      solid_bottom_attr = 1;
   }
   else
   {
      // Extract boundary attributes
      fluid_solid_interface = bdr_attr_sets.GetAttributeSet("Solid-Fluid");
      solid_cylinder_interface = bdr_attr_sets.GetAttributeSet("Cylinder-Solid");

      solid_lateral_attr = bdr_attr_sets.GetAttributeSet("Solid Lateral");
      solid_bottom_attr = bdr_attr_sets.GetAttributeSet("Solid Bottom");
   }

   Array<int> solid_domain_attributes = AttributeSets::AttrToMarker(solid_submesh->attributes.Max(), solid_domain_attribute);

   // NOTE: each submesh requires a different marker set as bdr attributes are generated per submesh (size can be different)
   // They can be converted as below using the attribute sets and the max attribute number for the specific submesh
   // If you don't want to convert and the attribute is just one number, you can add bcs or volumetric terms using the int directly (will take care of creating the marker array)
   // Array<int> fluid_cylinder_interface_c = AttributeSets::AttrToMarker(cylinder_submesh->bdr_attributes.Max(), fluid_cylinder_interface);
   // Array<int> fluid_cylinder_interface_f = AttributeSets::AttrToMarker(fluid_submesh->bdr_attributes.Max(), fluid_cylinder_interface);

   /////////////////////////////////////
   //           Heat Transfer         //
   /////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for Heat Transfer ...\033[0m";

   // Solid:
   // - T = T_solid on bottom/lateral walls
   // - Neumann   on  Γsc --> TODO ROBIN after
   // - Neumann   on  Γfs
   if (Mpi::Root())
      mfem::out << "\033[0m\nSetting up BCs for solid domain...\033[0m" << std::endl;

   heat_bcs_solid->AddDirichletBC(T_solid, solid_lateral_attr[0]);
   heat_bcs_solid->AddDirichletBC(T_solid, solid_bottom_attr[0]);

   Heat_Solid.AddVolumetricTerm(JouleHeating_coeff, solid_domain_attributes, false); // does not assume ownership of the coefficient

   /////////////////////////////////////
   //                RF               //
   /////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for RF ...\033[0m";

   // Solid:
   // - Phi = 0 on bottom wall
   // - Dirichlet   on  Γsc
   // - Homogeneous Neumann lateral wall
   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up RF BCs for solid domain...\033[0m" << std::endl;
   rf_bcs_solid->AddDirichletBC(phi_gnd, solid_bottom_attr[0]);
   rf_bcs_solid->AddDirichletBC(phi_applied, solid_cylinder_interface[0]);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up solvers and assembling forms... \033[0m" << std::endl;

   if (Mpi::Root())
      mfem::out << "\033[0mHeat Transfer problem ... \033[0m";

   StopWatch chrono_assembly;
   chrono_assembly.Start();
   Heat_Solid.EnablePA(pa_heat);
   Heat_Solid.Setup(dt);
   chrono_assembly.Stop();
   if (Mpi::Root())
      mfem::out << "\033[0mdone, in " << chrono_assembly.RealTime() << "s.\033[0m" << std::endl;

   if (Mpi::Root())
      mfem::out << "\033[0mRF problem... \033[0m";

   chrono_assembly.Clear();
   chrono_assembly.Start();
   RF_Solid.EnablePA(pa_rf);
   RF_Solid.Setup();
   chrono_assembly.Stop();

   if (Mpi::Root())
      mfem::out << "\033[0mdone, in " << chrono_assembly.RealTime() << "s.\033[0m" << std::endl;

   // Setup ouput
   ParaViewDataCollection paraview_dc_solid_heat("Heat-Solid", solid_submesh.get());
   ParaViewDataCollection paraview_dc_celldeath("CellDeath-Solid", solid_submesh.get());
   ParaViewDataCollection paraview_dc_solid_rf("RF-Solid", solid_submesh.get());
   if (paraview)
   {
      paraview_dc_solid_heat.SetPrefixPath(outfolder);
      paraview_dc_solid_heat.SetDataFormat(VTKFormat::ASCII);
      paraview_dc_solid_heat.SetCompressionLevel(9);
      Heat_Solid.RegisterParaviewFields(paraview_dc_solid_heat);

      paraview_dc_celldeath.SetPrefixPath(outfolder);
      paraview_dc_celldeath.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_celldeath.SetCompressionLevel(9);
      CellDeath_Solid->RegisterParaviewFields(paraview_dc_celldeath);

      paraview_dc_solid_rf.SetPrefixPath(outfolder);
      paraview_dc_solid_rf.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_solid_rf.SetCompressionLevel(9);
      RF_Solid.RegisterParaviewFields(paraview_dc_solid_rf);
      RF_Solid.AddParaviewField("Joule Heating", JouleHeating_gf);
   }

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Run Simulation
   //     - Solve RF
   //     For each time step:
   //       - Solve RF (if needed)
   //       - Solve Heat Transfer
   //       - Solve Cell Death
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ///////////////////////////////////////////////////
   // Initial conditions and setup of vectors
   ///////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nStarting time-integration... \033[0m" << std::endl;

   // Initial conditions
   ConstantCoefficient T0solid(T_solid);
   temperature_solid_gf->ProjectCoefficient(T0solid);
   Heat_Solid.SetInitialTemperature(*temperature_solid_gf);

   // Write fields to disk for VisIt
   if (paraview)
   {
      Heat_Solid.WriteFields(0, 0.0);
      CellDeath_Solid->WriteFields(0, 0.0);
      RF_Solid.WriteFields(0, 0.0);
   }

   // Dofs
   int heat_solid_dofs = Heat_Solid.GetProblemSize();
   int celldeath_solid_dofs = CellDeath_Solid->GetProblemSize();
   int rf_solid_dofs = RF_Solid.GetProblemSize();

   // Integration rule for the L2 error
   int order_quad_heat = std::max(2, 2 * order_heat + 2);
   const IntegrationRule *irs_heat[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs_heat[i] = &(IntRules.Get(i, order_quad_heat));
   }

   int order_quad_rf = std::max(2, 2 * order_rf + 2);
   const IntegrationRule *irs_rf[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs_rf[i] = &(IntRules.Get(i, order_quad_rf));
   }

   if (Mpi::Root())
   {
      out << " Solid dofs (Heat): " << heat_solid_dofs << std::endl;
      out << " Solid dofs (CellDeath): " << celldeath_solid_dofs << std::endl;
      out << " Solid dofs (RF): " << rf_solid_dofs << std::endl;
   }

   ///////////////////////////////////////////////////
   // Solve RF before time integration
   ///////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[31m\nSolving RF... \033[0m";

   StopWatch chrono;
   {
      // Timing
      real_t t_solve_solid, t_paraview, t_joule;

      chrono.Clear();
      chrono.Start();
      RF_Solid.Solve();
      chrono.Stop();
      t_solve_solid = chrono.RealTime();

      if (Mpi::Root())
         mfem::out << "\033[31mdone.\033[0m" << std::endl;


      // Compute Joule heating
      chrono.Clear();
      chrono.Start();
      // Output of time steps
      RF_Solid.GetJouleHeating(*JouleHeating_gf);
      chrono.Stop();
      t_joule = chrono.RealTime();

      // Export converged fields
      chrono.Clear();
      chrono.Start();
      if (paraview)
      {
         RF_Solid.WriteFields(0, 0.0);
      }
      chrono.Stop();
      t_paraview = chrono.RealTime();

      if (Mpi::Root() && print_timing)
      { // Print times
         out << "------------------------------------------------------------" << std::endl;
         out << "Solid Solve: " << t_solve_solid << " s" << std::endl;
         out << "Joule: " << t_joule << " s" << std::endl;
         out << "Paraview: " << t_paraview << " s" << std::endl;
         out << "------------------------------------------------------------" << std::endl;
      }
   }


   ///////////////////////////////////////////////////
   // Outer loop for time integration
   ///////////////////////////////////////////////////

   // Timing
   StopWatch chrono_total;

   if (Mpi::Root())
   {
      out << "-----------------------------------------------"
          << std::endl;
      out << std::left << std::setw(16) << "Step" << std::setw(16) << "Time" << std::setw(16) << "dt" << std::endl;
      out << "-----------------------------------------------"
          << std::endl;                                      
   }

   real_t t = 0.0;
   bool last_step = false;
   bool converged = false;
   real_t tol = 1.0e-4;
   int max_iter = 100;

   // Timing
   real_t t_solve_solid, t_solve_celldeath, t_paraview;

   for (int step = 1; !last_step; step++)
   {
      chrono_total.Clear();
      chrono_total.Start();
      if (Mpi::Root())
      {
         mfem::out << std::left << std::setw(16) << step << std::setw(16) << t << std::setw(16) << dt << std::setw(16) << std::endl;
      }

      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      /////////////////////////////////////////
      //              Solve RF               //
      /////////////////////////////////////////
      {
      }

      /////////////////////////////////////////
      //         Solve HEAT TRANSFER         //
      /////////////////////////////////////////
      if (Mpi::Root())
         mfem::out << "\033[34mSolving HeatTransfer problem... " << "\033[0m";

      // Step in the solid domain
      chrono.Clear();
      chrono.Start();
      Heat_Solid.Step(t, dt, step);
      t -= dt;
      chrono.Stop();
      t_solve_solid = chrono.RealTime();

      if (Mpi::Root())
         mfem::out << "\033[34mdone.\033[0m" << std::endl;

      /////////////////////////////////////////
      //           Solve CELL DEATH          //
      /////////////////////////////////////////

      chrono.Clear();
      chrono.Start();
      if (Mpi::Root())
         mfem::out << "\033[32mSolving CellDeath problem on solid ... \033[0m";

      CellDeath_Solid->Solve(t, dt);

      if (Mpi::Root())
         mfem::out << "\033[32mdone.\033[0m" << std::endl;
      chrono.Stop();
      t_solve_celldeath = chrono.RealTime();

      ////////////////////////////////////
      //         Postprocessing         //
      ////////////////////////////////////

      // Output of time steps
      chrono.Clear();
      chrono.Start();
      if (paraview && (step % save_freq == 0))
      {
         Heat_Solid.WriteFields(step, t);
         CellDeath_Solid->WriteFields(step, t);
      }
      chrono.Stop();
      t_paraview = chrono.RealTime();

      chrono_total.Stop();

      if (Mpi::Root() && print_timing)
      {
         out << "------------------------------------------------------------" << std::endl;
         out << "Solve Heat (Solid): " << t_solve_solid << " s" << std::endl;
         out << "Solve CellDeath (Solid): " << t_solve_celldeath << " s" << std::endl;
         out << "Paraview: " << t_paraview << " s" << std::endl;
         out << "Total: " << chrono_total.RealTime() << " s" << std::endl;
         out << "------------------------------------------------------------" << std::endl;
      }

      // Update time
      t += dt;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // No need to delete:
   // - BcHandler objects: owned by HeatOperator
   // - Coefficients provide to BCHandler: owned by CoeffContainers
   delete Id;

   delete CellDeath_Solid;

   delete rho_solid;
   delete Kappa_solid;
   delete c_solid;

   delete Sigma_solid;

   delete JouleHeating_gf;

   return 0;
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