// Solve RFA (Electrostatics+HeatTransfer+NavierStokes) problem with three domains (solid, fluid, cylinder).
//                 rho c dT/Sim_ctx.dt = ∇•κ∇T - α•∇T        in  fluid
//                 rho c dT/Sim_ctx.dt = ∇•κ∇T               in  solid
//                 rho c dT/Sim_ctx.dt = ∇•κ∇T + Q           in  cylinder
// Q is a Joule heating term (W/m^3) and is defined as Q = J • E, where J is the current density and E is the electric field.
// The Electrostatic problem is solved on the solid and fluid domains, before the temporal loop. 
// The heat transfer problem is solved using a segregated approach with two-way coupling (Neumann-Dirichlet)
// until convergence is reached.
// After the heat transfer problem is solved, a three-state cell death model is solved in the solid domain.
//                  dN/Sim_ctx.dt = k2U - k1N
//                  dU/Sim_ctx.dt = k1N - k2U - k3U
//                  dD/Sim_ctx.dt = k3U  
// with initial conditions N(0) = 1, U(0) = 0, D(0) = 0.
// The coefficients are defined as ki = Ai * exp(-ΔEi/RT) 
//
// Works on both hexahedral and tetrahedral meshes, with optional partial assembly (hexahedral only).
// Both implicit and explicit time integrators work with partial assembly.
// Potentially each domain can have different physics (advection α • u ∇T, diffusion ∇•κ∇T, reaction β T).
// The conductivity tensor κ can be anisotropic (Change EulerAngles function).
//
// Sample run:
// 1. Tetrahedral mesh
//    mpirun -np 4 ./multidomain-three-domains-celldeath-rf-navier-osm -tet -oh 2 -or 3 -dt 0.01 -tf 0.05 -ta 1 --preload-ns 2.0 -alpha-rf '1e3 1e-3' -alpha-h '1e3 1e3 1e-3 1e-3 1e-3 1e3' -omegat-rf 0.5
// 2. Hexahedral mesh  
//    mpirun -np 4 ./multidomain-three-domains-celldeath-rf-navier-osm -hex -oh 2 -or 3 -dt 0.01 -tf 0.05 -ta 1 --preload-ns 2.0 -alpha-rf '1e3 1e-3' -alpha-h '1e3 1e3 1e-3 1e-3 1e-3 1e3' -omegat-rf 0.5
// 3. Hexahedral mesh with partial assembly for RF
//    mpirun -np 4 ./multidomain-three-domains-celldeath-rf-navier-osm -hex -pa-heat -pa-rf -oh 2 -or 3 -dt 0.01 -tf 0.05 -ta 1 --preload-ns 2.0 -alpha-rf '1e3 1e-3' -alpha-h '1e3 1e3 1e-3 1e-3 1e-3 1e3' -omegat-rf 0.5
// 4. Hexahedral mesh with partial assembly for RF and Heat
//    mpirun -np 4 ./multidomain-three-domains-celldeath-rf-navier-osm -hex -pa-heat -pa-rf -oh 2 -or 3 -dt 0.01 -tf 0.05 -ta 1 --preload-ns 2.0 -alpha-rf '1e3 1e-3' -alpha-h '1e3 1e3 1e-3 1e-3 1e-3 1e3' -omegat-rf 0.5
// 5. Hexahedral mesh with partial assembly for RF and Heat, and anisotropic conductivity
//    mpirun -np 4 ./multidomain-three-domains-celldeath-rf-navier-osm -hex -pa-heat -pa-rf -oh 2 -or 3 -dt 0.01 -tf 0.05 --aniso-ratio-rf 1.0 --aniso-ratio-heat 1.0 -omegat-h 0.8 -omegat-rf 0.6 -ta 1 --preload-ns 2.0 -alpha-rf '1e2 -1e-2' -alpha-h '1e3 1e3 1e-3 1e-3 1e-3 1e3' -omegat-rf 0.5

// MFEM library
#include "mfem.hpp"

// Multiphysics modules
#include "lib/heat_solver.hpp"
#include "lib/celldeath_solver.hpp"
#include "lib/electrostatics_solver.hpp"
#include "lib/navier_solver.hpp"

// Interface transfer
#include "interface_transfer.hpp"

// Physical and Domain-Decomposition parameters
#include "contexts.hpp" 

// Utils
#include "anisotropy_utils.hpp"

// Output
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include "FilesystemHelper.hpp"


using namespace mfem;
using TransferBackend = InterfaceTransfer::Backend;

IdentityMatrixCoefficient *Id = NULL;
std::function<void(const Vector &, Vector &)> EulerAngles(real_t zmax, real_t zmin);

// Forward declaration
void print_matrix(const DenseMatrix &A);
void saveConvergenceArray(const Array2D<real_t> &data, const std::string &outfolder, const std::string &name, int step);
void saveSubiterationCount(const Array<int> &data, const std::string &outfolder, const std::string &name);

void inflow(const Vector &x, real_t t, Vector &u)
{
   u = 0.0;
   u(1) = Navier_ctx.u_inflow;
}

// Variables for domain decomposition convergence
static constexpr int MAX_ITER = 100;
static constexpr double TOL = 1e-8;
static constexpr double TOL_HEAT = 1e-4;


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

   real_t preload_ns = -1.0;

   OptionsParser args(argc, argv);
   // FE
   args.AddOption(&Heat_ctx.order, "-oh", "--order-heat",
                  "Finite element order for heat transfer (polynomial degree).");
   args.AddOption(&RF_ctx.order, "-or", "--order-rf",
                  "Finite element order for RF problem (polynomial degree).");
   args.AddOption(&CellDeath_ctx.order, "-oc", "--order-celldeath",
                  "Finite element order for cell death (polynomial degree).");
   args.AddOption(&Heat_ctx.pa, "-pa-heat", "--partial-assembly-heat", "-no-pa-heat", "--no-partial-assembly-heat",
                  "Enable or disable partial assembly.");
   args.AddOption(&RF_ctx.pa, "-pa-rf", "--partial-assembly-rf", "-no-pa-rf", "--no-partial-assembly-rf",
                  "Enable or disable partial assembly for RF problem.");   
   args.AddOption(&CellDeath_ctx.solver_type, "-cdt", "--celldeath-solver-type",
                  "Cell-death solver type: 0 - Eigen, 1 - GoTran.");
   // Mesh
   args.AddOption(&Mesh_ctx.hex, "-hex", "--hex-mesh", "-tet", "--tet-mesh",
                  "Use hexahedral mesh.");
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   // Physics
   args.AddOption(&RF_ctx.aniso_ratio, "-ar", "--aniso-ratio-rf",
                  "Anisotropy ratio for RF problem.");
   args.AddOption(&Heat_ctx.aniso_ratio, "-at", "--aniso-ratio-heat",
                  "Anisotropy ratio for temperature problem."); 
   args.AddOption(&Navier_ctx.u_inflow, "-ui", "--u-inflow",
                  "Inflow velocity for Navier-Stokes problem.");
   args.AddOption(&RF_ctx.phi_applied, "-phi", "--applied-potential",
                  "Applied potential.");
   // Time integrator
   args.AddOption(&Heat_ctx.ode_solver_type, "-ode", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   4 - Implicit Midpoint, 5 - SDIRK23, 6 - SDIRK34,\n\t"
                  "\t   7 - Forward Euler, 8 - RK2, 9 - RK3 SSP, 10 - RK4.");
   args.AddOption((int *)&Navier_ctx.time_adaptivity_type, 
                  "-ta",
                  "--time-adaptivity",
                  "Time adaptivity type (0: None, 1: CFL, 2: HOPC)");
   args.AddOption(&Sim_ctx.t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&Sim_ctx.dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&preload_ns, "-preload-ns", "--preload-ns",
                  "Preload time for Navier-Stokes problem.");
   // Domain decomposition
   args.AddOption(&DD_ctx.omega_heat, "-omegat-h", "--relaxation-parameter-heat",
                  "Relaxation parameter.");
   args.AddOption(&DD_ctx.omega_rf, "-omegat-rf", "--relaxation-parameter-rf",
                  "Relaxation parameter for RF problem.");
   args.AddOption(&DD_ctx.alpha_rf, "-alpha-rf", "--alpha-robin-rf",
                     "Robin-Robin coupling parameters for rf problem (FS_fluid, FS_Solid).");
   args.AddOption(&DD_ctx.alpha_heat, "-alpha-h", "--alpha-robin-heat",
                        "Robin-Robin coupling parameters for heat problem (FS_fluid, FC_fluid, FS_solid, SC_solid, FC_cylinder, SC_cylinder). ");
   // Postprocessing
   args.AddOption(&Sim_ctx.print_timing, "-pt", "--print-timing", "-no-pt", "--no-print-timing",
                  "Print timing data.");
   args.AddOption(&Sim_ctx.paraview, "--paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable Paraview visualization.");
   args.AddOption(&Sim_ctx.save_freq, "-sf", "--save-freq",
                  "Save fields every 'save_freq' time steps.");
   args.AddOption(&Sim_ctx.outfolder, "-of", "--out-folder",
                  "Output folder.");
   args.AddOption(&Sim_ctx.save_convergence, "-sc", "--save-convergence", "-no-sc", "--no-save-convergence",
                  "Save convergence data.");

   args.ParseCheck();

   // Determine order for cell death problem
   if (CellDeath_ctx.order < 0)
   {
      CellDeath_ctx.order = Heat_ctx.order;
   }

   // Convert temperature to kelvin
   Heat_ctx.T_solid = heat::CelsiusToKelvin(Heat_ctx.T_solid);
   Heat_ctx.T_fluid =  heat::CelsiusToKelvin(Heat_ctx.T_fluid);
   Heat_ctx.T_cylinder =  heat::CelsiusToKelvin(Heat_ctx.T_cylinder);

   // Set the relaxation parameters
   //if (RF_ctx.aniso_ratio > 1.5 && DD_ctx.omega_rf > 0.5)
   //{  
   //   if (Mpi::Root())
   //      mfem::out << "\033[31mAnisotropic RF problem detected. Reducing relaxation parameter to 0.5.\033[0m" << std::endl;  
   //   DD_ctx.omega_rf = 0.5;
   //}

   DD_ctx.omega_rf_fluid = DD_ctx.omega_rf; // TODO: Add different relaxation parameters for each domain
   DD_ctx.omega_rf_solid = DD_ctx.omega_rf;
   DD_ctx.omega_rf_cyl = DD_ctx.omega_rf;

   DD_ctx.omega_heat_fluid = DD_ctx.omega_heat; // TODO: Add different relaxation parameters for each domain
   DD_ctx.omega_heat_solid = DD_ctx.omega_heat;
   DD_ctx.omega_heat_cyl = DD_ctx.omega_heat;


   // Check on provided alpha values
   if (DD_ctx.alpha_rf.Size() != 2)
   {
      MFEM_ABORT("2 coupling parameters must be provided for Robin-Robin coupling in RF problem, but " << DD_ctx.alpha_rf.Size() << " were provided.");
   }

   if (DD_ctx.alpha_heat.Size() != 6)
   {
      MFEM_ABORT("6 coupling parameters must be provided for Robin-Robin coupling in heat problem, but " << DD_ctx.alpha_heat.Size() << " were provided.");
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nLoading mesh... \033[0m";

   // Load serial mesh
   Mesh *serial_mesh = nullptr;
   if (Mesh_ctx.hex)
   { // Load Hex mesh (NETCDF required)
#ifdef MFEM_USE_NETCDF
      serial_mesh = new Mesh("../../../data/three-domains-navier.e");
#else
      MFEM_ABORT("MFEM is not built with NetCDF support!");
#endif
   }
   else
   {
      serial_mesh = new Mesh("../../../data/three-domains-navier-tet.e");
   }

   int sdim = serial_mesh->SpaceDimension();

   for (int l = 0; l < Mesh_ctx.serial_ref_levels; l++)
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
   // ExportMeshwithPartitioning(Sim_ctx.outfolder, *serial_mesh, partitioning);
   delete[] partitioning;
   delete serial_mesh;

   for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
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

   if (Mesh_ctx.hex)
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

   auto fluid_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, fluid_domain_attribute));

   auto cylinder_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, cylinder_domain_attribute));


   Vector pmin, pmax;
   solid_submesh->GetBoundingBox(pmin, pmax);
   real_t zmin = pmin[2];
   real_t zmax = pmax[2];


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

   real_t reaction = 0.0;   // Reaction term

   // Conductivity
   // NOTE: if using PWMatrixCoefficient you need to create one for the boundary too
   auto *Kappa_cyl = new ScalarMatrixProductCoefficient(Heat_ctx.k_cylinder, *Id);
   auto *Kappa_fluid = new ScalarMatrixProductCoefficient(Heat_ctx.k_fluid, *Id);

   Vector k_vec_solid(3);
   k_vec_solid[0] = Heat_ctx.k_solid;                               // Along fibers
   k_vec_solid[1] = Heat_ctx.k_solid/Heat_ctx.aniso_ratio;       // Sheet direction 
   k_vec_solid[2] = Heat_ctx.k_solid/Heat_ctx.aniso_ratio;       // Sheet Normal to fibers
   auto *Kappa_solid = new MatrixFunctionCoefficient(3, ConductivityMatrix(k_vec_solid, EulerAngles(zmax, zmin)));

   // Heat Capacity
   auto *c_cyl = new ConstantCoefficient(Heat_ctx.c_cylinder);
   auto *c_fluid = new ConstantCoefficient(Heat_ctx.c_fluid);
   auto *c_solid = new ConstantCoefficient(Heat_ctx.c_solid);

   // Density
   auto *rho_cyl = new ConstantCoefficient(Heat_ctx.rho_cylinder);
   auto *rho_fluid = new ConstantCoefficient(Heat_ctx.rho_fluid);
   auto *rho_solid = new ConstantCoefficient(Heat_ctx.rho_solid);

   if (Mpi::Root())
      mfem::out << "\033[0mdone." << std::endl;

   // RF Problem
   if (Mpi::Root())
      mfem::out << "\033[0mRF problem... \033[0m";

   real_t sigma_fluid = 1.0;
   real_t sigma_solid = 1.0;

   // Conductivity
   // NOTE: if using PWMatrixCoefficient you need to create one for the boundary too
   auto *Sigma_fluid = new ConstantCoefficient(sigma_fluid);

   Vector sigma_vec_solid(3);
   sigma_vec_solid[0] = sigma_solid;                // Along fibers
   sigma_vec_solid[1] = sigma_solid/RF_ctx.aniso_ratio; // Sheet direction
   sigma_vec_solid[2] = sigma_solid/RF_ctx.aniso_ratio; // Sheet Normal to fibers
   auto *Sigma_solid = new MatrixFunctionCoefficient(3, ConductivityMatrix(sigma_vec_solid, EulerAngles(zmax, zmin)));

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
   heat::BCHandler *heat_bcs_cyl = new heat::BCHandler(cylinder_submesh, bc_verbose); // Boundary conditions handler for cylinder
   heat::BCHandler *heat_bcs_solid = new heat::BCHandler(solid_submesh, bc_verbose);  // Boundary conditions handler for solid
   heat::BCHandler *heat_bcs_fluid = new heat::BCHandler(fluid_submesh, bc_verbose);  // Boundary conditions handler for fluid

   // RF 
   electrostatics::BCHandler *rf_bcs_fluid = new electrostatics::BCHandler(fluid_submesh, bc_verbose); // Boundary conditions handler for fluid
   electrostatics::BCHandler *rf_bcs_solid = new electrostatics::BCHandler(solid_submesh, bc_verbose); // Boundary conditions handler for solid

   // Navier Stokes
   navier::BCHandler *bcs = new navier::BCHandler(fluid_submesh, bc_verbose); // Boundary conditions handler

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Create the Solvers
   ///////////////////////////////////////////////////////////////////////////////////////////////

   SolverParams sParams(1e-6, 1e-8, 1000, 0); // rtol, atol, maxiter, print-level

   // Solvers
   bool solv_verbose = false;

   navier::MonolithicNavierSolver Navier_Fluid(fluid_submesh, bcs, Navier_ctx.kinvis, Navier_ctx.uorder, Navier_ctx.porder, solv_verbose);
   Navier_Fluid.SetSolver(sParams);
   Navier_Fluid.SetMaxBDFOrder(Navier_ctx.bdf);

   ParGridFunction *velocity_fluid_gf = Navier_Fluid.GetVelocity();
   VectorGridFunctionCoefficient *wind_coeff = new VectorGridFunctionCoefficient(velocity_fluid_gf);

   heat::HeatSolver Heat_Cylinder(cylinder_submesh, Heat_ctx.order, heat_bcs_cyl, Kappa_cyl, c_cyl, rho_cyl, Heat_ctx.ode_solver_type, solv_verbose);    // Diffuson
   heat::HeatSolver Heat_Solid(solid_submesh, Heat_ctx.order, heat_bcs_solid, Kappa_solid, c_solid, rho_solid, Heat_ctx.ode_solver_type, true);  // Diffusion + Joule heating
   heat::HeatSolver Heat_Fluid(fluid_submesh, Heat_ctx.order, heat_bcs_fluid, Kappa_fluid, c_fluid, rho_fluid, 1.0, wind_coeff, 0.0, Heat_ctx.ode_solver_type, solv_verbose); // Advection + Diffusion

   electrostatics::ElectrostaticsSolver RF_Solid(solid_submesh, RF_ctx.order, rf_bcs_solid, Sigma_solid, solv_verbose);
   electrostatics::ElectrostaticsSolver RF_Fluid(fluid_submesh, RF_ctx.order, rf_bcs_fluid, Sigma_fluid, solv_verbose);

   // Grid functions in domain (inside solver)
   ParGridFunction *temperature_cylinder_gf = Heat_Cylinder.GetTemperatureGfPtr();
   ParGridFunction *temperature_solid_gf = Heat_Solid.GetTemperatureGfPtr();
   ParGridFunction *temperature_fluid_gf = Heat_Fluid.GetTemperatureGfPtr();

   ParGridFunction *phi_solid_gf = RF_Solid.GetPotentialGfPtr();
   ParGridFunction *phi_fluid_gf = RF_Fluid.GetPotentialGfPtr();


   // Cell Death solver (needs pointer to temperature grid function)
   celldeath::CellDeathSolver *CellDeath_Solid = nullptr;
   if (CellDeath_ctx.solver_type == 0)
      CellDeath_Solid = new celldeath::CellDeathSolverEigen( CellDeath_ctx.order, temperature_solid_gf, CellDeath_ctx.A1, CellDeath_ctx.A2, CellDeath_ctx.A3, CellDeath_ctx.deltaE1, CellDeath_ctx.deltaE2, CellDeath_ctx.deltaE3);
   else if (CellDeath_ctx.solver_type == 1)
      CellDeath_Solid = new celldeath::CellDeathSolverGotran( CellDeath_ctx.order, temperature_solid_gf, CellDeath_ctx.A1, CellDeath_ctx.A2, CellDeath_ctx.A3, CellDeath_ctx.deltaE1, CellDeath_ctx.deltaE2, CellDeath_ctx.deltaE3);
   else
      MFEM_ABORT("Invalid cell death solver type.");

   // Finite element spaces 
   ParFiniteElementSpace *heat_fes_cylinder = Heat_Cylinder.GetFESpace();
   ParFiniteElementSpace *heat_fes_solid = Heat_Solid.GetFESpace();
   ParFiniteElementSpace *heat_fes_fluid = Heat_Fluid.GetFESpace();
   ParFiniteElementSpace *heat_fes_grad_cylinder = Heat_Cylinder.GetVectorFESpace();
   ParFiniteElementSpace *heat_fes_grad_solid = Heat_Solid.GetVectorFESpace();
   ParFiniteElementSpace *heat_fes_grad_fluid = Heat_Fluid.GetVectorFESpace();

   ParFiniteElementSpace *rf_fes_solid = RF_Solid.GetFESpace();
   ParFiniteElementSpace *rf_fes_fluid = RF_Fluid.GetFESpace();
   ParFiniteElementSpace *rf_fes_grad_solid = new ParFiniteElementSpace(rf_fes_solid->GetParMesh(), rf_fes_solid->FEColl(), sdim);
   ParFiniteElementSpace *rf_fes_grad_fluid = new ParFiniteElementSpace(rf_fes_fluid->GetParMesh(), rf_fes_fluid->FEColl(), sdim);
   ParFiniteElementSpace *rf_fes_l2_solid = RF_Solid.GetL2FESpace();



   //// Grid functions for interface transfer --> need it for gridfunction coefficients
   // Heat-Transfer
   ParGridFunction *temperature_fc_fluid = new ParGridFunction(heat_fes_fluid); *temperature_fc_fluid = Heat_ctx.T_fluid;
   ParGridFunction *temperature_fc_cylinder = new ParGridFunction(heat_fes_cylinder); *temperature_fc_cylinder = Heat_ctx.T_cylinder;
   ParGridFunction *temperature_sc_solid = new ParGridFunction(heat_fes_solid); *temperature_sc_solid = Heat_ctx.T_solid;
   ParGridFunction *temperature_sc_cylinder = new ParGridFunction(heat_fes_cylinder); *temperature_sc_cylinder = Heat_ctx.T_cylinder;
   ParGridFunction *temperature_fs_fluid = new ParGridFunction(heat_fes_fluid); *temperature_fs_fluid = Heat_ctx.T_fluid;
   ParGridFunction *temperature_fs_solid = new ParGridFunction(heat_fes_solid); *temperature_fs_solid = Heat_ctx.T_solid;

   ParGridFunction *heatFlux_fs_fluid = new ParGridFunction(heat_fes_grad_fluid); *heatFlux_fs_fluid = 0.0;
   ParGridFunction *heatFlux_fs_solid = new ParGridFunction(heat_fes_grad_solid); *heatFlux_fs_solid = 0.0;
   ParGridFunction *heatFlux_fc_fluid = new ParGridFunction(heat_fes_grad_fluid); *heatFlux_fc_fluid = 0.0;
   ParGridFunction *heatFlux_fc_cylinder = new ParGridFunction(heat_fes_grad_cylinder); *heatFlux_fc_cylinder = 0.0;
   ParGridFunction *heatFlux_sc_solid = new ParGridFunction(heat_fes_grad_solid); *heatFlux_sc_solid = 0.0;
   ParGridFunction *heatFlux_sc_cylinder = new ParGridFunction(heat_fes_grad_cylinder); *heatFlux_sc_cylinder = 0.0;

   // RF
   ParGridFunction *phi_fs_fluid = new ParGridFunction(rf_fes_fluid); *phi_fs_fluid = 0.0;
   ParGridFunction *phi_fs_solid = new ParGridFunction(rf_fes_solid); *phi_fs_solid = 0.0;

   ParGridFunction *E_fs_solid = new ParGridFunction(rf_fes_grad_solid); *E_fs_solid = 0.0;
   ParGridFunction *E_fs_fluid = new ParGridFunction(rf_fes_grad_fluid); *E_fs_fluid = 0.0;



   // Grid functions and coefficients for error computation
   ParGridFunction *temperature_solid_prev_gf = new ParGridFunction(heat_fes_solid); *temperature_solid_prev_gf = Heat_ctx.T_solid;
   ParGridFunction *temperature_fluid_prev_gf = new ParGridFunction(heat_fes_fluid); *temperature_fluid_prev_gf = Heat_ctx.T_fluid;
   ParGridFunction *temperature_cylinder_prev_gf = new ParGridFunction(heat_fes_cylinder); *temperature_cylinder_prev_gf = Heat_ctx.T_cylinder;
   GridFunctionCoefficient temperature_solid_prev_coeff(temperature_solid_prev_gf);
   GridFunctionCoefficient temperature_fluid_prev_coeff(temperature_fluid_prev_gf);
   GridFunctionCoefficient temperature_cylinder_prev_coeff(temperature_cylinder_prev_gf);

   ParGridFunction *phi_solid_prev_gf = new ParGridFunction(rf_fes_solid);
   *phi_solid_prev_gf = 0.0;
   ParGridFunction *phi_fluid_prev_gf = new ParGridFunction(rf_fes_fluid);
   *phi_fluid_prev_gf = 0.0;
   GridFunctionCoefficient phi_solid_prev_coeff(phi_solid_prev_gf);
   GridFunctionCoefficient phi_fluid_prev_coeff(phi_fluid_prev_gf);

   // Auxiliary grid functions
   ParGridFunction *JouleHeating_gf = new ParGridFunction(rf_fes_l2_solid); *JouleHeating_gf = 0.0;
   Coefficient *JouleHeating_coeff = RF_Solid.GetJouleHeatingCoefficient();

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

  // Export fibers to disk
   if (Mpi::Root())
      mfem::out << "Exporting fibers to disk... \033[0m";

   ParFiniteElementSpace *fes_grad_solid = Heat_Solid.GetVectorFESpace();
   ParGridFunction *fiber_f_gf = new ParGridFunction(fes_grad_solid);
   ParGridFunction *fiber_t_gf = new ParGridFunction(fes_grad_solid);
   ParGridFunction *fiber_s_gf = new ParGridFunction(fes_grad_solid);
   ParGridFunction *euler_angles_gf = new ParGridFunction(fes_grad_solid);
   VectorFunctionCoefficient fiber_f_coeff(sdim, FiberDirection(EulerAngles(zmax, zmin), 0));
   VectorFunctionCoefficient fiber_t_coeff(sdim, FiberDirection(EulerAngles(zmax, zmin), 1));
   VectorFunctionCoefficient fiber_s_coeff(sdim, FiberDirection(EulerAngles(zmax, zmin), 2));
   VectorFunctionCoefficient euler_angles_coeff(sdim, EulerAngles(zmax, zmin));
   fiber_f_gf->ProjectCoefficient(fiber_f_coeff);
   fiber_t_gf->ProjectCoefficient(fiber_t_coeff);
   fiber_s_gf->ProjectCoefficient(fiber_s_coeff);
   euler_angles_gf->ProjectCoefficient(euler_angles_coeff);

   if (Sim_ctx.paraview)
   {
      ParaViewDataCollection* paraview_dc_fiber = new ParaViewDataCollection("Fiber", solid_submesh.get());
      paraview_dc_fiber->SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_fiber->SetDataFormat(VTKFormat::BINARY);
      paraview_dc_fiber->SetCompressionLevel(9);
      paraview_dc_fiber->RegisterField("Fiber", fiber_f_gf);
      paraview_dc_fiber->RegisterField("Sheet", fiber_t_gf);
      paraview_dc_fiber->RegisterField("Sheet-normal", fiber_s_gf);
      paraview_dc_fiber->RegisterField("Euler Angles", euler_angles_gf);
      if (Heat_ctx.order > 1)
      {
         paraview_dc_fiber->SetHighOrderOutput(true);
         paraview_dc_fiber->SetLevelsOfDetail(Heat_ctx.order);
      }
      paraview_dc_fiber->SetTime(0.0);
      paraview_dc_fiber->SetCycle(0);
      paraview_dc_fiber->Save();
      delete paraview_dc_fiber;
   }

   delete fiber_f_gf;
   delete fiber_t_gf;
   delete fiber_s_gf;

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Populate BC Handler
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Array<int> fluid_cylinder_interface;
   Array<int> fluid_solid_interface;
   Array<int> solid_cylinder_interface;

   Array<int> fluid_lateral_attr_tmp;   
   Array<int> fluid_lateral_attr;
   Array<int> fluid_lateral_1_attr;
   Array<int> fluid_lateral_2_attr;
   Array<int> fluid_lateral_3_attr;
   Array<int> fluid_lateral_4_attr;
   Array<int> fluid_top_attr;
   Array<int> solid_lateral_attr;
   Array<int> solid_bottom_attr;
   Array<int> cylinder_top_attr;

   Array<int> fluid_cylinder_interface_marker;
   Array<int> fluid_solid_interface_marker;
   Array<int> solid_cylinder_interface_marker;

   if (Mesh_ctx.hex)
   {
      // Extract boundary attributes
      fluid_cylinder_interface.SetSize(1);
      fluid_cylinder_interface = 7;
      fluid_solid_interface.SetSize(1);
      fluid_solid_interface = 3;
      solid_cylinder_interface.SetSize(1);
      solid_cylinder_interface = 4;

      fluid_lateral_1_attr.SetSize(1);
      fluid_lateral_1_attr = 8;
      fluid_lateral_2_attr.SetSize(1);
      fluid_lateral_2_attr = 9;
      fluid_lateral_3_attr.SetSize(1);
      fluid_lateral_3_attr = 10;
      fluid_lateral_4_attr.SetSize(1);
      fluid_lateral_4_attr = 11;

      fluid_top_attr.SetSize(1);
      fluid_top_attr = 5;
      solid_lateral_attr.SetSize(1);
      solid_lateral_attr = 2;
      solid_bottom_attr.SetSize(1);
      solid_bottom_attr = 1;
      cylinder_top_attr.SetSize(1);
      cylinder_top_attr = 6;

      // Extract boundary attributes markers on parent mesh (needed for GSLIB interpolation)
      fluid_cylinder_interface_marker = AttributeSets::AttrToMarker(parent_mesh.bdr_attributes.Max(), fluid_cylinder_interface);
      fluid_solid_interface_marker = AttributeSets::AttrToMarker(parent_mesh.bdr_attributes.Max(), fluid_solid_interface);
      solid_cylinder_interface_marker = AttributeSets::AttrToMarker(parent_mesh.bdr_attributes.Max(), solid_cylinder_interface);
   }
   else
   {
      // Extract boundary attributes
      fluid_cylinder_interface = bdr_attr_sets.GetAttributeSet("Cylinder-Fluid");
      fluid_solid_interface = bdr_attr_sets.GetAttributeSet("Solid-Fluid");
      solid_cylinder_interface = bdr_attr_sets.GetAttributeSet("Cylinder-Solid");

      fluid_lateral_1_attr = bdr_attr_sets.GetAttributeSet("Fluid Lateral 1");
      fluid_lateral_2_attr = bdr_attr_sets.GetAttributeSet("Fluid Lateral 2");
      fluid_lateral_3_attr = bdr_attr_sets.GetAttributeSet("Fluid Lateral 3");
      fluid_lateral_4_attr = bdr_attr_sets.GetAttributeSet("Fluid Lateral 4");

      fluid_top_attr = bdr_attr_sets.GetAttributeSet("Fluid Top");
      solid_lateral_attr = bdr_attr_sets.GetAttributeSet("Solid Lateral");
      solid_bottom_attr = bdr_attr_sets.GetAttributeSet("Solid Bottom");
      cylinder_top_attr = bdr_attr_sets.GetAttributeSet("Cylinder Top");

      // Extract boundary attributes markers on parent mesh (needed for GSLIB interpolation)
      fluid_cylinder_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Cylinder-Fluid");
      fluid_solid_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Solid-Fluid");
      solid_cylinder_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Cylinder-Solid");
   }

   fluid_lateral_attr_tmp.Append(fluid_lateral_1_attr);
   fluid_lateral_attr_tmp.Append(fluid_lateral_2_attr);
   fluid_lateral_attr_tmp.Append(fluid_lateral_3_attr);
   fluid_lateral_attr_tmp.Append(fluid_lateral_4_attr);
   fluid_lateral_attr = AttributeSets::AttrToMarker(fluid_submesh->bdr_attributes.Max(), fluid_lateral_attr_tmp);
   fluid_lateral_attr_tmp.DeleteAll();

   Array<int> solid_domain_attributes = AttributeSets::AttrToMarker(solid_submesh->attributes.Max(), solid_domain_attribute);
   Array<int> fluid_domain_attributes = AttributeSets::AttrToMarker(fluid_submesh->attributes.Max(), fluid_domain_attribute);
   Array<int> cylinder_domain_attributes = AttributeSets::AttrToMarker(cylinder_submesh->attributes.Max(), cylinder_domain_attribute);

   // NOTE: each submesh requires a different marker set as bdr attributes are generated per submesh (size can be different)
   // They can be converted as below using the attribute sets and the max attribute number for the specific submesh
   // If you don't want to convert and the attribute is just one number, you can add bcs or volumetric terms using the int directly (will take care of creating the marker array)
   // Array<int> fluid_cylinder_interface_c = AttributeSets::AttrToMarker(cylinder_submesh->bdr_attributes.Max(), fluid_cylinder_interface);
   // Array<int> fluid_cylinder_interface_f = AttributeSets::AttrToMarker(fluid_submesh->bdr_attributes.Max(), fluid_cylinder_interface);


   /////////////////////////////////////
   //          Navier Stokes         //
   /////////////////////////////////////

   // Fluid:
   // - No-slip   on  Γfs, Γfc, Γ fluid,top, Γ fluid,lateral,2-3
   // - Inflow    on  Γ fluid,lateral,1
   // - Outflow   on  Γ fluid,lateral,4
   // TODO: modify geometry to separate fluid lateral into 4 boundaries
   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for Navier Stokes ...\033[0m";
   
   bcs->AddVelDirichletBC(Navier_ctx.NoSlip, fluid_lateral_2_attr[0]);
   bcs->AddVelDirichletBC(Navier_ctx.NoSlip, fluid_lateral_4_attr[0]);
   bcs->AddVelDirichletBC(Navier_ctx.NoSlip, fluid_solid_interface[0]);
   bcs->AddVelDirichletBC(Navier_ctx.NoSlip, fluid_cylinder_interface[0]);
   bcs->AddVelDirichletBC(Navier_ctx.NoSlip, fluid_top_attr[0]);
   bcs->AddVelDirichletBC(inflow, fluid_lateral_1_attr[0]);
   // fluid_lateral_3 is outflow
   

   /////////////////////////////////////
   //           Heat Transfer         //
   /////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for Heat Transfer ...\033[0m";

   // Fluid:
   // - T = Heat_ctx.T_fluid on top/lateral walls
   // - Robin   on  Γfs
   // - Robin   on  Γfc

   if (Mpi::Root())
      mfem::out << "\033[0m\nSetting up BCs for fluid domain... \033[0m" << std::endl;

   ConstantCoefficient alpha_coeff_fs_fluid_heat(DD_ctx.alpha_heat[0]);
   ConstantCoefficient alpha_coeff_fc_fluid_heat(DD_ctx.alpha_heat[1]);
   GridFunctionCoefficient *temperature_fs_fluid_coeff = new GridFunctionCoefficient(temperature_fs_fluid);
   GridFunctionCoefficient *temperature_fc_fluid_coeff = new GridFunctionCoefficient(temperature_fc_fluid);
   VectorGridFunctionCoefficient *heatFlux_fs_fluid_coeff = new VectorGridFunctionCoefficient(heatFlux_fs_fluid);
   VectorGridFunctionCoefficient *heatFlux_fc_fluid_coeff = new VectorGridFunctionCoefficient(heatFlux_fc_fluid);

   heat_bcs_fluid->AddGeneralRobinBC(&alpha_coeff_fs_fluid_heat, &alpha_coeff_fs_fluid_heat, temperature_fs_fluid_coeff, heatFlux_fs_fluid_coeff, fluid_solid_interface[0], false); 
   heat_bcs_fluid->AddGeneralRobinBC(&alpha_coeff_fc_fluid_heat, &alpha_coeff_fc_fluid_heat, temperature_fc_fluid_coeff, heatFlux_fc_fluid_coeff, fluid_cylinder_interface[0], false);
   heat_bcs_fluid->AddDirichletBC(Heat_ctx.T_fluid, fluid_lateral_attr);
   heat_bcs_fluid->AddDirichletBC(Heat_ctx.T_fluid, fluid_top_attr[0]);

   // Solid:
   // - T = Heat_ctx.T_solid on bottom/lateral walls
   // - Robin   on  Γsc
   // - Robin   on  Γfs
   if (Mpi::Root())
      mfem::out << "\033[0m\nSetting up BCs for solid domain...\033[0m" << std::endl;

   ConstantCoefficient alpha_coeff_fs_solid_heat(DD_ctx.alpha_heat[2]);
   ConstantCoefficient alpha_coeff_sc_solid_heat(DD_ctx.alpha_heat[3]);
   GridFunctionCoefficient *temperature_fs_solid_coeff = new GridFunctionCoefficient(temperature_fs_solid);
   GridFunctionCoefficient *temperature_sc_solid_coeff = new GridFunctionCoefficient(temperature_sc_solid);
   VectorGridFunctionCoefficient *heatFlux_fs_solid_coeff = new VectorGridFunctionCoefficient(heatFlux_fs_solid);
   VectorGridFunctionCoefficient *heatFlux_sc_solid_coeff = new VectorGridFunctionCoefficient(heatFlux_sc_solid);
   heat_bcs_solid->AddGeneralRobinBC(&alpha_coeff_fs_solid_heat, &alpha_coeff_fs_solid_heat, temperature_fs_solid_coeff, heatFlux_fs_solid_coeff, fluid_solid_interface[0], false);
   heat_bcs_solid->AddGeneralRobinBC(&alpha_coeff_sc_solid_heat, &alpha_coeff_sc_solid_heat, temperature_sc_solid_coeff, heatFlux_sc_solid_coeff, solid_cylinder_interface[0], false);
   heat_bcs_solid->AddDirichletBC(Heat_ctx.T_solid, solid_lateral_attr[0]);
   heat_bcs_solid->AddDirichletBC(Heat_ctx.T_solid, solid_bottom_attr[0]);

   // Cylinder:
   // - T = Heat_ctx.T_cylinder on top wall
   // - Robin  on  Γsc
   // - Robin  on  Γfc

   if (Mpi::Root())
      mfem::out << "\033[0m\nSetting up BCs for cylinder domain...\033[0m" << std::endl;

   ConstantCoefficient alpha_coeff_fc_cylinder_heat(DD_ctx.alpha_heat[4]);
   ConstantCoefficient alpha_coeff_sc_cylinder_heat(DD_ctx.alpha_heat[5]);
   GridFunctionCoefficient *temperature_fc_cylinder_coeff = new GridFunctionCoefficient(temperature_fc_cylinder);
   GridFunctionCoefficient *temperature_sc_cylinder_coeff = new GridFunctionCoefficient(temperature_sc_cylinder);
   VectorGridFunctionCoefficient *heatFlux_fc_cylinder_coeff = new VectorGridFunctionCoefficient(heatFlux_fc_cylinder);
   VectorGridFunctionCoefficient *heatFlux_sc_cylinder_coeff = new VectorGridFunctionCoefficient(heatFlux_sc_cylinder);
   heat_bcs_cyl->AddGeneralRobinBC(&alpha_coeff_fc_cylinder_heat, &alpha_coeff_fc_cylinder_heat, temperature_sc_cylinder_coeff, heatFlux_fc_cylinder_coeff, solid_cylinder_interface[0], false);
   heat_bcs_cyl->AddGeneralRobinBC(&alpha_coeff_sc_cylinder_heat, &alpha_coeff_sc_cylinder_heat, temperature_fc_cylinder_coeff, heatFlux_fc_cylinder_coeff, fluid_cylinder_interface[0], false);
   heat_bcs_cyl->AddDirichletBC(Heat_ctx.T_cylinder, cylinder_top_attr[0]);

   Heat_Solid.AddVolumetricTerm(JouleHeating_coeff, solid_domain_attributes, false); // does not assume ownership of the coefficient


   /////////////////////////////////////
   //                RF               //
   /////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for RF ...\033[0m";

   // Fluid:
   // - Homogeneous Neumann on top/lateral walls
   // - Homogeneous Neumann on  Γfc
   // - Dirichlet   on  Γfs

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up RF BCs for fluid domain... \033[0m" << std::endl;

   ConstantCoefficient alpha_coeff_fs_fluid_rf(DD_ctx.alpha_rf[0]);
   GridFunctionCoefficient *phi_fs_fluid_coeff = new GridFunctionCoefficient(phi_fs_fluid);
   VectorGridFunctionCoefficient *E_fs_fluid_coeff = new VectorGridFunctionCoefficient(E_fs_fluid);
   rf_bcs_fluid->AddGeneralRobinBC(&alpha_coeff_fs_fluid_rf, &alpha_coeff_fs_fluid_rf, phi_fs_fluid_coeff, E_fs_fluid_coeff, fluid_solid_interface[0], false); 
   
   // Solid:
   // - Phi = 0 on bottom wall
   // - Dirichlet   on  Γsc
   // - Neumann   on  Γfs
   // - Homogeneous Neumann lateral wall
   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up RF BCs for solid domain...\033[0m" << std::endl;
   ConstantCoefficient alpha_coeff_fs_solid_rf(DD_ctx.alpha_rf[1]);
   GridFunctionCoefficient *phi_fs_solid_coeff = new GridFunctionCoefficient(phi_fs_solid);
   VectorGridFunctionCoefficient *E_fs_solid_coeff = new VectorGridFunctionCoefficient(E_fs_solid);
   rf_bcs_solid->AddDirichletBC(RF_ctx.phi_gnd, solid_bottom_attr[0]);
   rf_bcs_solid->AddDirichletBC(RF_ctx.phi_applied, solid_cylinder_interface[0]);
   rf_bcs_solid->AddGeneralRobinBC(&alpha_coeff_fs_solid_rf, &alpha_coeff_fs_solid_rf, phi_fs_solid_coeff, E_fs_solid_coeff, fluid_solid_interface[0], false); 

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Setup interface transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   /////////////////////////////////////
   //          Heat Transfer         //
   /////////////////////////////////////

   MPI_Barrier(parent_mesh.GetComm());

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up interface transfer for Heat Transfer... \033[0m" << std::endl;
      
   BidirectionalInterfaceTransfer finder_solid_to_cylinder_heat(heat_fes_solid, heat_fes_cylinder, solid_cylinder_interface_marker, TransferBackend::GSLIB);
   BidirectionalInterfaceTransfer finder_fluid_to_cylinder_heat(heat_fes_fluid, heat_fes_cylinder, fluid_cylinder_interface_marker, TransferBackend::GSLIB);
   BidirectionalInterfaceTransfer finder_fluid_to_solid_heat(heat_fes_fluid, heat_fes_solid, fluid_solid_interface_marker, TransferBackend::GSLIB);

   // Extract the indices of elements at the interface and convert them to markers
   // Useful to restrict the computation of the L2 error to the interface
   Array<int> tmp1, tmp2;
   finder_fluid_to_solid_heat.GetElementIdxDst(tmp1);
   finder_solid_to_cylinder_heat.GetElementIdxSrc(tmp2);
   Array<int> solid_interfaces_element_idx = tmp1 && tmp2;

   finder_fluid_to_solid_heat.GetElementIdxSrc(tmp1);
   finder_fluid_to_cylinder_heat.GetElementIdxSrc(tmp2);
   Array<int> fluid_interfaces_element_idx = tmp1 && tmp2;

   finder_fluid_to_cylinder_heat.GetElementIdxDst(tmp1);
   finder_solid_to_cylinder_heat.GetElementIdxDst(tmp2);
   Array<int> cylinder_interfaces_element_idx = tmp1 && tmp2;

   tmp1.DeleteAll();
   tmp2.DeleteAll();

  // Define QoI (heatflux) on the source meshes (cylinder, solid, fluid)
   int qoi_size_on_qp = sdim;

   // Define lamdbas to compute gradient of the temperature field
   auto heatFlux_cyl = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      DenseMatrix Kmat(sdim);
      Vector gradloc(sdim);

      Kappa_cyl->Eval(Kmat, Tr, ip);

      temperature_cylinder_gf->GetGradient(Tr, gradloc);
      Kmat.Mult(gradloc, qoi_loc);
   };

   auto heatFlux_solid = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      DenseMatrix Kmat(sdim);
      Vector gradloc(sdim);

      Kappa_solid->Eval(Kmat, Tr, ip);

      temperature_solid_gf->GetGradient(Tr, gradloc);
      Kmat.Mult(gradloc, qoi_loc);
   };

   auto heatFlux_fluid = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      DenseMatrix Kmat(sdim);
      Vector gradloc(sdim);

      Kappa_fluid->Eval(Kmat, Tr, ip);

      temperature_fluid_gf->GetGradient(Tr, gradloc);
      Kmat.Mult(gradloc, qoi_loc);
   };

   /////////////////////////////////////
   //               RF                //
   /////////////////////////////////////

   BidirectionalInterfaceTransfer finder_fluid_to_solid_rf(rf_fes_fluid, rf_fes_solid, fluid_solid_interface_marker, TransferBackend::GSLIB);

   // Define QoI (current density) on the source meshes (cylinder, solid, fluid)

   // Define lamdbas to compute the current density
   auto currentDensity_solid = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      DenseMatrix SigmaMat(sdim);
      Vector gradloc(sdim);

      Sigma_solid->Eval(SigmaMat, Tr, ip);

      phi_solid_gf->GetGradient(Tr, gradloc);
      SigmaMat.Mult(gradloc, qoi_loc);
   };

   auto currentDensity_fluid = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      Vector gradloc(sdim);

      real_t SigmaVal = Sigma_fluid->Eval(Tr, ip);
      phi_fluid_gf->GetGradient(Tr, qoi_loc);
      qoi_loc *= SigmaVal;
   };

      if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up solvers and assembling forms... \033[0m" << std::endl;


   if (Mpi::Root())
      mfem::out << "\033[0mHeat Transfer problem ... \033[0m";

   StopWatch chrono_assembly;
   chrono_assembly.Start();
   Heat_Solid.EnablePA(Heat_ctx.pa);
   Heat_Solid.Setup(Sim_ctx.dt);
   
   Heat_Cylinder.EnablePA(Heat_ctx.pa);
   Heat_Cylinder.Setup(Sim_ctx.dt);

   Heat_Fluid.EnablePA(Heat_ctx.pa);
   Heat_Fluid.Setup(Sim_ctx.dt);
   chrono_assembly.Stop();
   if (Mpi::Root())
      mfem::out << "\033[0mdone, in " << chrono_assembly.RealTime() << "s.\033[0m" << std::endl;


   if (Mpi::Root())
      mfem::out << "\033[0mRF problem... \033[0m";

   chrono_assembly.Clear();
   chrono_assembly.Start();
   RF_Solid.SetAssemblyLevel(RF_ctx.pa ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACY);
   RF_Solid.Setup();

   RF_Fluid.SetAssemblyLevel(RF_ctx.pa ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACY);
   RF_Fluid.Setup();
   chrono_assembly.Stop();

   if (Mpi::Root())
      mfem::out << "\033[0mdone, in " << chrono_assembly.RealTime() << "s.\033[0m" << std::endl;

   if (Mpi::Root())
      mfem::out << "\033[0mNavier Stokes problem... \033[0m";

   chrono_assembly.Clear();
   chrono_assembly.Start();
   Navier_Fluid.Setup(Sim_ctx.dt, Navier_ctx.pc_type, Navier_ctx.schur_pc_type, Navier_ctx.time_adaptivity_type, Navier_ctx.mass_lumping, Navier_ctx.stiff_strain);
   if (Navier_ctx.pc_type == navier::BlockPreconditionerType::YOSIDA_HIGH_ORDER_PRESSURE_CORRECTED)
   {
      Navier_Fluid.SetPressureCorrectionOrder(Navier_ctx.pressure_correction_order);
   }
   chrono_assembly.Stop();

   if (Mpi::Root())
      mfem::out << "\033[0mdone, in " << chrono_assembly.RealTime() << "s.\033[0m" << std::endl;

   // Setup ouput
   ParaViewDataCollection paraview_dc_cylinder_heat("Heat-Cylinder", cylinder_submesh.get());
   ParaViewDataCollection paraview_dc_solid_heat("Heat-Solid", solid_submesh.get());
   ParaViewDataCollection paraview_dc_fluid_heat("Heat-Fluid", fluid_submesh.get());
   ParaViewDataCollection paraview_dc_celldeath("CellDeath-Solid", solid_submesh.get());  
   ParaViewDataCollection paraview_dc_solid_rf("RF-Solid", solid_submesh.get());
   ParaViewDataCollection paraview_dc_fluid_rf("RF-Fluid", fluid_submesh.get());
   ParaViewDataCollection paraview_dc_navier("Navier-Stokes", fluid_submesh.get());
   if (Sim_ctx.paraview)
   {
      paraview_dc_cylinder_heat.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_cylinder_heat.SetDataFormat(VTKFormat::ASCII);
      paraview_dc_cylinder_heat.SetCompressionLevel(9);
      Heat_Cylinder.RegisterParaviewFields(paraview_dc_cylinder_heat);

      paraview_dc_solid_heat.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_solid_heat.SetDataFormat(VTKFormat::ASCII);
      paraview_dc_solid_heat.SetCompressionLevel(9);
      Heat_Solid.RegisterParaviewFields(paraview_dc_solid_heat);

      paraview_dc_fluid_heat.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_fluid_heat.SetDataFormat(VTKFormat::ASCII);
      paraview_dc_fluid_heat.SetCompressionLevel(9);
      Heat_Fluid.RegisterParaviewFields(paraview_dc_fluid_heat);

      paraview_dc_celldeath.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_celldeath.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_celldeath.SetCompressionLevel(9);
      CellDeath_Solid->RegisterParaviewFields(paraview_dc_celldeath);

      paraview_dc_solid_rf.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_solid_rf.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_solid_rf.SetCompressionLevel(9);
      RF_Solid.RegisterParaviewFields(paraview_dc_solid_rf);
      RF_Solid.AddParaviewField("Joule Heating", JouleHeating_gf);

      paraview_dc_fluid_rf.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_fluid_rf.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_fluid_rf.SetCompressionLevel(9);
      RF_Fluid.RegisterParaviewFields(paraview_dc_fluid_rf);

      paraview_dc_navier.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_navier.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_navier.SetCompressionLevel(9);
      Navier_Fluid.RegisterParaviewFields(paraview_dc_navier);
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
   ConstantCoefficient T0solid(Heat_ctx.T_solid);
   temperature_solid_gf->ProjectCoefficient(T0solid);
   Heat_Solid.SetInitialTemperature(*temperature_solid_gf);

   ConstantCoefficient T0cylinder(Heat_ctx.T_cylinder);
   temperature_cylinder_gf->ProjectCoefficient(T0cylinder);
   Heat_Cylinder.SetInitialTemperature(*temperature_cylinder_gf);

   ConstantCoefficient T0fluid(Heat_ctx.T_fluid);
   temperature_fluid_gf->ProjectCoefficient(T0fluid);
   Heat_Fluid.SetInitialTemperature(*temperature_fluid_gf);



   // Write fields to disk for VisIt
   if (Sim_ctx.paraview)
   {
      Heat_Solid.WriteFields(0, 0.0);
      Heat_Cylinder.WriteFields(0, 0.0);
      Heat_Fluid.WriteFields(0, 0.0);
      CellDeath_Solid->WriteFields(0, 0.0);
      RF_Solid.WriteFields(0, 0.0);
      RF_Fluid.WriteFields(0, 0.0);
   }

   // Vectors for error computation and relaxation
   Vector temperature_solid(heat_fes_solid->GetTrueVSize());   temperature_solid = Heat_ctx.T_solid;
   Vector temperature_cylinder(heat_fes_cylinder->GetTrueVSize()); temperature_cylinder = Heat_ctx.T_cylinder;
   Vector temperature_fluid(heat_fes_fluid->GetTrueVSize()); temperature_fluid = Heat_ctx.T_fluid;

   Vector temperature_solid_prev(heat_fes_solid->GetTrueVSize());
   temperature_solid_prev = Heat_ctx.T_solid;
   Vector temperature_cylinder_prev(heat_fes_cylinder->GetTrueVSize());
   temperature_cylinder_prev = Heat_ctx.T_cylinder;
   Vector temperature_fluid_prev(heat_fes_fluid->GetTrueVSize()); temperature_fluid_prev = Heat_ctx.T_fluid;

   Vector temperature_solid_tn(*temperature_solid_gf->GetTrueDofs()); temperature_solid_tn = Heat_ctx.T_solid;
   Vector temperature_cylinder_tn(*temperature_cylinder_gf->GetTrueDofs()); temperature_cylinder_tn = Heat_ctx.T_cylinder;
   Vector temperature_fluid_tn(*temperature_fluid_gf->GetTrueDofs()); temperature_fluid_tn = Heat_ctx.T_fluid;

   Vector phi_solid(rf_fes_solid->GetTrueVSize()); phi_solid = 0.0;
   Vector phi_fluid(rf_fes_fluid->GetTrueVSize()); phi_fluid = 0.0;
   Vector phi_solid_prev(rf_fes_solid->GetTrueVSize()); phi_solid_prev = 0.0;
   Vector phi_fluid_prev(rf_fes_fluid->GetTrueVSize()); phi_fluid_prev = 0.0;

   // Dofs
   int heat_cyl_dofs = Heat_Cylinder.GetProblemSize();
   int heat_fluid_dofs = Heat_Fluid.GetProblemSize();
   int heat_solid_dofs = Heat_Solid.GetProblemSize();
   int celldeath_solid_dofs = CellDeath_Solid->GetProblemSize();
   int rf_fluid_dofs = RF_Fluid.GetProblemSize();
   int rf_solid_dofs = RF_Solid.GetProblemSize();

   // Integration rule for the L2 error
   int order_quad_heat = std::max(2, 2*Heat_ctx.order + 2);
   const IntegrationRule *irs_heat[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs_heat[i] = &(IntRules.Get(i, order_quad_heat));
   }

   int order_quad_rf = std::max(2, 2*RF_ctx.order + 2);
   const IntegrationRule *irs_rf[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs_rf[i] = &(IntRules.Get(i, order_quad_rf));
   }

   if (Mpi::Root())
   {
      out << " Cylinder dofs (Heat): " << heat_cyl_dofs << std::endl;
      out << " Fluid dofs (Heat): " << heat_fluid_dofs << std::endl;
      out << " Solid dofs (Heat): " << heat_solid_dofs << std::endl;
      out << " Solid dofs (CellDeath): " << celldeath_solid_dofs << std::endl;
      out << " Fluid dofs (RF): " << rf_fluid_dofs << std::endl;
      out << " Solid dofs (RF): " << rf_solid_dofs << std::endl;
   }


   ///////////////////////////////////////////////////
   //            Preload of Navier Stokes
   ///////////////////////////////////////////////////


   if (preload_ns > 0)
   {
      if (Mpi::Root())
         mfem::out << "\033[31m\nPreloading Navier Stokes... \033[0m" << std::endl;
      
         real_t t = 0.0;
      real_t dt = Sim_ctx.dt;
      bool last_step = false;
      bool accept_step = false;
   
      real_t CFL = 0.0;
      
      Navier_Fluid.SetVerbose(true);

      for (int step = 1; !last_step; ++step)
      {
         if (t + dt >= preload_ns - dt / 2)
         {
            last_step = true;
         }
   
         accept_step = Navier_Fluid.Step(t, dt, step);
      }

      Navier_Fluid.SetVerbose(solv_verbose);
   }

   
   ///////////////////////////////////////////////////
   // Solve RF before time integration
   ///////////////////////////////////////////////////

   bool print_subiter = true;
   if (Mpi::Root())
      mfem::out << "\033[31m\nSolving RF... \033[0m" << std::endl;

   {
      bool converged = false;
      int iter = 0;
      int iter_solid = 0;
      int iter_fluid = 0;
      real_t norm_diff = 2 * TOL;
      real_t norm_diff_solid = 2 * TOL;
      real_t norm_diff_fluid = 2 * TOL;

      bool converged_solid = false;
      bool converged_fluid = false;

      // Timing
      StopWatch chrono, chrono_total;
      real_t t_transfer, t_interp, t_solve_fluid, t_solve_solid, t_relax_fluid, t_relax_solid, t_error, t_error_bdry, t_paraview, t_joule;

      bool assembleRHS = true;
      Array2D<real_t> convergence_rf(MAX_ITER, 3); convergence_rf = 0.0;

      while (!converged && iter <= MAX_ITER)
      {

         chrono_total.Clear();
         chrono_total.Start();

         // Store the previous temperature on domains for convergence
         phi_solid_gf->GetTrueDofs(phi_solid_prev);
         phi_fluid_gf->GetTrueDofs(phi_fluid_prev);
         phi_solid_prev_gf->SetFromTrueDofs(phi_solid_prev);
         phi_fluid_prev_gf->SetFromTrueDofs(phi_fluid_prev);

         ///////////////////////////////////////////////////
         //         Fluid Domain (F), Dirichlet(S)        //
         ///////////////////////////////////////////////////

         MPI_Barrier(parent_mesh.GetComm());

         chrono.Clear();
         chrono.Start();
         //if (!converged_fluid)
         { // S->F: Φ, J
            finder_fluid_to_solid_rf.InterpolateBackward(*phi_solid_gf, *phi_fs_fluid);
            finder_fluid_to_solid_rf.InterpolateQoIBackward(currentDensity_solid, *E_fs_fluid);
         }
         chrono.Stop();
         t_transfer = chrono.RealTime();

         chrono.Clear();
         chrono.Start();
         RF_Fluid.Solve(assembleRHS);
         chrono.Stop();
         t_solve_fluid = chrono.RealTime();

         chrono.Clear();
         chrono.Start();
         if (iter > 0)
         {
            phi_fluid_gf->GetTrueDofs(phi_fluid);
            phi_fluid *= DD_ctx.omega_rf_fluid;
            phi_fluid.Add(1.0 - DD_ctx.omega_rf_fluid, phi_fluid_prev);
            phi_fluid_gf->SetFromTrueDofs(phi_fluid);
         }
         chrono.Stop();
         t_relax_fluid = chrono.RealTime();

         /////////////////////////////////////////////////////////////////
         //          Solid Domain (S), Neumann(F)-Dirichlet(C)          //
         /////////////////////////////////////////////////////////////////

         MPI_Barrier(parent_mesh.GetComm());

         chrono.Clear();
         chrono.Start();
         // if (!converged_solid)
         { // F->S: Φ, J
            finder_fluid_to_solid_rf.InterpolateForward(*phi_fluid_gf, *phi_fs_solid);
            finder_fluid_to_solid_rf.InterpolateQoIForward(currentDensity_fluid, *E_fs_solid);
         }
         chrono.Stop();
         t_interp = chrono.RealTime();

         chrono.Clear();
         chrono.Start();
         RF_Solid.Solve(assembleRHS);
         chrono.Stop();
         t_solve_solid = chrono.RealTime();

         chrono.Clear();
         chrono.Start();
         if (iter > 0)
         {
            phi_solid_gf->GetTrueDofs(phi_solid);
            phi_solid *= DD_ctx.omega_rf_solid;
            phi_solid.Add(1.0 - DD_ctx.omega_rf_solid, phi_solid_prev);
            phi_solid_gf->SetFromTrueDofs(phi_solid);
         }
         chrono.Stop();
         t_relax_solid = chrono.RealTime();

         //////////////////////////////////////////////////////////////////////
         //                        Check convergence                         //
         //////////////////////////////////////////////////////////////////////

         chrono.Clear();
         chrono.Start();
         // Compute global norms directly
         real_t global_norm_diff_solid = phi_solid_gf->ComputeL2Error(phi_solid_prev_coeff, irs_rf, &solid_interfaces_element_idx);
         real_t global_norm_diff_fluid = phi_fluid_gf->ComputeL2Error(phi_fluid_prev_coeff, irs_rf, &fluid_interfaces_element_idx);
         chrono.Stop();
         t_error_bdry = chrono.RealTime();

         // Check convergence on domains
         converged_solid = (global_norm_diff_solid < TOL); //   &&(iter > 0);
         converged_fluid = (global_norm_diff_fluid < TOL); //   &&(iter > 0);

         // Check convergence
         converged = converged_solid && converged_fluid;

         iter++;

         if (Mpi::Root() && Sim_ctx.save_convergence)
         {
            convergence_rf(iter, 0) = iter;
            convergence_rf(iter, 1) = global_norm_diff_fluid;
            convergence_rf(iter, 2) = global_norm_diff_solid;
         }

         if (Mpi::Root() && print_subiter)
         {
            out << std::left << std::setw(16) << iter
                << std::scientific << std::setw(16) << global_norm_diff_fluid
                << std::setw(16) << global_norm_diff_solid
                << std::endl;
         }

         chrono_total.Stop();

         if (Mpi::Root() && Sim_ctx.print_timing)
         { // Print times
            out << "------------------------------------------------------------" << std::endl;
            out << "Transfer: " << t_transfer << " s" << std::endl;
            out << "Interpolation: " << t_interp << " s" << std::endl;
            out << "Fluid Solve: " << t_solve_fluid << " s" << std::endl;
            out << "Solid Solve: " << t_solve_solid << " s" << std::endl;
            out << "Relaxation Fluid: " << t_relax_fluid << " s" << std::endl;
            out << "Relaxation Solid: " << t_relax_solid << " s" << std::endl;
            // out << "Error: " << t_error << " s" << std::endl;
            out << "Error Boundary: " << t_error_bdry << " s" << std::endl;
            out << "Total: " << chrono_total.RealTime() << " s" << std::endl;
            out << "------------------------------------------------------------" << std::endl;
         }
      } // END OF CONVERGENCE LOOP

      if (Mpi::Root() && Sim_ctx.save_convergence)
      {
         std::string name_rf = "RF-pre";
         saveConvergenceArray(convergence_rf, Sim_ctx.outfolder, name_rf, 0);

         Array<int> subiter_count_rf; subiter_count_rf.Append(iter);
         saveSubiterationCount(subiter_count_rf, Sim_ctx.outfolder, name_rf);

         convergence_rf.DeleteAll();
         subiter_count_rf.DeleteAll();
      }

      // Compute Joule heating
      chrono.Clear(); chrono.Start();
      // Output of time steps
      RF_Solid.GetJouleHeating(*JouleHeating_gf);
      chrono.Stop();
      t_joule = chrono.RealTime();

      // Export converged fields
      chrono.Clear();
      chrono.Start();
      if (Sim_ctx.paraview)
      {
         RF_Solid.WriteFields(0, 0.0);
         RF_Fluid.WriteFields(0, 0.0);
      }
      chrono.Stop();
      t_paraview = chrono.RealTime();

      if (Mpi::Root() && Sim_ctx.print_timing)
      { // Print times
         out << "------------------------------------------------------------" << std::endl;
         out << "Joule: " << t_joule << " s" << std::endl;
         out << "Paraview: " << t_paraview << " s" << std::endl;
         out << "------------------------------------------------------------" << std::endl;
      }
   }

   if (Mpi::Root())
      mfem::out << "\033[31m\ndone.\033[0m" << std::endl;


   ///////////////////////////////////////////////////
   // Outer loop for time integration
   ///////////////////////////////////////////////////
   
   // Reset the Navier-Stokes time adaptivity
   Navier_Fluid.SetTimeAdaptivityType(navier::TimeAdaptivityType::NONE);

   // Timing
   StopWatch chrono_total;

   if (Mpi::Root())
   {
      out << "-------------------------------------------------------------------------------------------------"
          << std::endl;
      out << std::left << std::setw(16) << "Step" << std::setw(16) << "Time" << std::setw(16) << "Sim_ctx.dt" << std::setw(16) << "Sub-iterations (F-S-C)" << std::endl;
      out << "-------------------------------------------------------------------------------------------------"
          << std::endl;
   }


   real_t t = 0.0;
   bool last_step = false;
   bool converged = false;
   int num_steps = (int)(Sim_ctx.t_final / Sim_ctx.dt);
   real_t global_norm_diff_solid;
   real_t global_norm_diff_fluid;
   real_t global_norm_diff_cylinder;
   
   Array<int> subiter_count_heat;

   // Timing
   StopWatch chrono, chrono_total_subiter;
   real_t t_total_subiter, t_transfer_fluid, t_transfer_solid, t_transfer_cylinder, t_solve_navier, t_solve_fluid, t_solve_solid, t_solve_cylinder, t_solve_celldeath, t_relax_fluid, t_relax_solid, t_relax_cylinder, t_error_bdry, t_paraview;

   for (int step = 1; !last_step; step++)
   {
      if (Mpi::Root())
      {
         mfem::out << std::left << std::setw(16) << step << std::setw(16) << t << std::setw(16) << Sim_ctx.dt << std::setw(16) << std::endl;
      }

      if (t + Sim_ctx.dt >= Sim_ctx.t_final - Sim_ctx.dt / 2)
      {
         last_step = true;
      }

      /////////////////////////////////////////
      //              Solve RF               //
      /////////////////////////////////////////
      {
      }

      /////////////////////////////////////////
      //          Solver Navier-Stokes       //
      /////////////////////////////////////////
      {
         chrono.Clear();
         chrono.Start();

         if (Mpi::Root())
            mfem::out << "\033[31mSolving Navier-Stokes problem on fluid ... \033[0m";

         bool accept_step = Navier_Fluid.Step(t, Sim_ctx.dt, step);
         t -= Sim_ctx.dt; // Reset t to same time step, since t is incremented in the Step function

         if (Mpi::Root())
                  mfem::out << "\033[31mdone.\033[0m" << std::endl;

         t_solve_navier = chrono.RealTime();

         if( Sim_ctx.paraview && accept_step)
         {
            Navier_Fluid.WriteFields(step, t);
         }

         wind_coeff->SetGridFunction(velocity_fluid_gf);  // NOTE: do we need this since wind_coeff gets a pointer to velocity_fluid_gf?
      }

      /////////////////////////////////////////
      //         Solve HEAT TRANSFER         //
      /////////////////////////////////////////
      Array2D<real_t> convergence_heat(MAX_ITER, 3); convergence_heat = 0.0;
      {
         temperature_solid_tn = *temperature_solid_gf->GetTrueDofs();
         temperature_cylinder_tn = *temperature_cylinder_gf->GetTrueDofs();
         temperature_fluid_tn = *temperature_fluid_gf->GetTrueDofs();

         // Inner loop for the segregated solve
         int iter = 0;
         int iter_solid = 0;
         int iter_fluid = 0;
         int iter_cylinder = 0;
         real_t norm_diff = 2 * TOL_HEAT;
         real_t norm_diff_solid = 2 * TOL_HEAT;
         real_t norm_diff_fluid = 2 * TOL_HEAT;
         real_t norm_diff_cylinder = 2 * TOL_HEAT;

         bool converged_solid = false;
         bool converged_fluid = false;
         bool converged_cylinder = false;

         chrono_total.Clear();
         chrono_total.Start();

         while (!converged && iter <= MAX_ITER)
         {
            chrono_total_subiter.Clear();
            chrono_total_subiter.Start();

            // Store the previous temperature on domains for convergence
            temperature_solid_gf->GetTrueDofs(temperature_solid_prev);
            temperature_fluid_gf->GetTrueDofs(temperature_fluid_prev);
            temperature_cylinder_gf->GetTrueDofs(temperature_cylinder_prev);
            temperature_solid_prev_gf->SetFromTrueDofs(temperature_solid_prev);
            temperature_fluid_prev_gf->SetFromTrueDofs(temperature_fluid_prev);
            temperature_cylinder_prev_gf->SetFromTrueDofs(temperature_cylinder_prev);

            /////////////////////////////////////////////////////////////////
            //         Fluid Domain (F), Robin(S)-Robin(C)         //
            /////////////////////////////////////////////////////////////////

            MPI_Barrier(parent_mesh.GetComm());

            // if (!converged_fluid)
            { // S->F: Transfer T, C->F: Transfer T
               chrono.Clear();
               chrono.Start();
               finder_fluid_to_cylinder_heat.InterpolateBackward(*temperature_cylinder_gf, *temperature_fc_fluid);
               finder_fluid_to_cylinder_heat.InterpolateQoIBackward(heatFlux_cyl, *heatFlux_fc_fluid);
               finder_fluid_to_solid_heat.InterpolateBackward(*temperature_solid_gf, *temperature_fs_fluid);
               finder_fluid_to_solid_heat.InterpolateQoIBackward(heatFlux_solid, *heatFlux_fs_fluid);
               chrono.Stop();
               t_transfer_fluid = chrono.RealTime();

               // Step in the fluid domain
               chrono.Clear();
               chrono.Start();
               temperature_fluid_gf->SetFromTrueDofs(temperature_fluid_tn);
               Heat_Fluid.Step(t, Sim_ctx.dt, step, false);
               temperature_fluid_gf->GetTrueDofs(temperature_fluid);
               t -= Sim_ctx.dt; // Reset t to same time step, since t is incremented in the Step function
               chrono.Stop();
               t_solve_fluid = chrono.RealTime();

               // Relaxation
               // Heat_ctx.T_fluid(j+1) = ω * Heat_ctx.T_fluid,j+1 + (1 - ω) * Heat_ctx.T_fluid,j
               chrono.Clear();
               chrono.Start();
               if (iter > 0)
               {
                  temperature_fluid *= DD_ctx.omega_heat_fluid;
                  temperature_fluid.Add(1 - DD_ctx.omega_heat_fluid, temperature_fluid_prev);
                  temperature_fluid_gf->SetFromTrueDofs(temperature_fluid);
               }
               chrono.Stop();
               t_relax_fluid = chrono.RealTime();

               iter_fluid++;
            }

            /////////////////////////////////////////////////////////////////
            //          Solid Domain (S), Robin(F)-Robin(C)            //
            /////////////////////////////////////////////////////////////////

            MPI_Barrier(parent_mesh.GetComm());

            // if (!converged_solid)
            { // F->S: Transfer k ∇T_wall, C->S: Transfer k ∇T_wall
               chrono.Clear();
               chrono.Start();
               finder_fluid_to_solid_heat.InterpolateForward(*temperature_fluid_gf, *temperature_fs_solid);
               finder_fluid_to_solid_heat.InterpolateQoIForward(heatFlux_fluid, *heatFlux_fs_solid);
               finder_solid_to_cylinder_heat.InterpolateBackward(*temperature_cylinder_gf, *temperature_sc_solid);
               finder_solid_to_cylinder_heat.InterpolateQoIBackward(heatFlux_cyl, *heatFlux_sc_solid);
               chrono.Stop();
               t_transfer_solid = chrono.RealTime();

               // Step in the solid domain
               chrono.Clear();
               chrono.Start();
               temperature_solid_gf->SetFromTrueDofs(temperature_solid_tn);
               Heat_Solid.Step(t, Sim_ctx.dt, step, false);
               temperature_solid_gf->GetTrueDofs(temperature_solid);
               t -= Sim_ctx.dt; // Reset t to same time step, since t is incremented in the Step function
               chrono.Stop();
               t_solve_solid = chrono.RealTime();

               // Relaxation
               // T_wall(j+1) = ω * Heat_ctx.T_solid,j+1 + (1 - ω) * Heat_ctx.T_solid,j
               chrono.Clear();
               chrono.Start();
               if (iter > 0)
               {
                  temperature_solid *= DD_ctx.omega_heat_solid;
                  temperature_solid.Add(1 - DD_ctx.omega_heat_solid, temperature_solid_prev);
                  temperature_solid_gf->SetFromTrueDofs(temperature_solid);
               }
               chrono.Stop();
               t_relax_solid = chrono.RealTime();

               iter_solid++;
            }

            /////////////////////////////////////////////////////////////////
            //          Cylinder Domain (F), Neumann(F)-Dirichlet(S)       //
            /////////////////////////////////////////////////////////////////

            MPI_Barrier(parent_mesh.GetComm());

            // if (!converged_cylinder)
            { // F->C: Transfer k ∇T_wall, S->C: Transfer T
               chrono.Clear();
               chrono.Start();
               finder_fluid_to_cylinder_heat.InterpolateForward(*temperature_fluid_gf, *temperature_fc_cylinder); 
               finder_fluid_to_cylinder_heat.InterpolateQoIForward(heatFlux_fluid, *heatFlux_fc_cylinder);
               finder_solid_to_cylinder_heat.InterpolateForward(*temperature_solid_gf, *temperature_sc_cylinder);
               finder_solid_to_cylinder_heat.InterpolateQoIForward(heatFlux_solid, *heatFlux_sc_cylinder);
               chrono.Stop();
               t_transfer_cylinder = chrono.RealTime();

               // Step in the cylinder domain
               chrono.Clear();
               chrono.Start();
               temperature_cylinder_gf->SetFromTrueDofs(temperature_cylinder_tn);
               Heat_Cylinder.Step(t, Sim_ctx.dt, step, false);
               temperature_cylinder_gf->GetTrueDofs(temperature_cylinder);
               t -= Sim_ctx.dt; // Reset t to same time step, since t is incremented in the Step function
               chrono.Stop();
               t_solve_cylinder = chrono.RealTime();

               // Relaxation
               // Heat_ctx.T_cylinder(j+1) = ω * Heat_ctx.T_cylinder,j+1 + (1 - ω) * Heat_ctx.T_cylinder,j
               chrono.Clear();
               chrono.Start();
               if (iter > 0)
               {
                  temperature_cylinder *= DD_ctx.omega_heat_cyl;
                  temperature_cylinder.Add(1 - DD_ctx.omega_heat_cyl, temperature_cylinder_prev);
                  temperature_cylinder_gf->SetFromTrueDofs(temperature_cylinder);
               }
               chrono.Stop();
               t_relax_cylinder = chrono.RealTime();

               iter_cylinder++;
            }

            //////////////////////////////////////////////////////////////////////
            //                        Check convergence                         //
            //////////////////////////////////////////////////////////////////////

            // Compute local norms
            chrono.Clear();
            chrono.Start();
            global_norm_diff_solid = temperature_solid_gf->ComputeL2Error(temperature_solid_prev_coeff, irs_heat, &solid_interfaces_element_idx);
            global_norm_diff_fluid = temperature_fluid_gf->ComputeL2Error(temperature_fluid_prev_coeff, irs_heat, &fluid_interfaces_element_idx);
            global_norm_diff_cylinder = temperature_cylinder_gf->ComputeL2Error(temperature_cylinder_prev_coeff, irs_heat, &cylinder_interfaces_element_idx);
            chrono.Stop();
            t_error_bdry = chrono.RealTime();

            if (Mpi::Root() && Sim_ctx.save_convergence)
            {
               convergence_heat(iter, 0) = iter+1;
               convergence_heat(iter, 1) = global_norm_diff_fluid;
               convergence_heat(iter, 2) = global_norm_diff_solid;
            }

            // Check convergence
            converged_solid = global_norm_diff_solid < TOL_HEAT;
            converged_fluid = global_norm_diff_fluid < TOL_HEAT;
            converged_cylinder = global_norm_diff_cylinder < TOL_HEAT;
            converged = converged_solid && converged_fluid && converged_cylinder;
            
            iter++;

            chrono_total_subiter.Stop();
            t_total_subiter = chrono_total_subiter.RealTime();
         } // END OF CONVERGENCE LOOP

         if (iter > MAX_ITER)
         {
            if (Mpi::Root())
               mfem::out << "Warning: Maximum number of iterations reached. Errors: "
                         << std::scientific << std::setw(16) << " Fluid: " << global_norm_diff_fluid
                         << std::setw(16) << " Solid: " << global_norm_diff_solid
                         << std::setw(16) << " Cylinder: " << global_norm_diff_cylinder
                         << std::endl;
            break;
         }

         if (Mpi::Root() && Sim_ctx.save_convergence)
         { // Save convergence data            
            subiter_count_heat.Append(iter);
            std::string name_heat = "Heat";
            saveConvergenceArray(convergence_heat, Sim_ctx.outfolder, name_heat, step);
            convergence_heat.DeleteAll();
         }

         // Reset the convergence flag and time for the next iteration
         if (Mpi::Root())
         { 
            // Print the message and iterations on the same line
            mfem::out << "\033[34mSolving HeatTransfer problem... "
                     << std::setw(20) << " " // Adjust the width to align with "Sub-iterations (F-S-C)"
                     << std::setw(2) << iter_fluid << std::setw(2) << "-"
                     << std::setw(2) << iter_solid << std::setw(2) << "-"
                     << std::setw(2) << iter_cylinder << "\033[0m" << std::endl;
         }

         if (Mpi::Root() && Sim_ctx.print_timing)
         {
            out << "------------------------------------------------------------" << std::endl;
            out << "Transfer (Fluid): " << t_transfer_fluid << " s" << std::endl;
            out << "Transfer (Solid): " << t_transfer_solid << " s" << std::endl;
            out << "Transfer (Cylinder): " << t_transfer_cylinder << " s" << std::endl;
            out << "Solve Heat (Fluid): " << t_solve_fluid << " s" << std::endl;
            out << "Solve Heat (Solid): " << t_solve_solid << " s" << std::endl;
            out << "Solve Heat (Cylinder): " << t_solve_cylinder << " s" << std::endl;
            out << "Relax (Fluid): " << t_relax_fluid << " s" << std::endl;
            out << "Relax (Solid): " << t_relax_solid << " s" << std::endl;
            out << "Relax (Cylinder): " << t_relax_cylinder << " s" << std::endl;
            out << "Error (Boundary): " << t_error_bdry << " s" << std::endl;
            out << "Total (Subiter): " << t_total_subiter << " s" << std::endl;
            out << "------------------------------------------------------------" << std::endl;
         }
      }

      /////////////////////////////////////////
      //           Solve CELL DEATH          //
      /////////////////////////////////////////

      {
         chrono.Clear();
         chrono.Start();
         if (Mpi::Root())
            mfem::out << "\033[32mSolving CellDeath problem on solid ... \033[0m";

         CellDeath_Solid->Solve(t, Sim_ctx.dt);

         if (Mpi::Root())
            mfem::out << "\033[32mdone.\033[0m" << std::endl;
         chrono.Stop();
         t_solve_celldeath = chrono.RealTime();
      }


      ///////////////////////////////////////////////
      //         Update for next time step         //
      ///////////////////////////////////////////////

      // Update time step history
      Heat_Solid.UpdateTimeStepHistory();
      Heat_Cylinder.UpdateTimeStepHistory();
      Heat_Fluid.UpdateTimeStepHistory();

      temperature_solid_tn = *temperature_solid_gf->GetTrueDofs();
      temperature_cylinder_tn = *temperature_cylinder_gf->GetTrueDofs();
      temperature_fluid_tn = *temperature_fluid_gf->GetTrueDofs();

      t += Sim_ctx.dt;
      converged = false;

      // Output of time steps
      chrono.Clear(); chrono.Start();
      if (Sim_ctx.paraview && (step % Sim_ctx.save_freq == 0))
      {
         Heat_Solid.WriteFields(step, t);
         Heat_Cylinder.WriteFields(step, t);
         Heat_Fluid.WriteFields(step, t);
         CellDeath_Solid->WriteFields(step, t);
      }
      chrono.Stop();
      t_paraview = chrono.RealTime();

      chrono_total.Stop();

      if (Mpi::Root() && Sim_ctx.print_timing )
      {
         out << "------------------------------------------------------------" << std::endl;
         out << "Solve Navier Stokes: " << t_solve_navier << " s" << std::endl;
         out << "Solve CellDeath (Solid): " << t_solve_celldeath << " s" << std::endl;
         out << "Paraview: " << t_paraview << " s" << std::endl;
         out << "Total: " << chrono_total.RealTime() << " s" << std::endl;
         out << "------------------------------------------------------------" << std::endl;
      }
   } // END OF TIME INTEGRATION

   // Save convergence data
   if (Mpi::Root() && Sim_ctx.save_convergence)
   {
      std::string name_heat = "Heat";
      saveSubiterationCount(subiter_count_heat, Sim_ctx.outfolder, name_heat);
   }
   
   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // No need to delete:
   // - BcHandler objects: owned by HeatOperator
   // - Coefficients provide to BCHandler: owned by CoeffContainers
   delete Id;

   delete CellDeath_Solid;

   delete rho_fluid;
   delete rho_solid;
   delete rho_cyl;

   delete Kappa_cyl;
   delete Kappa_fluid;
   delete Kappa_solid;

   delete c_cyl;
   delete c_fluid;
   delete c_solid;

   delete Sigma_fluid;
   delete Sigma_solid;

   delete temperature_fc_fluid;
   delete temperature_sc_cylinder;
   delete temperature_fs_fluid;

   delete temperature_fc_cylinder_coeff;
   delete temperature_sc_cylinder_coeff;
   delete temperature_fs_fluid_coeff;
   delete temperature_fc_fluid_coeff;

   delete heatFlux_fs_solid;
   delete heatFlux_fc_cylinder;
   delete heatFlux_sc_solid;

   delete temperature_solid_prev_gf;
   delete temperature_fluid_prev_gf;
   delete temperature_cylinder_prev_gf;

   delete rf_fes_grad_solid;
   delete rf_fes_grad_fluid;

   delete E_fs_solid;
   delete E_fs_fluid;
   delete phi_fs_fluid;
   delete phi_solid_prev_gf;
   delete phi_fluid_prev_gf;
   delete JouleHeating_gf;

   delete euler_angles_gf;
   
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

void saveConvergenceArray(const Array2D<real_t> &data, const std::string &outfolder, const std::string &name, int step)
{
   // Create the output folder path + name
   std::string outputFolder = outfolder + "/convergence/" + name;

   if (!fs::is_directory(outputFolder.c_str()) || !fs::exists(outputFolder.c_str())) { // Check if folder exists
      fs::create_directories(outputFolder); // create folder
   }

   // Construct the filename
   std::ostringstream filename;
   filename << outputFolder << "/step_" << step << ".txt";

   // Open the file and save the data
   std::ofstream outFile(filename.str());
   if (outFile.is_open())
   {
      data.Save(outFile);
      outFile.close();
   }
   else
   {
      std::cerr << "Unable to open file: " << filename.str() << std::endl;
   }
}

void saveSubiterationCount(const Array<int> &data, const std::string &outfolder, const std::string &name)
{
   // Create the output folder path + name
   std::string outputFolder = outfolder + "/convergence/" + name;

   if (!fs::is_directory(outputFolder.c_str()) || !fs::exists(outputFolder.c_str())) { // Check if folder exists
      fs::create_directories(outputFolder); // create folder
   }

   // Construct the filename
   std::ostringstream filename;
   filename << outputFolder << "/Subiter.txt";

   // Open the file and save the data
   std::ofstream outFile(filename.str());
   if (outFile.is_open())
   {
      data.Save(outFile);
      outFile.close();
   }
   else
   {
      std::cerr << "Unable to open file: " << filename.str() << std::endl;
   }
}



std::function<void(const Vector &, Vector &)> EulerAngles(real_t zmin, real_t zmax)
{
   return [zmin, zmax](const Vector &x, Vector &e)
   {
      const int dim = x.Size();

      // Compute the linear interpolation factor
      real_t t = (x(2) - zmin) / (zmax - zmin);

      // Compute the angle in degrees
      real_t angle = 60.0 * (2.0 * t - 1.0);

      // Convert the angle to radians
      real_t angle_rad = angle * M_PI / 180.0;

      // Set the Euler angles (assuming rotation around the z-axis)
      e.SetSize(3);
      e(0) = 0.0;          // Roll
      e(1) = 0.0;          // Pitch
      e(2) = angle_rad;    // Yaw
   };
}