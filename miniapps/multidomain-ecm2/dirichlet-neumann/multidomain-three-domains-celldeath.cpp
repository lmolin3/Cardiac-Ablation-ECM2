// Solve HeatTransfer problem with three domains (solid, fluid, cylinder).
//                 rho c dT/Sim_ctx.dt = ∇•κ∇T         in  fluid, solid
//                 rho c dT/Sim_ctx.dt = ∇•κ∇T + Q     in  cylinder
// The problem is solved using a segregated approach with two-way coupling (Neumann-Dirichlet)
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
//    mpirun -np 4 ./multidomain-three-domains-celldeath -o 2 -dt 0.01 -tf 0.05 -kc 1 -kb 1 --relaxation-parameter 0.8  
// 2. Hexahedral mesh
//    mpirun -np 4 ./multidomain-three-domains-celldeath -hex -o 2 -dt 0.01 -tf 0.05 -kc 1 -kb 1 --relaxation-parameter 0.8  
// 3. Hexahedral mesh with partial assembly
//    mpirun -np 4 ./multidomain-three-domains-celldeath -hex -pa -o 2 -dt 0.01 -tf 0.05 -kc 1 -kb 1 --relaxation-parameter 0.8 

// MFEM library
#include "mfem.hpp"

// Multiphysics modules
#include "lib/heat_solver.hpp"
#include "lib/celldeath_solver.hpp"

// Utils

// Physical and Domain-Decomposition parameters
#include "contexts.hpp" 

// Output
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include "FilesystemHelper.hpp"

using namespace mfem;

using InterfaceTransfer = ecm2_utils::InterfaceTransfer;
using TransferBackend = InterfaceTransfer::Backend;

IdentityMatrixCoefficient *Id = NULL;

// Forward declaration
void print_matrix(const DenseMatrix &A);
void saveConvergenceSubiter(const Array<real_t> &convergence_subiter, const std::string &outfolder, int step);

// Volumetric heat source in the sphere
constexpr real_t Sphere_Radius = 0.2;
const Vector Sphere_Center = []()
{
   Vector v(3);
   v[0] = 2.5;
   v[1] = 2.5;
   v[2] = 3.0;
   return v;
}();

real_t Qval = 1e3; // W/m^3
real_t HeatingSphere(const Vector &x, real_t t);

// Advection velocity profile
void velocity_profile(const Vector &x, Vector &b);
real_t viscosity = 1.0;
real_t dPdx = 1.0;
constexpr real_t R = 0.25; // cylinder mesh radius

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

   // Physics   
   real_t alpha = 0.0;      // Advection coefficient
   real_t reaction = 0.0;   // Reaction term

   OptionsParser args(argc, argv);
   // FE
   args.AddOption(&Heat_ctx.order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&CellDeath_ctx.order, "-oc", "--order-celldeath",
                  "Finite element order for cell death (polynomial degree).");
   args.AddOption(&Heat_ctx.pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&CellDeath_ctx.solver_type, "-cdt", "--celldeath-solver-type",
                  "Cell-death solver type: 0 - Eigen, 1 - GoTran.");
   // Mesh
   args.AddOption(&Mesh_ctx.hex, "-hex", "--hex-mesh", "-tet", "--tet-mesh",
                  "Use hexahedral mesh.");
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   // Time integrator
   args.AddOption(&Heat_ctx.ode_solver_type, "-ode", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   4 - Implicit Midpoint, 5 - SDIRK23, 6 - SDIRK34,\n\t"
                  "\t   7 - Forward Euler, 8 - RK2, 9 - RK3 SSP, 10 - RK4.");
   args.AddOption(&Sim_ctx.t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&Sim_ctx.dt, "-dt", "--time-step",
                  "Time step.");
   // Domain decomposition
   args.AddOption(&DD_ctx.omega_heat, "-omega", "--relaxation-parameter",
                  "Relaxation parameter.");
   // Physics
   args.AddOption(&Heat_ctx.k_cylinder, "-kc", "--k-cylinder",
                  "Thermal conductivity of the cylinder (W/mK).");
   args.AddOption(&Heat_ctx.k_solid, "-kb", "--k-solid",
                  "Thermal conductivity of the solid (W/mK).");
   args.AddOption(&Heat_ctx.k_fluid, "-kf", "--k-fluid",
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
   args.AddOption(&Sim_ctx.print_timing, "-pt", "--print-timing", "-no-pt", "--no-print-timing",
                  "Print timing data.");
   args.AddOption(&Sim_ctx.paraview, "-paraview", "-paraview", "-no-paraview", "--no-paraview",
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
      serial_mesh = new Mesh("../../../data/three-domains.e");
#else
      MFEM_ABORT("MFEM is not built with NetCDF support!");
#endif
   }
   else
   {
      serial_mesh = new Mesh("../../../data/three-domains.msh");
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

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Set up coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up coefficients... \033[0m";

   auto Id = new IdentityMatrixCoefficient(sdim);

   Array<int> attr_solid(0), attr_fluid(0), attr_cyl(0);
   attr_solid.Append(1), attr_fluid.Append(2), attr_cyl.Append(3);

   Heat_ctx.c_cylinder = 1.0;   // J/kgK
   Heat_ctx.rho_cylinder = 1.0; // kg/m^3

   Heat_ctx.c_solid = 1.0;   // J/kgK
   Heat_ctx.rho_solid = 1.0; // kg/m^3

   Heat_ctx.c_fluid = 1.0;   // J/kgK
   Heat_ctx.rho_fluid = 1.0; // kg/m^3

   // Conductivity
   // NOTE: if using PWMatrixCoefficient you need to create one for the boundary too
   auto *Kappa_cyl = new ScalarMatrixProductCoefficient(Heat_ctx.k_cylinder, *Id);
   auto *Kappa_fluid = new ScalarMatrixProductCoefficient(Heat_ctx.k_fluid, *Id);
   auto *Kappa_solid = new ScalarMatrixProductCoefficient(Heat_ctx.k_solid, *Id);

   // Heat Capacity
   auto *c_cyl = new ConstantCoefficient(Heat_ctx.c_cylinder);
   auto *c_fluid = new ConstantCoefficient(Heat_ctx.c_fluid);
   auto *c_solid = new ConstantCoefficient(Heat_ctx.c_solid);

   // Density
   auto *rho_cyl = new ConstantCoefficient(Heat_ctx.rho_cylinder);
   auto *rho_fluid = new ConstantCoefficient(Heat_ctx.rho_fluid);
   auto *rho_solid = new ConstantCoefficient(Heat_ctx.rho_solid);

   // Velocity profile for advectio term in cylinder α∇•(q T)
   VectorCoefficient *q = new VectorFunctionCoefficient(sdim, velocity_profile);

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create BC Handler (not populated yet)
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "Creating BCHandlers and Solvers... \033[0m";

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
   heat::HeatSolver Heat_Cylinder(cylinder_submesh, Heat_ctx.order, bcs_cyl, Kappa_cyl, c_cyl, rho_cyl, alpha, q, reaction, Heat_ctx.ode_solver_type, solv_verbose);
   heat::HeatSolver Heat_Solid(solid_submesh, Heat_ctx.order, bcs_solid, Kappa_solid, c_solid, rho_solid, Heat_ctx.ode_solver_type, solv_verbose);
   heat::HeatSolver Heat_Fluid(fluid_submesh, Heat_ctx.order, bcs_fluid, Kappa_fluid, c_fluid, rho_fluid, alpha, q, reaction, Heat_ctx.ode_solver_type, solv_verbose);

   // Grid functions in domain (inside solver)
   ParGridFunction *temperature_cylinder_gf = Heat_Cylinder.GetTemperatureGfPtr();
   ParGridFunction *temperature_solid_gf = Heat_Solid.GetTemperatureGfPtr();
   ParGridFunction *temperature_fluid_gf = Heat_Fluid.GetTemperatureGfPtr();

   celldeath::CellDeathSolver *CellDeath_Solid = nullptr;
   if (CellDeath_ctx.solver_type == 0)
      CellDeath_Solid = new celldeath::CellDeathSolverEigen(solid_submesh, CellDeath_ctx.order, temperature_solid_gf, CellDeath_ctx.A1, CellDeath_ctx.A2, CellDeath_ctx.A3, CellDeath_ctx.deltaE1, CellDeath_ctx.deltaE2, CellDeath_ctx.deltaE3);
   else if (CellDeath_ctx.solver_type == 1)
      CellDeath_Solid = new celldeath::CellDeathSolverGotran(solid_submesh, CellDeath_ctx.order, temperature_solid_gf, CellDeath_ctx.A1, CellDeath_ctx.A2, CellDeath_ctx.A3, CellDeath_ctx.deltaE1, CellDeath_ctx.deltaE2, CellDeath_ctx.deltaE3);
   else
      MFEM_ABORT("Invalid cell death solver type.");

   ParFiniteElementSpace *fes_cylinder = Heat_Cylinder.GetFESpace();
   ParFiniteElementSpace *fes_solid = Heat_Solid.GetFESpace();
   ParFiniteElementSpace *fes_fluid = Heat_Fluid.GetFESpace();
   ParFiniteElementSpace *fes_grad_cylinder = Heat_Cylinder.GetVectorFESpace();
   ParFiniteElementSpace *fes_grad_solid = Heat_Solid.GetVectorFESpace();
   ParFiniteElementSpace *fes_grad_fluid = Heat_Fluid.GetVectorFESpace();

   // Grid functions for interface transfer --> need it for gridfunction coefficients
   ParGridFunction *temperature_fc_cylinder = new ParGridFunction(fes_cylinder); *temperature_fc_cylinder = 0.0;
   ParGridFunction *temperature_fc_fluid = new ParGridFunction(fes_fluid); *temperature_fc_fluid = 0.0;
   ParGridFunction *temperature_sc_solid = new ParGridFunction(fes_solid); *temperature_sc_solid = 0.0;
   ParGridFunction *temperature_sc_cylinder = new ParGridFunction(fes_cylinder); *temperature_sc_cylinder = 0.0;
   ParGridFunction *temperature_fs_solid = new ParGridFunction(fes_solid); *temperature_fs_solid = 0.0;
   ParGridFunction *temperature_fs_fluid = new ParGridFunction(fes_fluid); *temperature_fs_fluid = 0.0;

   ParGridFunction *heatFlux_fs_solid = new ParGridFunction(fes_grad_solid); *heatFlux_fs_solid = 0.0;
   ParGridFunction *heatFlux_fs_fluid = new ParGridFunction(fes_grad_fluid); *heatFlux_fs_fluid = 0.0;
   ParGridFunction *heatFlux_fc_cylinder = new ParGridFunction(fes_grad_cylinder); *heatFlux_fc_cylinder = 0.0;
   ParGridFunction *heatFlux_fc_fluid = new ParGridFunction(fes_grad_fluid); *heatFlux_fc_fluid = 0.0;
   ParGridFunction *heatFlux_sc_solid = new ParGridFunction(fes_grad_solid); *heatFlux_sc_solid = 0.0;
   ParGridFunction *heatFlux_sc_cylinder = new ParGridFunction(fes_grad_cylinder); *heatFlux_sc_cylinder = 0.0;

   // Grid functions and coefficients for error computation
   ParGridFunction *temperature_solid_prev_gf = new ParGridFunction(fes_solid); *temperature_solid_prev_gf = 0.0;
   ParGridFunction *temperature_fluid_prev_gf = new ParGridFunction(fes_fluid); *temperature_fluid_prev_gf = 0.0;
   ParGridFunction *temperature_cylinder_prev_gf = new ParGridFunction(fes_cylinder); *temperature_cylinder_prev_gf = 0.0;
   GridFunctionCoefficient temperature_solid_prev_coeff(temperature_solid_prev_gf);
   GridFunctionCoefficient temperature_fluid_prev_coeff(temperature_fluid_prev_gf);
   GridFunctionCoefficient temperature_cylinder_prev_coeff(temperature_cylinder_prev_gf);

   // Grid functions for visualization
   ParGridFunction *velocity_gf = new ParGridFunction(fes_grad_fluid); *velocity_gf = 0.0;

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Populate BC Handler
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Array<int> fluid_cylinder_interface;
   Array<int> fluid_solid_interface;
   Array<int> solid_cylinder_interface;

   Array<int> fluid_lateral_attr;
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
      fluid_cylinder_interface = 8;
      fluid_solid_interface.SetSize(1);
      fluid_solid_interface = 3;
      solid_cylinder_interface.SetSize(1);
      solid_cylinder_interface = 4;

      fluid_lateral_attr.SetSize(1);
      fluid_lateral_attr = 5;
      fluid_top_attr.SetSize(1);
      fluid_top_attr = 6;
      solid_lateral_attr.SetSize(1);
      solid_lateral_attr = 2;
      solid_bottom_attr.SetSize(1);
      solid_bottom_attr = 1;
      cylinder_top_attr.SetSize(1);
      cylinder_top_attr = 7;

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

      fluid_lateral_attr = bdr_attr_sets.GetAttributeSet("Fluid Lateral");
      fluid_top_attr = bdr_attr_sets.GetAttributeSet("Fluid Top");
      solid_lateral_attr = bdr_attr_sets.GetAttributeSet("Solid Lateral");
      solid_bottom_attr = bdr_attr_sets.GetAttributeSet("Solid Bottom");
      cylinder_top_attr = bdr_attr_sets.GetAttributeSet("Cylinder Top");

      // Extract boundary attributes markers on parent mesh (needed for GSLIB interpolation)
      fluid_cylinder_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Cylinder-Fluid");
      fluid_solid_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Solid-Fluid");
      solid_cylinder_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Cylinder-Solid");
   }

   Array<int> solid_domain_attributes = AttributeSets::AttrToMarker(solid_submesh->attributes.Max(), solid_domain_attribute);
   Array<int> fluid_domain_attributes = AttributeSets::AttrToMarker(fluid_submesh->attributes.Max(), fluid_domain_attribute);
   Array<int> cylinder_domain_attributes = AttributeSets::AttrToMarker(cylinder_submesh->attributes.Max(), cylinder_domain_attribute);

   // NOTE: each submesh requires a different marker set as bdr attributes are generated per submesh (size can be different)
   // They can be converted as below using the attribute sets and the max attribute number for the specific submesh
   // If you don't want to convert and the attribute is just one number, you can add bcs or volumetric terms using the int directly (will take care of creating the marker array)
   // Array<int> fluid_cylinder_interface_c = AttributeSets::AttrToMarker(cylinder_submesh->bdr_attributes.Max(), fluid_cylinder_interface);
   // Array<int> fluid_cylinder_interface_f = AttributeSets::AttrToMarker(fluid_submesh->bdr_attributes.Max(), fluid_cylinder_interface);


   // Fluid:
   // - T = T_fluid on top/lateral walls
   // - Dirichlet   on  Γfs
   // - Dirichlet   on  Γfc

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for fluid domain... \033[0m" << std::endl;

   GridFunctionCoefficient *temperature_fs_solid_coeff = new GridFunctionCoefficient(temperature_fs_fluid);
   GridFunctionCoefficient *temperature_fc_cylinder_coeff = new GridFunctionCoefficient(temperature_fc_fluid);
   bcs_fluid->AddDirichletBC(temperature_fs_solid_coeff, fluid_solid_interface[0]);
   bcs_fluid->AddDirichletBC(temperature_fc_cylinder_coeff, fluid_cylinder_interface[0]); // Don't own the coefficient, it'll be deleted already in the previous line
   bcs_fluid->AddDirichletBC(Heat_ctx.T_fluid, fluid_lateral_attr[0]);
   bcs_fluid->AddDirichletBC(Heat_ctx.T_fluid, fluid_top_attr[0]);

   // Solid:
   // - T = T_solid on bottom/lateral walls
   // - Neumann   on  Γsc
   // - Neumann   on  Γfs
   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for solid domain...\033[0m" << std::endl;

   VectorGridFunctionCoefficient *heatFlux_fs_solid_coeff = new VectorGridFunctionCoefficient(heatFlux_fs_solid);
   VectorGridFunctionCoefficient *heatFlux_sc_solid_coeff = new VectorGridFunctionCoefficient(heatFlux_sc_solid);

   bcs_solid->AddNeumannVectorBC(heatFlux_fs_solid_coeff, fluid_solid_interface[0]);
   bcs_solid->AddNeumannVectorBC(heatFlux_sc_solid_coeff, solid_cylinder_interface[0]);
   bcs_solid->AddDirichletBC(Heat_ctx.T_solid, solid_lateral_attr[0]);
   bcs_solid->AddDirichletBC(Heat_ctx.T_solid, solid_bottom_attr[0]);

   // Cylinder:
   // - T = T_cylinder on top wall
   // - Dirichlet  on  Γsc
   // - Neumann    on  Γfc

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for cylinder domain...\033[0m" << std::endl;

   VectorGridFunctionCoefficient *heatFlux_fc_cylinder_coeff = new VectorGridFunctionCoefficient(heatFlux_fc_cylinder);
   GridFunctionCoefficient *temperature_sc_cylinder_coeff = new GridFunctionCoefficient(temperature_sc_cylinder);

   bcs_cyl->AddNeumannVectorBC(heatFlux_fc_cylinder_coeff, fluid_cylinder_interface[0]);
   bcs_cyl->AddDirichletBC(temperature_sc_cylinder_coeff, solid_cylinder_interface[0]);
   bcs_cyl->AddDirichletBC(Heat_ctx.T_cylinder, cylinder_top_attr[0]);

   Heat_Cylinder.AddVolumetricTerm(HeatingSphere, cylinder_domain_attributes);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Setup interface transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Note: 
   // - GSLIB is required to transfer custom qoi (e.g. heatflux) 
   // - Transfer of the temperature field can be done both with GSLIB or with transfer map from ParSubMesh objects
   // - In this case GSLIB is used for both qoi and temperature field transfer

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up interface transfer... \033[0m" << std::endl;

   BidirectionalInterfaceTransfer finder_solid_to_cylinder(*temperature_solid_gf, *temperature_cylinder_gf, solid_cylinder_interface_marker, TransferBackend::GSLIB);
   BidirectionalInterfaceTransfer finder_fluid_to_cylinder(*temperature_fluid_gf, *temperature_cylinder_gf, fluid_cylinder_interface_marker, TransferBackend::GSLIB);
   BidirectionalInterfaceTransfer finder_fluid_to_solid(*temperature_fluid_gf, *temperature_solid_gf, fluid_solid_interface_marker, TransferBackend::GSLIB);

   // Extract the indices of elements at the interface and convert them to markers
   // Useful to restrict the computation of the L2 error to the interface
   Array<int> tmp1, tmp2;
   finder_fluid_to_solid.GetElementIdxDst(tmp1);
   finder_solid_to_cylinder.GetElementIdxSrc(tmp2);
   Array<int> solid_interfaces_element_idx = tmp1 && tmp2;

   finder_fluid_to_solid.GetElementIdxSrc(tmp1);
   finder_fluid_to_cylinder.GetElementIdxSrc(tmp2);
   Array<int> fluid_interfaces_element_idx = tmp1 && tmp2;

   finder_fluid_to_cylinder.GetElementIdxDst(tmp1);
   finder_solid_to_cylinder.GetElementIdxDst(tmp2);
   Array<int> cylinder_interfaces_element_idx = tmp1 && tmp2;

   tmp1.DeleteAll();
   tmp2.DeleteAll();
   
   // 3. Define QoI (heatflux) on the source meshes (cylinder, solid, fluid)
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

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up solvers and assembling forms... \033[0m" << std::endl;

   Heat_Solid.EnablePA(Heat_ctx.pa);
   Heat_Solid.Setup();

   Heat_Cylinder.EnablePA(Heat_ctx.pa);
   Heat_Cylinder.Setup();

   Heat_Fluid.EnablePA(Heat_ctx.pa);
   Heat_Fluid.Setup();

   // Setup ouput
   ParaViewDataCollection paraview_dc_cylinder("Heat-Cylinder", cylinder_submesh.get());
   ParaViewDataCollection paraview_dc_solid("Heat-Solid", solid_submesh.get());
   ParaViewDataCollection paraview_dc_fluid("Heat-Fluid", fluid_submesh.get());
   ParaViewDataCollection paraview_dc_celldeath("CellDeath-Solid", solid_submesh.get());  
   if (Sim_ctx.paraview)
   {
      paraview_dc_cylinder.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_cylinder.SetDataFormat(VTKFormat::ASCII);
      paraview_dc_cylinder.SetCompressionLevel(9);
      Heat_Cylinder.RegisterParaviewFields(paraview_dc_cylinder);

      paraview_dc_solid.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_solid.SetDataFormat(VTKFormat::ASCII);
      paraview_dc_solid.SetCompressionLevel(9);
      Heat_Solid.RegisterParaviewFields(paraview_dc_solid);

      paraview_dc_fluid.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_fluid.SetDataFormat(VTKFormat::ASCII);
      paraview_dc_fluid.SetCompressionLevel(9);
      Heat_Fluid.RegisterParaviewFields(paraview_dc_fluid);
      Heat_Fluid.AddParaviewField("velocity", velocity_gf);

      paraview_dc_celldeath.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_celldeath.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_celldeath.SetCompressionLevel(9);
      CellDeath_Solid->RegisterParaviewFields(paraview_dc_celldeath);
   }

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Perform time-integration (looping over the time iterations, step, with a
   //     time-step Sim_ctx.dt).
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nStarting time-integration... \033[0m" << std::endl;

   ConstantCoefficient T0solid(Heat_ctx.T_solid);
   temperature_solid_gf->ProjectCoefficient(T0solid);
   Heat_Solid.SetInitialTemperature(*temperature_solid_gf);

   ConstantCoefficient T0cylinder(Heat_ctx.T_cylinder);
   temperature_cylinder_gf->ProjectCoefficient(T0cylinder);
   Heat_Cylinder.SetInitialTemperature(*temperature_cylinder_gf);

   ConstantCoefficient T0fluid(Heat_ctx.T_fluid);
   temperature_fluid_gf->ProjectCoefficient(T0fluid);
   Heat_Fluid.SetInitialTemperature(*temperature_fluid_gf);

   velocity_gf->ProjectCoefficient(*q);

   real_t t = 0.0;
   bool last_step = false;

   // Write fields to disk for VisIt
   if (Sim_ctx.paraview)
   {
      Heat_Solid.WriteFields(0, t);
      Heat_Cylinder.WriteFields(0, t);
      Heat_Fluid.WriteFields(0, t);
      CellDeath_Solid->WriteFields(0, t);
   }

   bool converged = false;
   real_t tol = 1.0e-10;
   int max_iter = 100;

   // Vectors for error computation and relaxation
   Vector temperature_solid(temperature_solid_gf->Size());   temperature_solid = Heat_ctx.T_solid;
   Vector temperature_cylinder(temperature_cylinder_gf->Size()); temperature_cylinder = Heat_ctx.T_cylinder;
   Vector temperature_fluid(temperature_fluid_gf->Size()); temperature_fluid = Heat_ctx.T_fluid;

   Vector temperature_solid_prev(temperature_solid_gf->Size());
   temperature_solid_prev = Heat_ctx.T_solid;
   Vector temperature_cylinder_prev(temperature_cylinder_gf->Size());
   temperature_cylinder_prev = Heat_ctx.T_cylinder;
   Vector temperature_fluid_prev(temperature_fluid_gf->Size()); temperature_fluid_prev = Heat_ctx.T_fluid;

   Vector temperature_solid_tn(*temperature_solid_gf->GetTrueDofs()); temperature_solid_tn = Heat_ctx.T_solid;
   Vector temperature_cylinder_tn(*temperature_cylinder_gf->GetTrueDofs()); temperature_cylinder_tn = Heat_ctx.T_cylinder;
   Vector temperature_fluid_tn(*temperature_fluid_gf->GetTrueDofs()); temperature_fluid_tn = Heat_ctx.T_fluid;

   int cyl_dofs = Heat_Cylinder.GetProblemSize();
   int fluid_dofs = Heat_Fluid.GetProblemSize();
   int solid_temperature_dofs = Heat_Solid.GetProblemSize();
   int solid_celldeath_dofs = CellDeath_Solid->GetProblemSize();

   // Integration rule for the L2 error
   int order_quad = std::max(2, Heat_ctx.order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   if (Mpi::Root())
   {
      out << " Cylinder dofs: " << cyl_dofs << std::endl;
      out << " Fluid dofs: " << fluid_dofs << std::endl;
      out << " Solid temperature dofs: " << solid_temperature_dofs << std::endl;
      out << " Solid celldeath dofs: " << solid_celldeath_dofs << std::endl;
   }

   // Timing
   StopWatch chrono, chrono_total_subiter, chrono_total;
   real_t t_total_subiter, t_transfer_fluid, t_transfer_solid, t_transfer_cylinder, t_solve_fluid, t_solve_solid, t_solve_cylinder, t_solve_celldeath, t_relax_fluid, t_relax_solid, t_relax_cylinder, t_error_bdry, t_paraview;

   if (Mpi::Root())
   {
      out << "-------------------------------------------------------------------------------------------------"
          << std::endl;
      out << std::left << std::setw(16) << "Step" << std::setw(16) << "Time" << std::setw(16) << "Sim_ctx.dt" << std::setw(16) << "Sub-iterations (F-S-C)" << std::endl;
      out << "-------------------------------------------------------------------------------------------------"
          << std::endl;
   }

   // Outer loop for time integration
   DD_ctx.omega_heat_fluid = DD_ctx.omega_heat; // TODO: Add different relaxation parameters for each domain
   DD_ctx.omega_heat_solid = DD_ctx.omega_heat;
   DD_ctx.omega_heat_cyl = DD_ctx.omega_heat;

   int num_steps = (int)(Sim_ctx.t_final / Sim_ctx.dt);
   Array2D<real_t> convergence(num_steps, 3);
   Array<real_t> convergence_subiter;
   for (int step = 1; !last_step; step++)
   {
      if (Mpi::Root())
      {
         mfem::out << std::left << std::setw(16) << step << std::setw(16) << t << std::setw(16) << Sim_ctx.dt << std::setw(16);
      }

      if (t + Sim_ctx.dt >= Sim_ctx.t_final - Sim_ctx.dt / 2)
      {
         last_step = true;
      }

      temperature_solid_tn = *temperature_solid_gf->GetTrueDofs();
      temperature_cylinder_tn = *temperature_cylinder_gf->GetTrueDofs();
      temperature_fluid_tn = *temperature_fluid_gf->GetTrueDofs();

      // Inner loop for the segregated solve
      int iter = 0;
      int iter_solid = 0;
      int iter_fluid = 0;
      int iter_cylinder = 0;
      real_t norm_diff = 2 * tol;
      real_t norm_diff_solid = 2 * tol;
      real_t norm_diff_fluid = 2 * tol;
      real_t norm_diff_cylinder = 2 * tol;

      bool converged_solid = false;
      bool converged_fluid = false;
      bool converged_cylinder = false;
      
      chrono_total.Clear(); chrono_total.Start();

      while (!converged && iter <= max_iter)
      {
         chrono_total_subiter.Clear(); chrono_total_subiter.Start();

         // Store the previous temperature on domains for convergence
         temperature_solid_gf->GetTrueDofs(temperature_solid_prev);
         temperature_fluid_gf->GetTrueDofs(temperature_fluid_prev);
         temperature_cylinder_gf->GetTrueDofs(temperature_cylinder_prev);
         temperature_solid_prev_gf->SetFromTrueDofs(temperature_solid_prev);
         temperature_fluid_prev_gf->SetFromTrueDofs(temperature_fluid_prev);
         temperature_cylinder_prev_gf->SetFromTrueDofs(temperature_cylinder_prev);

         /////////////////////////////////////////////////////////////////
         //         Fluid Domain (F), Dirichlet(S)-Dirichlet(C)         //
         /////////////////////////////////////////////////////////////////

         MPI_Barrier(parent_mesh.GetComm());

         // if (!converged_fluid)
         { // S->F: Transfer T, C->F: Transfer T
            chrono.Clear(); chrono.Start();
            finder_fluid_to_cylinder.InterpolateBackward(*temperature_cylinder_gf, *temperature_fc_fluid);
            finder_fluid_to_solid.InterpolateBackward(*temperature_solid_gf, *temperature_fs_fluid);
            chrono.Stop(); 
            t_transfer_fluid = chrono.RealTime();

            // Step in the fluid domain
            chrono.Clear(); chrono.Start();
            temperature_fluid_gf->SetFromTrueDofs(temperature_fluid_tn);
            Heat_Fluid.Step(t, Sim_ctx.dt, step, false);
            temperature_fluid_gf->GetTrueDofs(temperature_fluid);
            t -= Sim_ctx.dt; // Reset t to same time step, since t is incremented in the Step function
            chrono.Stop();
            t_solve_fluid = chrono.RealTime();

            // Relaxation
            // T_fluid(j+1) = ω * T_fluid,j+1 + (1 - ω) * T_fluid,j
            chrono.Clear(); chrono.Start();
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
         //          Solid Domain (S), Neumann(F)-Neumann(C)            //
         /////////////////////////////////////////////////////////////////

         MPI_Barrier(parent_mesh.GetComm());

         // if (!converged_solid)
         { // F->S: Transfer k ∇T_wall, C->S: Transfer k ∇T_wall
            chrono.Clear(); chrono.Start();
            finder_fluid_to_solid.InterpolateQoIForward(heatFlux_fluid, *heatFlux_fs_solid);
            finder_solid_to_cylinder.InterpolateQoIBackward(heatFlux_cyl, *heatFlux_sc_solid);
            chrono.Stop();
            t_transfer_solid = chrono.RealTime();

            // Step in the solid domain
            chrono.Clear(); chrono.Start();
            temperature_solid_gf->SetFromTrueDofs(temperature_solid_tn);
            Heat_Solid.Step(t, Sim_ctx.dt, step, false);
            temperature_solid_gf->GetTrueDofs(temperature_solid);
            t -= Sim_ctx.dt; // Reset t to same time step, since t is incremented in the Step function
            chrono.Stop();
            t_solve_solid = chrono.RealTime();

            // Relaxation
            // T_wall(j+1) = ω * T_solid,j+1 + (1 - ω) * T_solid,j
            chrono.Clear(); chrono.Start();
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
            chrono.Clear(); chrono.Start();
            finder_fluid_to_cylinder.InterpolateQoIForward(heatFlux_fluid, *heatFlux_fc_cylinder);
            finder_solid_to_cylinder.InterpolateForward(*temperature_solid_gf, *temperature_sc_cylinder);
            chrono.Stop();
            t_transfer_cylinder = chrono.RealTime();

            // Step in the cylinder domain
            chrono.Clear(); chrono.Start();
            temperature_cylinder_gf->SetFromTrueDofs(temperature_cylinder_tn);
            Heat_Cylinder.Step(t, Sim_ctx.dt, step, false);
            temperature_cylinder_gf->GetTrueDofs(temperature_cylinder);
            t -= Sim_ctx.dt; // Reset t to same time step, since t is incremented in the Step function
            chrono.Stop();
            t_solve_cylinder = chrono.RealTime();

            // Relaxation
            // T_cylinder(j+1) = ω * T_cylinder,j+1 + (1 - ω) * T_cylinder,j
            chrono.Clear(); chrono.Start();
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
         chrono.Clear(); chrono.Start();
         real_t global_norm_diff_solid = temperature_solid_gf->ComputeL2Error(temperature_solid_prev_coeff, irs, &solid_interfaces_element_idx);
         real_t global_norm_diff_fluid = temperature_fluid_gf->ComputeL2Error(temperature_fluid_prev_coeff, irs, &fluid_interfaces_element_idx);
         real_t global_norm_diff_cylinder = temperature_cylinder_gf->ComputeL2Error(temperature_cylinder_prev_coeff, irs, &cylinder_interfaces_element_idx);   
         chrono.Stop();
         t_error_bdry = chrono.RealTime();

         // Check convergence on domains
         converged_solid = global_norm_diff_solid < tol;
         converged_fluid = global_norm_diff_fluid < tol;
         converged_cylinder = global_norm_diff_cylinder < tol;

         // Check convergence
         converged = converged_solid && converged_fluid && converged_cylinder;

         iter++;

         if (Mpi::Root())
         {
            convergence_subiter.Append(norm_diff);
         }

         chrono_total_subiter.Stop();
         t_total_subiter = chrono_total_subiter.RealTime();
      }

      if (Mpi::Root() && Sim_ctx.save_convergence)
      {
         convergence(step - 1, 0) = t;
         convergence(step - 1, 1) = iter;
         convergence(step - 1, 2) = norm_diff;
         saveConvergenceSubiter(convergence_subiter, Sim_ctx.outfolder, step);
         convergence_subiter.DeleteAll();
      }

      // Reset the convergence flag and time for the next iteration
      if (Mpi::Root())
      { // Print iterations for (fluid-solid-cylinder)
         mfem::out << std::setw(2) << iter_fluid << std::setw(2) << "-"
                   << std::setw(2) << iter_solid << std::setw(2) << "-"
                   << std::setw(2) << iter_cylinder << std::endl;
      }

      if (iter > max_iter)
      {
         if (Mpi::Root())
            mfem::out << "Warning: Maximum number of iterations reached. Error: " << norm_diff << " << Aborting!" << std::endl;
         break;
      }

      /////////////////////////////////////////
      //         Solve CELL DEATH         //
      /////////////////////////////////////////
      
      chrono.Clear(); chrono.Start();
      if (Mpi::Root())
         mfem::out << "\033[32mSolving CellDeath problem on solid ... \033[0m";
      
      CellDeath_Solid->Solve(t, Sim_ctx.dt);
      
      if (Mpi::Root())
         mfem::out << "\033[32mdone.\033[0m" << std::endl;
      chrono.Stop();
      t_solve_celldeath = chrono.RealTime();

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
         out << "Solve CellDeath (Solid): " << t_solve_celldeath << " s" << std::endl;
         out << "Paraview: " << t_paraview << " s" << std::endl;
         out << "Total: " << chrono_total.RealTime() << " s" << std::endl;
         out << "------------------------------------------------------------" << std::endl;
      }
   }

   // Save convergence data
   if (Mpi::Root() && Sim_ctx.save_convergence)
   {
      std::string outputFilePath = std::string(Sim_ctx.outfolder) + "/convergence" + "/convergence.txt";
      std::ofstream outFile(outputFilePath);
      if (outFile.is_open())
      {
         convergence.Save(outFile);
         outFile.close();
      }
      else
      {
         std::cerr << "Unable to open file: " << std::string(Sim_ctx.outfolder) + "/convergence.txt" << std::endl;
      }
   }

   
   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // No need to delete:
   // - BcHandler objects: owned by HeatOperator
   // - Coefficients provide to BCHandler: owned by CoeffContainers

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

   delete q;

   delete temperature_fc_cylinder;
   delete temperature_fc_fluid;
   delete temperature_sc_solid;
   delete temperature_sc_cylinder;
   delete temperature_fs_solid;
   delete temperature_fs_fluid;

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

   delete Id;


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

void saveConvergenceSubiter(const Array<real_t> &convergence_subiter, const std::string &outfolder, int step)
{
   // Create the output folder path
   std::string outputFolder = outfolder + "/convergence";

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
   real_t x = c(0);
   real_t y = c(1);

   q(0) = dPdx;
   q(1) = 0.0;
   q(2) = 0.0;
}

real_t HeatingSphere(const Vector &x, real_t t)
{
   real_t Q = 0.0;
   real_t r = sqrt((x[0] - Sphere_Center[0]) * (x[0] - Sphere_Center[0]) +
                   (x[1] - Sphere_Center[1]) * (x[1] - Sphere_Center[1]) +
                   (x[2] - Sphere_Center[2]) * (x[2] - Sphere_Center[2]));
   Q = r < Sphere_Radius ? Qval : 0.0; // W/m^2

   return Q;
}
