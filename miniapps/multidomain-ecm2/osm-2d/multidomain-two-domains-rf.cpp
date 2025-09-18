// Solve quasi-static electrostatics problem with two domains (fluid and solid).
//                            ∇•σ∇Φ = 0
// The problem is solved using a segregated approach with two-way coupling (Robin-Robin)
// until convergence is reached.
// Works on both hexahedral and tetrahedral meshes, with optional partial assembly (hexahedral only). 
// The conductivity tensor σ can be anisotropic (Change EulerAngles function).
//
// Sample run:
// 1. Tetrahedral mesh
//    mpirun -np 4 ./multidomain-two-domains-rf-osm-2d -o 3 -tet -alpha '1e3 1e-3'
// 2. Hexahedral mesh
//    mpirun -np 4 ./multidomain-two-domains-rf-osm-2d -o 3 -hex -alpha '1e3 1e-3'
// 3. Hexahedral mesh with partial assembly
//    mpirun -np 4 ./multidomain-two-domains-rf-osm-2d -o 3 -hex -pa-rf -alpha '1e3 1e-3'


// MFEM library
#include "mfem.hpp"

// Multiphysics modules
#include "lib/electrostatics_solver.hpp"

// Interface transfer
#include "interface_transfer.hpp"

// Utils
#include "../common/mesh_extras.hpp"
#include "anisotropy_utils.hpp"

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
void saveConvergenceArray(const Array2D<real_t> &data, const std::string &outfolder, const std::string &name, int step);

int main(int argc, char *argv[]){

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

   DD_ctx.alpha_rf.SetSize(2);
   DD_ctx.alpha_rf[0] = 1e3; // Fluid
   DD_ctx.alpha_rf[1] = 1e-3; // Solid

   // FE
   args.AddOption(&RF_ctx.order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&RF_ctx.pa, "-pa-rf", "--partial-assembly-rf", "-no-pa-rf", "--no-partial-assembly-rf",
                  "Enable or disable partial assembly.");
   // Mesh
   args.AddOption(&Mesh_ctx.hex, "-hex", "--hex-mesh", "-tet", "--tet-mesh",
                  "Use hexahedral mesh.");
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   // Domain decomposition
   args.AddOption(&DD_ctx.omega_rf, "-omega", "--relaxation-parameter",
                  "Relaxation parameter.");
   args.AddOption(&DD_ctx.alpha_rf, "-alpha", "--alpha-robin",
                  "Robin-Robin coupling parameters (Fluid, Solid).");
   // Physics
   args.AddOption(&RF_ctx.phi_applied, "-phi", "--applied-potential",
                  "Applied potential.");
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


   // Check on provided alpha values
   if (DD_ctx.alpha_rf.Size() != 2)
   {
      MFEM_ABORT("2 coupling parameters must be provided (fluid, solid), but " << DD_ctx.alpha_rf.Size() << " were provided.");
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nLoading mesh... \033[0m";

   // Load serial mesh
   Mesh *serial_mesh = nullptr;
#ifdef MFEM_USE_NETCDF
   if (Mesh_ctx.hex)
   // Load mesh (NETCDF required)
      serial_mesh = new Mesh("../../../data/rfa-quad.e");
   else
      serial_mesh = new Mesh("../../../data/rfa-tri.e");
   #else
   MFEM_ABORT("MFEM is not built with NetCDF support!");
#endif

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
   ExportMeshwithPartitioning(Sim_ctx.outfolder, *serial_mesh, partitioning);
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

   solid_domain_attribute = attr_sets.GetAttributeSet("Solid");
   fluid_domain_attribute = attr_sets.GetAttributeSet("Fluid");

   auto solid_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, solid_domain_attribute));

   auto fluid_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, fluid_domain_attribute));

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

      
   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Set up coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up coefficients... \033[0m";

   auto Id = new IdentityMatrixCoefficient(sdim);

   real_t sigma_fluid = 1.0;
   real_t sigma_solid = 1.0;

   // Conductivity
   // NOTE: if using PWMatrixCoefficient you need to create one for the boundary too
   auto *Sigma_fluid = new ScalarMatrixProductCoefficient(sigma_fluid, *Id);
   auto *Sigma_solid = new ScalarMatrixProductCoefficient(sigma_solid, *Id);

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create BC Handler (not populated yet)
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "Creating BCHandlers and Solvers... \033[0m";

   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool bc_verbose = true;

   electrostatics::BCHandler *bcs_solid = new electrostatics::BCHandler(solid_submesh, bc_verbose); // Boundary conditions handler for solid
   electrostatics::BCHandler *bcs_fluid = new electrostatics::BCHandler(fluid_submesh, bc_verbose); // Boundary conditions handler for fluid

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Create the Electrostatics Solver
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Solvers
   bool solv_verbose = false;
   electrostatics::ElectrostaticsSolver RF_Solid(solid_submesh, RF_ctx.order, bcs_solid, Sigma_solid, solv_verbose);
   electrostatics::ElectrostaticsSolver RF_Fluid(fluid_submesh, RF_ctx.order, bcs_fluid, Sigma_fluid, solv_verbose);

   // Grid functions
   ParGridFunction *phi_solid_gf = RF_Solid.GetPotentialGfPtr();
   ParGridFunction *phi_fluid_gf = RF_Fluid.GetPotentialGfPtr();

   ParFiniteElementSpace *fes_solid = RF_Solid.GetFESpace();
   ParFiniteElementSpace *fes_fluid = RF_Fluid.GetFESpace();
   ParFiniteElementSpace *fes_grad_solid = new ParFiniteElementSpace(fes_solid->GetParMesh(), fes_solid->FEColl(), sdim);
   ParFiniteElementSpace *fes_grad_fluid = new ParFiniteElementSpace(fes_fluid->GetParMesh(), fes_fluid->FEColl(), sdim);
   ParFiniteElementSpace *fes_l2_solid = RF_Solid.GetL2FESpace();

   ParGridFunction *phi_fs_fluid = new ParGridFunction(fes_fluid);
   *phi_fs_fluid = 0.0;
   ParGridFunction *phi_fs_solid = new ParGridFunction(fes_solid);
   *phi_fs_solid = 0.0;
   ParGridFunction *J_fs_solid = new ParGridFunction(fes_grad_solid);
   *J_fs_solid = 0.0;
   ParGridFunction *J_fs_fluid = new ParGridFunction(fes_grad_fluid);
   *J_fs_fluid = 0.0;

   ParGridFunction *phi_solid_prev_gf = new ParGridFunction(fes_solid);
   *phi_solid_prev_gf = 0.0;
   ParGridFunction *phi_fluid_prev_gf = new ParGridFunction(fes_fluid);
   *phi_fluid_prev_gf = 0.0;
   GridFunctionCoefficient phi_solid_prev_coeff(phi_solid_prev_gf);
   GridFunctionCoefficient phi_fluid_prev_coeff(phi_fluid_prev_gf);

   ParGridFunction *JouleHeating_gf = new ParGridFunction(fes_l2_solid); 
   *JouleHeating_gf = 0.0;

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;


   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Populate BC Handler
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Array<int> fluid_solid_interface;
   Array<int> solid_cylinder_interface;

   Array<int> fluid_lateral_left_attr;
   Array<int> fluid_lateral_right_attr;
   Array<int> fluid_top_attr;
   Array<int> solid_lateral_attr;
   Array<int> solid_bottom_attr;

   Array<int> fluid_solid_interface_marker;
   Array<int> solid_cylinder_interface_marker;

   // Extract boundary attributes
   fluid_solid_interface = bdr_attr_sets.GetAttributeSet("Fluid-Solid");
   solid_cylinder_interface = bdr_attr_sets.GetAttributeSet("Cylinder-Solid");

   fluid_lateral_left_attr = bdr_attr_sets.GetAttributeSet("Fluid-Lateral-Left");
   fluid_lateral_right_attr = bdr_attr_sets.GetAttributeSet("Fluid-Lateral-Right");
   fluid_top_attr = bdr_attr_sets.GetAttributeSet("Fluid-Top");
   solid_lateral_attr = bdr_attr_sets.GetAttributeSet("Solid-Lateral");
   solid_bottom_attr = bdr_attr_sets.GetAttributeSet("Solid-Bottom");

   // Extract boundary attributes markers on parent mesh (needed for GSLIB interpolation)
   fluid_solid_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Fluid-Solid");
   solid_cylinder_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Cylinder-Solid");

   Array<int> solid_domain_attributes = AttributeSets::AttrToMarker(solid_submesh->attributes.Max(), solid_domain_attribute);
   Array<int> fluid_domain_attributes = AttributeSets::AttrToMarker(fluid_submesh->attributes.Max(), fluid_domain_attribute);


   // NOTE: each submesh requires a different marker set as bdr attributes are generated per submesh (size can be different)
   // They can be converted as below using the attribute sets and the max attribute number for the specific submesh
   // If you don't want to convert and the attribute is just one number, you can add bcs or volumetric terms using the int directly (will take care of creating the marker array)
   // Array<int> fluid_cylinder_interface_c = AttributeSets::AttrToMarker(cylinder_submesh->bdr_attributes.Max(), fluid_cylinder_interface);
   // Array<int> fluid_cylinder_interface_f = AttributeSets::AttrToMarker(fluid_submesh->bdr_attributes.Max(), fluid_cylinder_interface);

   // Fluid:
   // - Homogeneous Neumann on top/lateral walls
   // - Homogeneous Neumann on  Γfc
   // - Robin-Robin on  Γfs

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up RF BCs for fluid domain... \033[0m" << std::endl;

   ConstantCoefficient alpha_coeff1(DD_ctx.alpha_rf[0]);
   GridFunctionCoefficient *phi_fs_fluid_coeff = new GridFunctionCoefficient(phi_fs_fluid);
   VectorGridFunctionCoefficient *J_fs_fluid_coeff = new VectorGridFunctionCoefficient(J_fs_fluid);
   bcs_fluid->AddDirichletBC(RF_ctx.phi_applied, solid_cylinder_interface[0]);
   bcs_fluid->AddGeneralRobinBC(&alpha_coeff1, &alpha_coeff1, phi_fs_fluid_coeff, J_fs_fluid_coeff, fluid_solid_interface[0], false); 

   // Solid:
   // - Dirichlet on bottom wall (ground)
   // - Dirichlet   on  Γsc      (applied potential)
   // - Robin-Robin on  Γfs
   // - Homogeneous Neumann lateral wall
   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for solid domain...\033[0m" << std::endl;
   ConstantCoefficient alpha_coeff2(DD_ctx.alpha_rf[1]);
   GridFunctionCoefficient *phi_fs_solid_coeff = new GridFunctionCoefficient(phi_fs_solid);
   VectorGridFunctionCoefficient *J_fs_solid_coeff = new VectorGridFunctionCoefficient(J_fs_solid);
   bcs_solid->AddDirichletBC(RF_ctx.phi_gnd, solid_bottom_attr[0]);
   bcs_solid->AddDirichletBC(RF_ctx.phi_applied, solid_cylinder_interface[0]);
   bcs_solid->AddGeneralRobinBC(&alpha_coeff2, &alpha_coeff2, phi_fs_solid_coeff, J_fs_solid_coeff, fluid_solid_interface[0], false); 


   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Setup interface transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////
   
   MPI_Barrier(parent_mesh.GetComm());

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up interface transfer... \033[0m" << std::endl;

   BidirectionalInterfaceTransfer finder_fluid_to_solid(fes_fluid, fes_solid, fluid_solid_interface_marker, TransferBackend::GSLIB, parent_mesh.GetComm());

   // Extract the indices of elements at the interface and convert them to markers
   // Useful to restrict the computation of the L2 error to the interface
   Array<int> fs_fluid_element_idx;
   Array<int> fs_solid_element_idx;
   finder_fluid_to_solid.GetElementIdxSrc(fs_fluid_element_idx);
   finder_fluid_to_solid.GetElementIdxDst(fs_solid_element_idx);

   Vector bdr_element_coords_fluid = finder_fluid_to_solid.GetBdrElementCoordsSrc();
   Vector bdr_element_coords_solid = finder_fluid_to_solid.GetBdrElementCoordsDst();
   int npts_fluid = bdr_element_coords_fluid.Size() / sdim;
   int npts_solid = bdr_element_coords_solid.Size() / sdim;

   // 3. Define QoI (current density) on the source meshes (cylinder, solid, fluid)
   int qoi_size_on_qp = sdim;

   // Define lamdbas to compute the current density
   auto J_solid = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      DenseMatrix SigmaMat(sdim);
      Vector gradloc(sdim);

      Sigma_solid->Eval(SigmaMat, Tr, ip);

      phi_solid_gf->GetGradient(Tr, gradloc);
      SigmaMat.Mult(gradloc, qoi_loc);
   };

   auto J_fluid = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      DenseMatrix SigmaMat(sdim);
      Vector gradloc(sdim);

      Sigma_fluid->Eval(SigmaMat, Tr, ip);

      phi_fluid_gf->GetGradient(Tr, gradloc);
      SigmaMat.Mult(gradloc, qoi_loc);
   };

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////


   MPI_Barrier(parent_mesh.GetComm());
   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up solvers and assembling forms... \033[0m" << std::endl;

   StopWatch chrono_assembly;

   chrono_assembly.Start();

   if (Mpi::Root())
      mfem::out << "\033[0mAssembling Fluid \033[0m";

   RF_Fluid.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   RF_Fluid.Setup();   

   if (Mpi::Root())
      mfem::out << "\033[0m\nAssembling Solid \033[0m" << std::endl;

   RF_Solid.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   RF_Solid.Setup();

   chrono_assembly.Stop();
   real_t assembly_time = chrono_assembly.RealTime();

   // Setup ouput
   ParaViewDataCollection paraview_dc_solid("RF-Solid", solid_submesh.get());
   ParaViewDataCollection paraview_dc_fluid("RF-Fluid", fluid_submesh.get());
   if (Sim_ctx.paraview)
   {
      paraview_dc_solid.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_solid.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_solid.SetCompressionLevel(9);
      RF_Solid.RegisterParaviewFields(paraview_dc_solid);
      RF_Solid.AddParaviewField("Joule Heating", JouleHeating_gf);

      paraview_dc_fluid.SetPrefixPath(Sim_ctx.outfolder);
      paraview_dc_fluid.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_fluid.SetCompressionLevel(9);
      RF_Fluid.RegisterParaviewFields(paraview_dc_fluid);
   }

   if (Sim_ctx.paraview)
   {
      RF_Solid.WriteFields(0, 0.0);
      RF_Fluid.WriteFields(0, 0.0);
   }

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   if (Sim_ctx.paraview)
   {
      RF_Solid.WriteFields(0, 0.1);
      RF_Fluid.WriteFields(0, 0.1);
   }

///////////////////////////////////////////////////////////////////////////////////////////////
/// 9. Solve the problem until convergence
///////////////////////////////////////////////////////////////////////////////////////////////

if (Mpi::Root())
   mfem::out << "\033[34m\nSolving... \033[0m" << std::endl;

// Write fields to disk for VisIt
bool converged = false;
real_t tol = 1.0e-12;
int max_iter = 100;
int step = 0;

if (Sim_ctx.paraview)
{
   RF_Solid.WriteFields(0);
   RF_Fluid.WriteFields(0);
}

Vector phi_solid(fes_solid->GetTrueVSize());
phi_solid = 0.0;
Vector phi_fluid(fes_fluid->GetTrueVSize());
phi_fluid = 0.0;
Vector phi_solid_prev(fes_solid->GetTrueVSize());
phi_solid_prev = 0.0;
Vector phi_fluid_prev(fes_fluid->GetTrueVSize());
phi_fluid_prev = 0.0;

int fluid_dofs = RF_Fluid.GetProblemSize();
int solid_dofs = RF_Solid.GetProblemSize();

if (Mpi::Root())
{
   mfem::out << " Fluid dofs: " << fluid_dofs << std::endl;
   mfem::out << " Solid dofs: " << solid_dofs << std::endl;
}

// Outer loop for time integration
DD_ctx.omega_rf_fluid = DD_ctx.omega_rf; // TODO: Add different relaxation parameters for each domain
DD_ctx.omega_rf_solid = DD_ctx.omega_rf;

Array2D<real_t> convergence_rf(max_iter, 3); convergence_rf = 0.0;
// Inner loop for the segregated solve
int iter = 0;
int iter_solid = 0;
int iter_fluid = 0;
real_t norm_diff = 2 * tol;
real_t norm_diff_solid = 2 * tol;
real_t norm_diff_fluid = 2 * tol;

bool converged_solid = false;
bool converged_fluid = false;

// Enable re-assembly of the RHS
bool assembleRHS = true;

// Integration rule for the L2 error
int order_quad = std::max(2, RF_ctx.order + 1);
const IntegrationRule *irs[Geometry::NumGeom];
for (int i = 0; i < Geometry::NumGeom; ++i)
{
   irs[i] = &(IntRules.Get(i, order_quad));
}

// Timing
StopWatch chrono, chrono_total;
real_t t_transfer, t_interp, t_solve_fluid, t_solve_solid, t_relax_fluid, t_relax_solid, t_error, t_error_bdry, t_paraview, t_joule;

if (Mpi::Root())
{
   out << "------------------------------------------------------------" << std::endl;
   out << std::left << std::setw(16) << "Iteration" << std::setw(16) << "Fluid Error" << std::setw(16) << "Solid Error" << std::endl;
   out << "------------------------------------------------------------" << std::endl;
}
while (!converged && iter <= max_iter)
{

   chrono_total.Clear(); chrono_total.Start();

   // Store the previous temperature on domains for convergence
   phi_solid_gf->GetTrueDofs(phi_solid_prev);
   phi_fluid_gf->GetTrueDofs(phi_fluid_prev);
   phi_solid_prev_gf->SetFromTrueDofs(phi_solid_prev);
   phi_fluid_prev_gf->SetFromTrueDofs(phi_fluid_prev);

   phi_solid_prev = 0.0;

   ///////////////////////////////////////////////////
   //         Fluid Domain (F), Dirichlet(S)        //
   ///////////////////////////////////////////////////

   MPI_Barrier(parent_mesh.GetComm());

   // Transfer
   chrono.Clear(); chrono.Start();
   //if (!converged_fluid)
   { // S->F: Φ, J
      finder_fluid_to_solid.InterpolateBackward(*phi_solid_gf, *phi_fs_fluid);
      finder_fluid_to_solid.InterpolateQoIBackward(J_solid, *J_fs_fluid);
   }
   chrono.Stop();
   t_transfer = chrono.RealTime();

   // Solve
   chrono.Clear(); chrono.Start();
   RF_Fluid.Solve(assembleRHS);
   chrono.Stop();
   t_solve_fluid = chrono.RealTime();

   // Relaxation
   chrono.Clear(); chrono.Start();
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

   // Transfer
   chrono.Clear(); chrono.Start();
   // if (!converged_solid)
   { // F->S: Φ, J
      finder_fluid_to_solid.InterpolateForward(*phi_fluid_gf, *phi_fs_solid);
      finder_fluid_to_solid.InterpolateQoIForward(J_fluid, *J_fs_solid);
   }
   chrono.Stop();
   t_interp = chrono.RealTime();

   // Solve
   chrono.Clear(); chrono.Start();
   RF_Solid.Solve(assembleRHS);
   chrono.Stop();
   t_solve_solid = chrono.RealTime();

   // Relaxation
   chrono.Clear(); chrono.Start();
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

   chrono.Clear(); chrono.Start();
   // Compute global norms
   real_t global_norm_diff_solid = phi_solid_gf->ComputeL2Error(phi_solid_prev_coeff, irs, &fs_solid_element_idx);
   real_t global_norm_diff_fluid = phi_fluid_gf->ComputeL2Error(phi_fluid_prev_coeff, irs, &fs_fluid_element_idx);
   chrono.Stop();
   t_error_bdry = chrono.RealTime();

   // Check convergence on domains
   converged_solid = (global_norm_diff_solid < tol); //   &&(iter > 0);
   converged_fluid = (global_norm_diff_fluid < tol); //   &&(iter > 0);

   // Check convergence
   converged = converged_solid && converged_fluid;

   iter++;

   if (Mpi::Root() && Sim_ctx.save_convergence)
   {
      convergence_rf(iter, 0) = iter;
      convergence_rf(iter, 1) = global_norm_diff_fluid;
      convergence_rf(iter, 2) = global_norm_diff_solid;
   }

   if (Mpi::Root())
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
      //out << "Error: " << t_error << " s" << std::endl;
      out << "Error Boundary: " << t_error_bdry << " s" << std::endl;
      out << "Total: " << chrono_total.RealTime() << " s" << std::endl;
      out << "------------------------------------------------------------" << std::endl;
   }

}

// Compute Joule heating
chrono.Clear(); chrono.Start();
// Output of time steps
RF_Solid.GetJouleHeating(*JouleHeating_gf);
chrono.Stop();
t_joule = chrono.RealTime();

// Export converged fields
chrono.Clear(); chrono.Start();
real_t t_iter = 0.1; // This will be replaced by the actual time in transient simulations
if (Sim_ctx.paraview )
{
   RF_Solid.WriteFields(1, t_iter);
   RF_Fluid.WriteFields(1, t_iter);
}
chrono.Stop();
t_paraview = chrono.RealTime();

if (Mpi::Root() && Sim_ctx.print_timing)
{ // Print times
   out << "------------------------------------------------------------" << std::endl;
   out << "Assembly: " << assembly_time << " s" << std::endl;
   out << "Joule: " << t_joule << " s" << std::endl;
   out << "Paraview: " << t_paraview << " s" << std::endl;
   out << "------------------------------------------------------------" << std::endl;
}

// Save convergence
if (Mpi::Root() && Sim_ctx.save_convergence)
{
   std::string name_rf = "RF-pre";
   saveConvergenceArray(convergence_rf, Sim_ctx.outfolder, name_rf, 0);
   convergence_rf.DeleteAll();
}

//////////////////////////////////////////////////////////////////////
//                        Clean up                                  //
//////////////////////////////////////////////////////////////////////

// Need to clean:
// - Coefficients provided to Solvers
// - Custom allocations in main
// - Call FreeData() on GSLIB finders
//
// Coefficients passed to BCHandlers are deleted, unless own=false is specified

delete Id;
delete Sigma_fluid;
delete Sigma_solid;

delete fes_grad_solid;
delete fes_grad_fluid;

delete J_fs_solid;
delete J_fs_fluid;
delete phi_fs_fluid;
delete phi_fs_solid;
delete phi_solid_prev_gf;
delete phi_fluid_prev_gf;
delete JouleHeating_gf;

delete phi_fs_fluid_coeff;
delete J_fs_fluid_coeff;
delete phi_fs_solid_coeff;
delete J_fs_solid_coeff;

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
