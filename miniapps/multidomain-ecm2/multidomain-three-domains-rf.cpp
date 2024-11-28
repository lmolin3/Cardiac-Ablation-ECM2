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
// mpirun -np 4 ./multidomain-three-domains-rf -o 4 --relaxation-parameter 0.5 --no-paraview --print-timing
// mpirun -np 4 ./multidomain-three-domains-rf -o 4 --relaxation-parameter 0.5 --paraview -of ./Output/RF/o3

#include "mfem.hpp"
#include "lib/celldeath_solver.hpp"
#include "lib/electrostatics_solver.hpp"

#include "../common/mesh_extras.hpp"

#include <fstream>
#include <sstream>
#include <sys/stat.h> // Include for mkdir
#include <iostream>
#include <memory>

using namespace mfem;

IdentityMatrixCoefficient *Id = NULL;

// Forward declaration
void print_matrix(const DenseMatrix &A);
void saveConvergenceSubiter(const Array<real_t> &convergence_subiter, const std::string &outfolder, int step);

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

   // Mesh
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   // Domain decomposition
   real_t omega = 0.5; // Relaxation parameter
   real_t omega_fluid;
   real_t omega_solid;
   real_t omega_cyl;
   // Postprocessing
   bool print_timing = false;
   bool visit = false;
   bool paraview = true;
   int save_freq = 1; // Save fields every 'save_freq' time steps
   const char *outfolder = "";

   OptionsParser args(argc, argv);
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
   // Domain decomposition
   args.AddOption(&omega, "-omega", "--relaxation-parameter",
                  "Relaxation parameter.");
   // Physics

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

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nLoading mesh... \033[0m";

   // Load serial mesh
   Mesh *serial_mesh = new Mesh("../../data/three-domains.msh");
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

   Array<int> solid_domain_attribute = attr_sets.GetAttributeSet("Solid");
   Array<int> fluid_domain_attribute = attr_sets.GetAttributeSet("Fluid");
   Array<int> cylinder_domain_attribute = attr_sets.GetAttributeSet("Cylinder");

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
   attr_solid.Append(1), attr_fluid.Append(2);

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
   electrostatics::ElectrostaticsSolver RF_Solid(solid_submesh, order, bcs_solid, Sigma_solid, solv_verbose);
   electrostatics::ElectrostaticsSolver RF_Fluid(fluid_submesh, order, bcs_fluid, Sigma_fluid, solv_verbose);

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
   ParGridFunction *E_fs_solid = new ParGridFunction(fes_grad_solid);
   *E_fs_solid = 0.0;
   ParGridFunction *E_fs_fluid = new ParGridFunction(fes_grad_fluid);
   *E_fs_fluid = 0.0;

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

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Populate BC Handler
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Extract boundary attributes
   Array<int> fluid_cylinder_interface = bdr_attr_sets.GetAttributeSet("Cylinder-Fluid");
   Array<int> fluid_solid_interface = bdr_attr_sets.GetAttributeSet("Solid-Fluid");
   Array<int> solid_cylinder_interface = bdr_attr_sets.GetAttributeSet("Cylinder-Solid");

   Array<int> fluid_lateral_attr = bdr_attr_sets.GetAttributeSet("Fluid Lateral");
   Array<int> fluid_top_attr = bdr_attr_sets.GetAttributeSet("Fluid Top");
   Array<int> solid_lateral_attr = bdr_attr_sets.GetAttributeSet("Solid Lateral");
   Array<int> solid_bottom_attr = bdr_attr_sets.GetAttributeSet("Solid Bottom");

   Array<int> solid_domain_attributes = AttributeSets::AttrToMarker(solid_submesh->attributes.Max(), solid_domain_attribute);
   Array<int> fluid_domain_attributes = AttributeSets::AttrToMarker(fluid_submesh->attributes.Max(), fluid_domain_attribute);

   // Extract boundary attributes markers on parent mesh (needed for GSLIB interpolation)
   Array<int> fluid_cylinder_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Cylinder-Fluid");
   Array<int> fluid_solid_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Solid-Fluid");
   Array<int> solid_cylinder_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Cylinder-Solid");

   // NOTE: each submesh requires a different marker set as bdr attributes are generated per submesh (size can be different)
   // They can be converted as below using the attribute sets and the max attribute number for the specific submesh
   // If you don't want to convert and the attribute is just one number, you can add bcs or volumetric terms using the int directly (will take care of creating the marker array)
   // Array<int> fluid_cylinder_interface_c = AttributeSets::AttrToMarker(cylinder_submesh->bdr_attributes.Max(), fluid_cylinder_interface);
   // Array<int> fluid_cylinder_interface_f = AttributeSets::AttrToMarker(fluid_submesh->bdr_attributes.Max(), fluid_cylinder_interface);

   // Fluid:
   // - Homogeneous Neumann on top/lateral walls
   // - Homogeneous Neumann on  Γfc
   // - Dirichlet   on  Γfs

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for fluid domain... \033[0m" << std::endl;

   GridFunctionCoefficient *phi_fs_fluid_coeff = new GridFunctionCoefficient(phi_fs_fluid);
   bcs_fluid->AddDirichletBC(phi_fs_fluid_coeff, fluid_solid_interface[0]);

   // Solid:
   // - Phi = 0 on bottom wall
   // - Dirichlet   on  Γsc
   // - Neumann   on  Γfs
   // - Homogeneous Neumann lateral wall
   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up BCs for solid domain...\033[0m" << std::endl;
   real_t phi_gnd = 0.0;
   real_t phi_applied = 1.0;
   VectorGridFunctionCoefficient *E_fs_solid_coeff = new VectorGridFunctionCoefficient(E_fs_solid);
   bcs_solid->AddDirichletBC(phi_gnd, solid_bottom_attr[0]);
   bcs_solid->AddDirichletBC(phi_applied, solid_cylinder_interface[0]);
   bcs_solid->AddNeumannVectorBC(E_fs_solid_coeff, fluid_solid_interface[0]);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Setup interface transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Note:
   // - GSLIB is required to transfer custom qoi (e.g. current density)
   // - Transfer of the potential field can be done both with GSLIB or with transfer map from ParSubMesh objects
   // - In this case GSLIB is used for both qoi and potential field transfer

   if (Mpi::Root())
      mfem::out << "\033[34m\nSetting up interface transfer... \033[0m" << std::endl;

   // Setup GSLIB for gradient transfer:
   // 1. Find points on the DESTINATION mesh
   // 2. Setup GSLIB finder on the SOURCE mesh
   // 3. Define QoI (Electric field) on the SOURCE meshes (cylinder, solid, fluid)

   // Fluid (S) --> Solid (D)
   MPI_Barrier(parent_mesh.GetComm());
   if (Mpi::Root())
      mfem::out << "\033[34mSetting up GSLIB for gradient transfer: Fluid (S) --> Solid (D)\033[0m" << std::endl;
   Array<int> fs_solid_bdr_element_idx;
   Vector fs_solid_element_coords;
   ecm2_utils::ComputeBdrQuadraturePointsCoords(fluid_solid_interface_marker, fes_solid, fs_solid_bdr_element_idx, fs_solid_element_coords);

   FindPointsGSLIB finder_fluid_to_solid(MPI_COMM_WORLD);
   finder_fluid_to_solid.Setup(*fluid_submesh);
   finder_fluid_to_solid.FindPoints(fs_solid_element_coords, Ordering::byVDIM);

   // Solid (S) --> Fluid (D)
   MPI_Barrier(parent_mesh.GetComm());
   if (Mpi::Root())
      mfem::out << "\033[34mSetting up GSLIB for gradient transfer: Solid (S) --> Fluid (D)\033[0m" << std::endl;
   Array<int> fs_fluid_bdr_element_idx;
   Vector fs_fluid_element_coords;
   ecm2_utils::ComputeBdrQuadraturePointsCoords(fluid_solid_interface_marker, fes_fluid, fs_fluid_bdr_element_idx, fs_fluid_element_coords);

   FindPointsGSLIB finder_solid_to_fluid(MPI_COMM_WORLD);
   finder_solid_to_fluid.Setup(*solid_submesh);
   finder_solid_to_fluid.FindPoints(fs_fluid_element_coords, Ordering::byVDIM);

   // Extract the indices of elements at the interface and convert them to markers
   // Useful to restrict the computation of the L2 error to the interface
   Array<int> fs_fluid_element_idx;
   Array<int> fs_solid_element_idx;
   ecm2_utils::GSLIBAttrToMarker(solid_submesh->GetNE(), finder_solid_to_fluid.GetElem(), fs_solid_element_idx);
   ecm2_utils::GSLIBAttrToMarker(fluid_submesh->GetNE(), finder_fluid_to_solid.GetElem(), fs_fluid_element_idx);

   // 3. Define QoI (current density) on the source meshes (cylinder, solid, fluid)
   int qoi_size_on_qp = sdim;

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

   RF_Solid.EnablePA(pa);
   RF_Solid.Setup();

   RF_Fluid.EnablePA(pa);
   RF_Fluid.Setup();

   // Setup ouput
   ParaViewDataCollection paraview_dc_solid("RF-Solid", solid_submesh.get());
   ParaViewDataCollection paraview_dc_fluid("RF-Fluid", fluid_submesh.get());
   if (paraview)
   {
      paraview_dc_solid.SetPrefixPath(outfolder);
      paraview_dc_solid.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_solid.SetCompressionLevel(9);
      RF_Solid.RegisterParaviewFields(paraview_dc_solid);
      RF_Solid.AddParaviewField("Joule Heating", JouleHeating_gf);

      paraview_dc_fluid.SetPrefixPath(outfolder);
      paraview_dc_fluid.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_fluid.SetCompressionLevel(9);
      RF_Fluid.RegisterParaviewFields(paraview_dc_fluid);
   }

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Solve the problem until convergence
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nSolving... \033[0m" << std::endl;

   // Write fields to disk for VisIt
   bool converged = false;
   double tol = 1.0e-4;
   int max_iter = 100;
   int step = 0;

   if (paraview)
   {
      RF_Solid.WriteFields(0);
      RF_Fluid.WriteFields(0);
   }

   Vector phi_solid(phi_solid_gf->Size());
   phi_solid = 0.0;
   Vector phi_fluid(phi_fluid_gf->Size());
   phi_fluid = 0.0;
   Vector phi_solid_prev(phi_solid_gf->Size());
   phi_solid_prev = 0.0;
   Vector phi_fluid_prev(phi_fluid_gf->Size());
   phi_fluid_prev = 0.0;

   int fluid_dofs = RF_Fluid.GetProblemSize();
   int solid_dofs = RF_Solid.GetProblemSize();

   if (Mpi::Root())
   {
      out << " Fluid dofs: " << fluid_dofs << std::endl;
      out << " Solid dofs: " << solid_dofs << std::endl;
   }

   // Outer loop for time integration
   omega_fluid = omega; // TODO: Add different relaxation parameters for each domain
   omega_solid = omega;

   Array<real_t> convergence_subiter;
   // Inner loop for the segregated solve
   int iter = 0;
   int iter_solid = 0;
   int iter_fluid = 0;
   double norm_diff = 2 * tol;
   double norm_diff_solid = 2 * tol;
   double norm_diff_fluid = 2 * tol;

   bool converged_solid = false;
   bool converged_fluid = false;

   // Enable re-assembly of the RHS
   bool assembleRHS = true;

   // Integration rule for the L2 error
   int order_quad = std::max(2, order + 1);
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

      ///////////////////////////////////////////////////
      //         Fluid Domain (F), Dirichlet(S)        //
      ///////////////////////////////////////////////////

      MPI_Barrier(parent_mesh.GetComm());

      chrono.Clear(); chrono.Start();
      //if (!converged_fluid)
      { // S->F: Φ
         ecm2_utils::GSLIBTransfer(finder_solid_to_fluid, fs_fluid_bdr_element_idx, *phi_solid_gf, *phi_fs_fluid);
      }
      chrono.Stop();
      t_transfer = chrono.RealTime();

      chrono.Clear(); chrono.Start();
      RF_Fluid.Solve(assembleRHS);
      chrono.Stop();
      t_solve_fluid = chrono.RealTime();

      chrono.Clear(); chrono.Start();
      if (iter > 0)
      {
         phi_fluid_gf->GetTrueDofs(phi_fluid);
         phi_fluid *= omega_fluid;
         phi_fluid.Add(1.0 - omega_fluid, phi_fluid_prev);
         phi_fluid_gf->SetFromTrueDofs(phi_fluid);
      }
      chrono.Stop();
      t_relax_fluid = chrono.RealTime();

      /////////////////////////////////////////////////////////////////
      //          Solid Domain (S), Neumann(F)-Dirichlet(C)          //
      /////////////////////////////////////////////////////////////////

      MPI_Barrier(parent_mesh.GetComm());

      chrono.Clear(); chrono.Start();
      // if (!converged_solid)
      { // F->S: grad Φ
         ecm2_utils::GSLIBInterpolate(finder_fluid_to_solid, fs_solid_bdr_element_idx, fes_grad_fluid, currentDensity_fluid, *E_fs_solid, qoi_size_on_qp);
      }
      chrono.Stop();
      t_interp = chrono.RealTime();

      chrono.Clear(); chrono.Start();
      RF_Solid.Solve(assembleRHS);
      chrono.Stop(); 
      t_solve_solid = chrono.RealTime();

      chrono.Clear(); chrono.Start();
      if (iter > 0)
      {
         phi_solid_gf->GetTrueDofs(phi_solid);
         phi_solid *= omega_solid;
         phi_solid.Add(1.0 - omega_solid, phi_solid_prev);
         phi_solid_gf->SetFromTrueDofs(phi_solid);
      }
      chrono.Stop();
      t_relax_solid = chrono.RealTime();

      //////////////////////////////////////////////////////////////////////
      //                        Check convergence                         //
      //////////////////////////////////////////////////////////////////////

      //chrono.Clear(); chrono.Start();
      // Compute global norms directly
      //double global_norm_diff_solid_domain = phi_solid_gf->ComputeL2Error(phi_solid_prev_coeff);
      //double global_norm_diff_fluid_domain = phi_fluid_gf->ComputeL2Error(phi_fluid_prev_coeff);
      //chrono.Stop();
      //t_error = chrono.RealTime();

      chrono.Clear(); chrono.Start();
      // Compute global norms directly
      double global_norm_diff_solid = phi_solid_gf->ComputeL2Error(phi_solid_prev_coeff, irs, &fs_solid_element_idx);
      double global_norm_diff_fluid = phi_fluid_gf->ComputeL2Error(phi_fluid_prev_coeff, irs, &fs_fluid_element_idx);
      chrono.Stop();
      t_error_bdry = chrono.RealTime();

      // Check convergence on domains
      converged_solid = (global_norm_diff_solid < tol); //   &&(iter > 0);
      converged_fluid = (global_norm_diff_fluid < tol); //   &&(iter > 0);
      
      // Check convergence
      converged = converged_solid && converged_fluid;
      
      iter++;

      if (Mpi::Root())
      {
         convergence_subiter.Append(norm_diff);
      }

      if (Mpi::Root())
      {
         out << std::left << std::setw(16) << iter
             << std::scientific << std::setw(16) << global_norm_diff_fluid
             << std::setw(16) << global_norm_diff_solid
             << std::endl;
      }

      chrono_total.Stop();

      if (Mpi::Root() && print_timing)
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
   if (paraview )
   {
      RF_Solid.WriteFields(1, t_iter);
      RF_Fluid.WriteFields(1, t_iter);
   }
   chrono.Stop();
   t_paraview = chrono.RealTime();

   if (Mpi::Root() && print_timing)
   { // Print times
      out << "------------------------------------------------------------" << std::endl;
      out << "Joule: " << t_joule << " s" << std::endl;
      out << "Paraview: " << t_paraview << " s" << std::endl;
      out << "------------------------------------------------------------" << std::endl;
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

   delete E_fs_solid;
   delete E_fs_fluid;
   delete phi_fs_fluid;
   delete phi_solid_prev_gf;
   delete phi_fluid_prev_gf;
   delete JouleHeating_gf;

   finder_fluid_to_solid.FreeData();
   finder_solid_to_fluid.FreeData();

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
