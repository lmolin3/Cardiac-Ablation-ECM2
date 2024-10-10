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
// A diffusion equation is described on the outer box domain
//
//                 rho c dT/dt = κΔT          in outer box
//                           T = T_out        on outside wall           (IF TEST 1 is selected)
//                       k∇T•n = k∇T_wall•n   on cylinder interface
//                       k∇T•n = 0            elsewhere
//
// An advection-diffusion-reaction equation is described inside the cylinder domain
//
//                 rho c dT/dt = κΔT - α • u ∇T - β u       in cylinder
//                           T = T_wall                     on cylinder interface
//                       k∇T•n = 0                          elsewhere
//                           Q = Qval                       inside sphere (IF TEST 2 is selected)
//
// with temperature T, coefficients κ, α and prescribed velocity profile b.
//
// To couple the solutions of both equations, a segregated solve with two way
// coupling (Neumann-Dirichlet) approach is used to solve the timestep tn tn+dt:
//
// Algorithm:
//
//  0. Initialize: T_wall(0) = T_cyl(tn)
//
//  Repeat 1-3 until convergence on T_wall(j) :
//
//    1. Solve the outer box problem:
//       - Use T_wall as a Neumann boundary condition on the cylinder interface
//       - Result: T_wall,box
//
//    2. Solve the cylinder problem:
//       - Use T_wall,box as a Dirichlet boundary condition on the cylinder interface
//       - Result: T_wall,cyl = T_wall,box (set as dirichlet bc)
//
//    3. (Optional) Relax T_wall:
//       - T_wall(j+1) = ω * T_wall,cyl + (1 - ω) * T_cyl(tn)
//
//  4. Update:
//     - T_box(tn+1) = T_wall(j+1)
//
// Convergence reached when ||T_wall(j+1) - T_wall(j)|| < tol
//
// Sample run:
// Box heating:
//    mpirun -np 4 ./multidomain-neumann-dirichlet --problem 0 -rs 1 --paraview -of ./Output/Test
//
// Cylinder heating:
//    mpirun -np 4 ./multidomain-neumann-dirichlet --problem 1 -rs 1 -Q 1e4 --paraview -of ./Output/Test
//
// Advection-Diffusion:
//   mpirun -np 4 ./multidomain-neumann-dirichlet --problem 1 -rs 1 -dPdx 500 --relaxation-parameter 0.5 --paraview -of ./Output/Test
//
// Reaction-Diffusion:
//  mpirun -np 4 ./multidomain-neumann-dirichlet --problem 1 -rs 1 -beta 1 --relaxation-parameter 0.5 --paraview -of ./Output/Test
//
// Advection-Reaction-Diffusion:
//   mpirun -np 4 ./multidomain-neumann-dirichlet --problem 1 -rs 1 -dPdx 500 -beta 1 --relaxation-parameter 0.5 --paraview -of ./Output/Test
//
// Heterogeneous conductivity:
//  mpirun -np 4 ./multidomain-neumann-dirichlet --problem 1 -rs 1 -kc 0.1 -kb 10 --relaxation-parameter 0.5 --paraview -of ./Output/Test
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
   double kval_block = 1.0; // W/mK
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
   args.AddOption(&kval_block, "-kb", "--k-block",
                  "Thermal conductivity of the block (W/mK).");
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

   Mesh *serial_mesh = new Mesh("multidomain-hex.mesh");
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

   // Create the sub-domains for the cylinder and the outer block
   Array<int> cylinder_domain_attributes(1);
   cylinder_domain_attributes[0] = 1;

   Array<int> block_domain_attributes(1);
   block_domain_attributes[0] = 2;

   auto cylinder_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, cylinder_domain_attributes));

   auto block_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, block_domain_attributes));

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Set up coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   auto Id = new IdentityMatrixCoefficient(sdim);

   Array<int> attr_cyl(0), attr_block(0);
   attr_cyl.Append(1), attr_block.Append(2);

   Array<int> block_bdry_attributes(block_submesh->bdr_attributes.Max());
   block_bdry_attributes = 1;
   block_bdry_attributes[5] = 0;
   block_bdry_attributes[7] = 0;

   double cval_cyl, rhoval_cyl;
   cval_cyl = 1.0;   // J/kgK
   rhoval_cyl = 1.0; // kg/m^3

   double cval_block, rhoval_block;
   cval_block = 1.0;   // J/kgK
   rhoval_block = 1.0; // kg/m^3

   // Conductivity
   Array<MatrixCoefficient *> coefs_k_cyl(0);
   coefs_k_cyl.Append(new ScalarMatrixProductCoefficient(kval_cyl, *Id));
   PWMatrixCoefficient *Kappa_cyl = new PWMatrixCoefficient(sdim, attr_cyl, coefs_k_cyl);
   ScalarMatrixProductCoefficient *Kappa_cylinder_bdr = new ScalarMatrixProductCoefficient(kval_cyl, *Id);

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

   // Velocity profile for advectio term in cylinder α∇•(q T)
   VectorCoefficient *q = new VectorFunctionCoefficient(sdim, velocity_profile);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create BC Handler (not populated yet)
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool bc_verbose = true;

   heat::BCHandler *bcs_cyl = new heat::BCHandler(cylinder_submesh, bc_verbose); // Boundary conditions handler for cylinder
   heat::BCHandler *bcs_block = new heat::BCHandler(block_submesh, bc_verbose);  // Boundary conditions handler for block

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Create the Heat Solver
   ///////////////////////////////////////////////////////////////////////////////////////////////

   bool solv_verbose = false;
   heat::HeatSolver Heat_Cylinder(cylinder_submesh, order, bcs_cyl, Kappa_cyl, c_cyl, rho_cyl, alpha, q, reaction, ode_solver_type, solv_verbose);
   heat::HeatSolver Heat_Block(block_submesh, order, bcs_block, Kappa_block, c_block, rho_block, ode_solver_type, solv_verbose);

   ParGridFunction *temperature_cylinder_gf = Heat_Cylinder.GetTemperatureGfPtr();
   ParGridFunction *temperature_block_gf = Heat_Block.GetTemperatureGfPtr();
   ParGridFunction *k_grad_temperature_wall_gf = new ParGridFunction(Heat_Block.GetVectorFESpace());
   ParGridFunction *temperature_wall_cylinder_gf = new ParGridFunction(Heat_Cylinder.GetFESpace());
   ParGridFunction *velocity_gf = new ParGridFunction(Heat_Cylinder.GetVectorFESpace());

   ParGridFunction *temperature_wall_block_prev_gf = new ParGridFunction(Heat_Block.GetFESpace());
   GridFunctionCoefficient temperature_wall_block_prev_coeff(temperature_wall_block_prev_gf);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Populate BC Handler
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Outer Box:
   // - T = T_out on the outside wall
   // - k ∇T•n = k ∇T_cyl•n on the inside (cylinder) wall and top/bottom surfaces

   Array<int> block_outer_wall_attributes(block_submesh->bdr_attributes.Max());
   block_outer_wall_attributes = 0;
   block_outer_wall_attributes[0] = 1;
   block_outer_wall_attributes[1] = 1;
   block_outer_wall_attributes[2] = 1;
   block_outer_wall_attributes[3] = 1;

   Array<int> block_inner_wall_attributes(block_submesh->bdr_attributes.Max());
   block_inner_wall_attributes = 0;
   block_inner_wall_attributes[8] = 1;

   double T_out = 1.0;
   VectorGridFunctionCoefficient *k_grad_temperature_wall_coeff = new VectorGridFunctionCoefficient(k_grad_temperature_wall_gf);

   bcs_block->AddNeumannVectorBC(k_grad_temperature_wall_coeff, block_inner_wall_attributes);

   if (test == 0)
   {
      bcs_block->AddDirichletBC(T_out, block_outer_wall_attributes);
   }

   // Cylinder:
   // - T = T_wall on the cylinder wall (obtained from heat equation on box)
   // - k ∇T•n = 0 else

   Array<int> inner_cylinder_wall_attributes(cylinder_submesh->bdr_attributes.Max()); // CHECK
   inner_cylinder_wall_attributes = 0;
   inner_cylinder_wall_attributes[8] = 1;

   GridFunctionCoefficient *T_wall = new GridFunctionCoefficient(temperature_wall_cylinder_gf);
   bcs_cyl->AddDirichletBC(T_wall, inner_cylinder_wall_attributes);

   int cyl_domain_attr = 1;

   if (test == 1)
   {
      Heat_Cylinder.AddVolumetricTerm(HeatingSphere, cyl_domain_attr);
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Setup interface transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create the submesh transfer map for the temperature transfer block -> cylinder
   auto temperature_block_to_cylinder_map = ParSubMesh::CreateTransferMap(*temperature_block_gf, *temperature_wall_cylinder_gf);

   // Setup GSLIB for gradient transfer cylinder -> block
   // 1. Find points on the destination mesh (box)
   std::vector<int> block_element_idx;
   Vector block_element_coords;
   ecm2_utils::ComputeBdrQuadraturePointsCoords(block_inner_wall_attributes, *Heat_Block.GetFESpace(), block_element_idx, block_element_coords);

   // 2. Setup GSLIB finder on the source mesh (cylinder)
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(*cylinder_submesh);
   finder.FindPoints(block_element_coords, Ordering::byVDIM);

   // 3. Compute QoI (gradient of the temperature field) on the source mesh (cylinder)
   int qoi_size_on_qp = sdim;
   Vector qoi_src, qoi_dst; // QoI vector, used to store qoi_src in qoi_func and in call to GSLIB interpolator
   auto qoi_func = [&](ElementTransformation &Tr, int pt_idx, const IntegrationPoint &ip)
   {
      DenseMatrix Kmat(sdim);
      Vector gradloc(sdim);
      Vector kgradloc(qoi_src.GetData() + pt_idx * sdim, sdim); // ref to qoi_src

      Kappa_cylinder_bdr->Eval(Kmat, Tr, ip);

      temperature_cylinder_gf->GetGradient(Tr, gradloc);
      Kmat.Mult(gradloc, kgradloc);
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Heat_Cylinder.EnablePA(pa);
   Heat_Cylinder.Setup();

   Heat_Block.EnablePA(pa);
   Heat_Block.Setup();

   // Setup ouput
   ParaViewDataCollection paraview_dc_cylinder("Heat-Cylinder", cylinder_submesh.get());
   ParaViewDataCollection paraview_dc_block("Heat-Block", block_submesh.get());
   if (paraview)
   {
      paraview_dc_cylinder.SetPrefixPath(outfolder);
      paraview_dc_cylinder.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_cylinder.SetCompressionLevel(9);
      Heat_Cylinder.RegisterParaviewFields(paraview_dc_cylinder);
      Heat_Cylinder.AddParaviewField("velocity", velocity_gf);

      paraview_dc_block.SetPrefixPath(outfolder);
      paraview_dc_block.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_block.SetCompressionLevel(9);
      Heat_Block.RegisterParaviewFields(paraview_dc_block);
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Perform time-integration (looping over the time iterations, step, with a
   //     time-step dt).
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ConstantCoefficient T0(0.0);
   temperature_block_gf->ProjectCoefficient(T0);
   Heat_Block.SetInitialTemperature(*temperature_block_gf);

   temperature_cylinder_gf->ProjectCoefficient(T0);
   Heat_Cylinder.SetInitialTemperature(*temperature_cylinder_gf);

   velocity_gf->ProjectCoefficient(*q);

   real_t t = 0.0;
   bool last_step = false;

   // Write fields to disk for VisIt
   if (paraview)
   {
      Heat_Block.WriteFields(0, t);
      Heat_Cylinder.WriteFields(0, t);
   }

   bool converged = false;
   double tol = 1.0e-4;
   int max_iter = 100;

   Vector temperature_block(temperature_block_gf->Size());
   Vector temperature_block_prev(temperature_block_gf->Size());
   Vector temperature_block_tn(*temperature_block_gf->GetTrueDofs());
   Vector temperature_cylinder_tn(*temperature_cylinder_gf->GetTrueDofs());

   int cyl_dofs = Heat_Cylinder.GetProblemSize();
   int block_dofs = Heat_Block.GetProblemSize();

   if (Mpi::Root())
   {
      out << " Cylinder dofs: " << cyl_dofs << std::endl;
      out << " Block dofs: " << block_dofs << std::endl;
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

      temperature_block_tn = *temperature_block_gf->GetTrueDofs();
      temperature_cylinder_tn = *temperature_cylinder_gf->GetTrueDofs();

      // Inner loop for the segregated solve
      int iter = 0;
      double norm_diff = 2 * tol;
      // double dt_iter = dt / 10;
      // double t_iter = 0;
      while (!converged && iter <= max_iter)
      {
         // Store the previous temperature on the box wall to check convergence
         temperature_block_prev = *(temperature_block_gf->GetTrueDofs());
         temperature_wall_block_prev_gf->SetFromTrueDofs(temperature_block_prev);

         // Transfer k ∇T_wall from cylinder -> block
         {
            ecm2_utils::GSLIBInterpolate(finder, *Heat_Cylinder.GetFESpace(), qoi_func, qoi_src, qoi_dst, qoi_size_on_qp);
            ecm2_utils::TransferQoIToDest(block_element_idx, *Heat_Block.GetVectorFESpace(), qoi_dst, *k_grad_temperature_wall_gf);
         }

         // Advance the diffusion equation on the outer block to the next time step
         temperature_block_gf->SetFromTrueDofs(temperature_block_tn);
         Heat_Block.Step(t, dt, step, false);
         temperature_block_gf->GetTrueDofs(temperature_block);
         t -= dt; // Reset t to same time step, since t is incremented in the Step function

         // Relaxation
         // T_wall(j+1) = ω * T_block,j+1 + (1 - ω) * T_block,j
         if (iter > 0)
         {
            temperature_block *= omega;
            temperature_block.Add(1 - omega, temperature_block_prev);
            temperature_block_gf->SetFromTrueDofs(temperature_block);
         }

         // Transfer temperature from block -> cylinder
         temperature_block_to_cylinder_map.Transfer(*temperature_block_gf,
                                                    *temperature_wall_cylinder_gf);

         // Advance the convection-diffusion equation on the outer block to the
         // next time step
         temperature_cylinder_gf->SetFromTrueDofs(temperature_cylinder_tn);
         Heat_Cylinder.Step(t, dt, step, false);
         t -= dt; // Reset t to same time step, since t is incremented in the Step function

         // Check convergence
         //  || T_wall(j+1) - T_wall(j) || < tol
         norm_diff = temperature_block_gf->ComputeL2Error(temperature_wall_block_prev_coeff);
         converged = norm_diff < tol;

         iter++;

         // Output of subiterations   --> TODO: Create a new paraview collection for saving subiterations data (there's no copy constructor for ParaViewDataCollection)
         // t_iter += dt_iter;
         // Heat_Block.WriteFields(iter, t_iter);
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
      Heat_Block.UpdateTimeStepHistory();
      Heat_Cylinder.UpdateTimeStepHistory();

      temperature_block_tn = *temperature_block_gf->GetTrueDofs();
      temperature_cylinder_tn = *temperature_cylinder_gf->GetTrueDofs();

      t += dt;
      converged = false;

      // Output of time steps
      if (paraview && (step % save_freq == 0))
      {
         Heat_Block.WriteFields(step, t);
         Heat_Cylinder.WriteFields(step, t);
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

   finder.FreeData();

   delete temperature_wall_cylinder_gf;

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