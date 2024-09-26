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
//                 rho c dT/dt = κΔT          in outer box
//                       k∇T•n = k∇T_wall•n   on cylinder interface
//                       k∇T•n = 0            elsewhere
//
// with temperature T and coefficient κ (non-physical in this example).
//
// A heat equation equation is described inside the cylinder domain
//
//                 rho c dT/dt = κΔT          in cylinder
//                           T = T_wall       on cylinder interface
//                       k∇T•n = 0            elsewhere
//                           Q = Qval         inside sphere
//
// with temperature T, coefficients κ, α and prescribed velocity profile b.
//
// To couple the solutions of both equations, a segregated solve with two way
// coupling approach is used to solve the timestep tn tn+dt:
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
//      mpirun -np 4 ./multidomain-neumann-dirichlet-ecm2 -o 3 -tf 1e0 -dt 1.0e-2 -Q 1e6 -kc 1e0 -kb 1e0 --relaxation-parameter 0.8 --paraview -of ./Output

#include "mfem.hpp"
#include "lib/heat_solver.hpp"

#include <fstream>
#include <iostream>
#include <memory>

using namespace mfem;

IdentityMatrixCoefficient *Id = NULL;

constexpr double Sphere_Radius = 0.1;
double Qval = 1e5; // W/m^3
double HeatingSphere(const Vector &x, double t);

// Forward declaration
void TransferQoIToDest(const std::vector<int> &elem_idx, const ParFiniteElementSpace &fes_grad, const Vector &grad_vec, ParGridFunction &grad_gf, MatrixCoefficient *K);
void print_matrix(const DenseMatrix &A);

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
   // Mesh
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   // Time integrator
   int ode_solver_type = 1;
   real_t t_final = 5.0;
   real_t dt = 1.0e-5;
   // Domain decomposition
   real_t omega = 0.5; // Relaxation parameter
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
   args.AddOption(&omega, "-omega", "--relaxation-parameter",
                  "Relaxation parameter.");
   args.AddOption(&Qval, "-Q", "--volumetric-heat-source",
                  "Volumetric heat source (W/m^3).");
   args.AddOption(&kval_cyl, "-kc", "--k-cylinder",
                  "Thermal conductivity of the cylinder (W/mK).");
   args.AddOption(&kval_block, "-kb", "--k-block",
                  "Thermal conductivity of the block (W/mK).");
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

   Array<MatrixCoefficient *> coefs_k_block(0);
   coefs_k_block.Append(new ScalarMatrixProductCoefficient(kval_block, *Id));
   PWMatrixCoefficient *Kappa_block = new PWMatrixCoefficient(sdim, attr_block, coefs_k_block);
   ScalarMatrixProductCoefficient *Kappa_block_bdr = new ScalarMatrixProductCoefficient(kval_block, *Id);

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

   verbose = false;
   heat::HeatSolver Heat_Cylinder(cylinder_submesh, order, bcs_cyl, Kappa_cyl, c_cyl, rho_cyl, ode_solver_type, verbose);
   heat::HeatSolver Heat_Block(block_submesh, order, bcs_block, Kappa_block, c_block, rho_block, ode_solver_type, verbose);

   H1_ParFESpace *GradH1FESpace_block = new H1_ParFESpace(block_submesh.get(), order, block_submesh->Dimension(), BasisType::GaussLobatto, block_submesh->Dimension());
   H1_ParFESpace *GradH1FESpace_cylinder = new H1_ParFESpace(cylinder_submesh.get(), order, cylinder_submesh->Dimension(), BasisType::GaussLobatto, cylinder_submesh->Dimension());

   ParGridFunction *temperature_cylinder_gf = Heat_Cylinder.GetTemperatureGfPtr();
   ParGridFunction *temperature_block_gf = Heat_Block.GetTemperatureGfPtr();
   ParGridFunction *k_grad_temperature_wall_gf = new ParGridFunction(GradH1FESpace_block);
   ParGridFunction *temperature_wall_cylinder_gf = new ParGridFunction(Heat_Cylinder.GetFESpace());

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

   // Cylinder:
   // - T = T_wall on the cylinder wall (obtained from heat equation on box)
   // - k ∇T•n = 0 else

   Array<int> inner_cylinder_wall_attributes(cylinder_submesh->bdr_attributes.Max()); // CHECK
   inner_cylinder_wall_attributes = 0;
   inner_cylinder_wall_attributes[8] = 1;

   GridFunctionCoefficient *T_wall = new GridFunctionCoefficient(temperature_wall_cylinder_gf);
   bcs_cyl->AddDirichletBC(T_wall, inner_cylinder_wall_attributes);
   int cyl_domain_attr = 1;
   Heat_Cylinder.AddVolumetricTerm(HeatingSphere, cyl_domain_attr);

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
   auto qoi_func = [&](ElementTransformation &Tr, int pt_idx, int num_pts)
   {
      Vector gradloc(qoi_src.GetData() + pt_idx * sdim, sdim);
      temperature_cylinder_gf->GetGradient(Tr, gradloc);
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

      paraview_dc_block.SetPrefixPath(outfolder);
      paraview_dc_block.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_block.SetCompressionLevel(9);
      Heat_Block.RegisterParaviewFields(paraview_dc_block);
      Heat_Block.AddParaviewField("K-Grad-Temperature", k_grad_temperature_wall_gf);
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
   int max_iter = 200;

   Vector temperature_block(temperature_block_gf->Size());
   Vector temperature_block_prev(temperature_block_gf->Size());
   Vector temperature_block_tn(*temperature_block_gf->GetTrueDofs());
   Vector temperature_cylinder_tn(*temperature_cylinder_gf->GetTrueDofs());

   if (Mpi::Root())
   {
      out << "----------------------------------------------------------------------------------------"
          << std::endl;
      out << std::left << std::setw(16) << "Step" << std::setw(16) << "Time" << std::setw(16) << "dt" << std::setw(16) << "Sub-iterations" << std::endl;
      out << "----------------------------------------------------------------------------------------"
          << std::endl;
   }

   // Outer loop for time integration
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
            TransferQoIToDest(block_element_idx, *GradH1FESpace_block, qoi_dst, *k_grad_temperature_wall_gf, Kappa_block_bdr);
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
         // Heat_Block.WriteFields(iter, t);
         // Heat_Cylinder.WriteFields(iter, t);
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

double HeatingSphere(const Vector &x, double t)
{
   double Q = 0.0;
   double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
   Q = r < Sphere_Radius / 4.0 ? Qval : 0.0; // W/m^2

   return Q;
}