// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
//            -----------------------------------------------------
//            RF Miniapp:  Simple Electrostatics Simulation Code
//            -----------------------------------------------------
//
// This miniapp assembles a simple 2D electrostatic problem (Quasi-static Maxwell).
//
//                            Div sigma Grad Phi = 0
//
// The test is in 2D and assembles the system for a specific set of boundary conditions to check well-posedness.
//
// Sample runs:
//
//   1. Dirichlet boundary conditions
//      ./test_welposedness -rs 2 -o 2 -bc 0 -of ./Output/TestWellPosedness
//   2. Full Neumann boundary conditions
//     ./test_welposedness -rs 2 -o 2 -bc 1 -of ./Output/TestWellPosedness
//   3. Robin boundary conditions
//     ./test_welposedness -rs 2 -o 2 -bc 2 -of ./Output/TestWellPosedness
//

#include <mfem.hpp>

#include "../lib/electrostatics_solver.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/stat.h> // Include for mkdir

#include "../../common-ecm2/FilesystemHelper.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::electrostatics;


// Boundary Conditions
void SetupBCHandler(BCHandler *bcs, int bc_type, int attr);
int sdim = 2;

int main(int argc, char *argv[])
{
   /// 1. Initialize MPI and Hypre
   Mpi::Init(argc, argv);
   Hypre::Init();


   /// 2. Parse command-line options.
   const char *mesh_file = "../../data/square-disc.mesh";
   int order = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool paraview = false;
   const char *outfolder = "./Output/Test/";
   const char *device_config = "cpu";
   int bc_type = 0; // 0 - Dirichlet, 1 - Neumann, 2 - Robin
   int attr = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&bc_type, "-bc", "--bc-type",
                  "Boundary condition type: 0 - Dirichlet, 1 - Neumann, 2 - Robin.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&outfolder,
                  "-of",
                  "--output-folder",
                  "Output folder.");
   args.ParseCheck();

   //    Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (Mpi::Root())
      device.Print();


   /// 3. Read Mesh and create parallel
   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int sdim = mesh->SpaceDimension();
   
   StopWatch sw_initialization;
   sw_initialization.Start();
   if (Mpi::Root())
   {
      cout << "Starting initialization." << endl;
   }

   // Project a NURBS mesh to a piecewise-quadratic curved mesh
   if (mesh->NURBSext)
   {
      mesh->UniformRefinement();
      if (serial_ref_levels > 0)
      {
         serial_ref_levels--;
      }

      mesh->SetCurvature(2);
   }

   // Ensure that quad and hex meshes are treated as non-conforming.
   mesh->EnsureNCMesh();

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. NURBS meshes are
   // refined at least twice, as they are typically coarse.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   auto pmesh = make_shared<ParMesh>(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = parallel_ref_levels;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }
   // Make sure tet-only meshes are marked for local refinement.
   pmesh->Finalize(true);

   /// 4. Set up conductivity coefficient
   int d = pmesh->Dimension();

   auto sigmaCoeff = new ConstantCoefficient(1.0);

   /// 5. Set up boundary conditions
   bool verbose = true;
   BCHandler *bcs = new BCHandler(pmesh, verbose); // Boundary conditions handler
   SetupBCHandler(bcs, bc_type, attr);

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   /// 6. Create the Electrostatics Solver
   // Create the Electrostatic solver
   if (Mpi::Root())
      mfem::out << "Creating Electrostatics solver..." << std::endl;

   ElectrostaticsSolver RF_solver(pmesh, order, bcs, sigmaCoeff, verbose);

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   if (Mpi::Root())
   {
      if (!fs::is_directory(outfolder) || !fs::exists(outfolder))
      {                                     // Check if folder exists
         fs::create_directories(outfolder); // create folder
      }
   }

   RF_solver.display_banner(std::cout);

   if (Mpi::Root())
      mfem::out << "\nCreating DataCollection...";

   // Initialize Paraview visualization
   ParaViewDataCollection paraview_dc("RF_solver-Parallel", pmesh.get());

   if (paraview)
   {
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetPrefixPath(outfolder);
         paraview_dc.SetLevelsOfDetail(order);
         RF_solver.RegisterParaviewFields(paraview_dc);
   }

   if (Mpi::Root())
      mfem::out << "done." << endl;

   sw_initialization.Stop();
   real_t my_rt[1], rt_max[1];
   my_rt[0] = sw_initialization.RealTime();
   MPI_Reduce(my_rt, rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   if (Mpi::Root())
   {
      cout << "Initialization done.  Elapsed time " << my_rt[0] << " s." << endl;
   }

   /// 7. Solve the problem
   if (Mpi::Root())
   {
      cout << "\nSolving... " << endl;
   }

   // Display the current number of DoFs in each finite element space
   RF_solver.PrintSizes();

   // Setup solver and Assemble all forms
   int pl = 1;
   int prec_type = 1; // 0 - Jacobi, 1 - BoomerAMG
   //RF_solver.EnablePA(pa);
   RF_solver.Setup(prec_type, pl);
   auto A = (RF_solver.GetOperator()).As<HypreParMatrix>();

   // Export matrix  
   std::vector<const char*> bc_names = {"Dirichlet", "Neumann", "Robin"};
   const char* bc_name = bc_names[bc_type];
   std::ofstream A_file(std::string(outfolder) + '/' + "A_" + std::string(bc_name) + ".dat");
   A->PrintMatlab(A_file);

   /// 8. Cleanup
   delete sigmaCoeff;

   return 0;
}



void SetupBCHandler(BCHandler *bcs, int bc_type, int attr)
{
   Vector zero_vec(sdim);
   zero_vec = 0.0;

   real_t alpha_robin = 1e0;

   // Based on the bc_type, set up the appropriate boundary conditions
   switch (bc_type)
   {
      case 0: // Dirichlet
         bcs->AddDirichletBC(new ConstantCoefficient(1.0), attr);
         break;
      case 1: // Neumann
         break;
      case 2: // Robin
         bcs->AddGeneralRobinBC(new ConstantCoefficient(alpha_robin), new ConstantCoefficient(1.0), new ConstantCoefficient(0.0), new VectorConstantCoefficient(zero_vec),new ConstantCoefficient(0.0), attr, true); 
         break; 
      default:
         MFEM_ABORT("Unknown boundary condition type.");
   }
}