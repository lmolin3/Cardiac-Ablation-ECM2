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

#include <mfem.hpp>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{

   /// 1. Initialize MPI and Hypre
   Mpi::Init(argc, argv);
   Hypre::Init();

   StopWatch sw;
   sw.Start();

   /// 2. Parse command-line options.
   const char *mesh_file = "../multidomain/multidomain-hex.mesh";
   int order = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.ParseCheck();

   /// 3. Read Mesh and create parallel
   
   Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
   for (int l = 0; l < serial_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }

   auto pmesh = make_shared<ParMesh>(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = parallel_ref_levels;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }
   // Make sure tet-only meshes are marked for local refinement.
   pmesh->Finalize(true);

   /// Create spaces and vectors (like in the solver)
   const int dim = pmesh->Dimension();

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1 for potential and ND for the electric field.
   H1_FECollection *H1_fec = new H1_FECollection(order, dim);

   ParFiniteElementSpace *H1_fes = new ParFiniteElementSpace(pmesh.get(), H1_fec);

   // Build grid functions
   ParGridFunction *phi = new ParGridFunction(H1_fes);
   *phi = 0.0;
   
   // Initialize vector/s
   Vector B;
   B.UseDevice(true);
   B.SetSize(H1_fes->GetTrueVSize());

   sw.Stop();
   double rt = sw.RealTime();

   // Global time 
   double global_rt;
   MPI_Reduce(&rt, &global_rt, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   if (Mpi::Root())
   {
      cout << "Time " << global_rt << " s." << endl;
   }

   return 0;
}

