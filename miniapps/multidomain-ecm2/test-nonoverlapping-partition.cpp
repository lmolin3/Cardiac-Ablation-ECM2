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
//             mpirun -np 10 ./test-nonoverlapping-partition -np1 3 -np2 2 -np3 7

#include "mfem.hpp"

#include <fstream>
#include <sstream>
#include <sys/stat.h> // Include for mkdir
#include <iostream>
#include <memory>

using namespace mfem;

static constexpr real_t Tfluid = 303.15;    // Fluid temperature
static constexpr real_t Tcylinder = 293.15; // Cylinder temperature

void ExportMeshwithPartitioning(const std::string &outfolder, const std::string &filename, Mesh &mesh, const int *partitioning_, const int *world_ranks);

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

   std::vector<std::string> domain_names = {"Solid", "Fluid", "Cylinder"};

   // Communicator
   int nmeshes = domain_names.size();
   Array<int> np_list(3);
   // FE
   int order = 1;
   // Mesh
   Array<int> serial_ref_levels(3); serial_ref_levels = 0;
   Array<int> parallel_ref_levels(3); parallel_ref_levels = 0;
   // Postprocessing
   bool paraview = true;
   const char *outfolder = "";

   OptionsParser args(argc, argv);
   // Communicator
   args.AddOption(&np_list[0], "-np1", "--np1",
                  "number of MPI ranks for solid mesh");
   args.AddOption(&np_list[1], "-np2", "--np2",
                  "number of MPI ranks for fluid mesh");
   args.AddOption(&np_list[2], "-np3", "--np3",
                  "number of MPI ranks for cylinder mesh");
   // FE
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   // Mesh
   args.AddOption(&serial_ref_levels[0], "-rs1", "--serial-ref-levels",
                  "Number of serial refinement levels for solid mesh.");
   args.AddOption(&serial_ref_levels[1], "-rs2", "--serial-ref-levels",
                  "Number of serial refinement levels for fluid mesh.");
   args.AddOption(&serial_ref_levels[2], "-rs3", "--serial-ref-levels",
                  "Number of serial refinement levels for cylinder mesh.");
   args.AddOption(&parallel_ref_levels[0], "-rp1", "--parallel-ref-levels",
                  "Number of parallel refinement levels for solid mesh.");
   args.AddOption(&parallel_ref_levels[1], "-rp2", "--parallel-ref-levels",   
                  "Number of parallel refinement levels for fluid mesh."); 
   args.AddOption(&parallel_ref_levels[2], "-rp3", "--parallel-ref-levels",
                  "Number of parallel refinement levels for cylinder mesh.");
   // Postprocessing
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&outfolder, "-of", "--out-folder",
                  "Output folder.");

   args.ParseCheck();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // 1. Setup and split MPI communicator
   int color = 0;
   int npsum = 0;
   Array<int> color_count(3); color_count = 0;
   for (int i = 0; i < nmeshes; i++)
   {
      npsum += np_list[i];
      if (myid < npsum) { color = i; break; }
   }
   
   color_count[color]++;

   Array<int> global_color_count(3);
   global_color_count = 0;
   MPI_Reduce(color_count.GetData(), global_color_count.GetData(), 3, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   // Perform the check on the root process
   if (Mpi::Root())
   {
      for (int i = 0; i < nmeshes; i++)
      {
         if (global_color_count[i] == 0)
         {
            MFEM_ABORT("Mesh " + std::string(domain_names[i]) + " has no processes assigned. Aborting.");
         }
      }
   }

   MPI_Comm *comml = new MPI_Comm;
   MPI_Comm_split(MPI_COMM_WORLD, color, myid, comml);
   int myidlocal, numproclocal;
   MPI_Comm_rank(*comml, &myidlocal);
   MPI_Comm_size(*comml, &numproclocal);

   if (Mpi::Root())
      mfem::out << "\033[34m\nMPI Communicator Setup\033[0m" << std::endl;
   MPI_Barrier(MPI_COMM_WORLD);
   printf("\033[0mGLOBAL RANK/SIZE: %d/%d \t LOCAL RANK/SIZE: %d/%d \t MESH: %s (color %d)\n",
          myid, num_procs, myidlocal, numproclocal, domain_names[color].c_str(),color);


   MPI_Barrier(MPI_COMM_WORLD);
   if (numproclocal < np_list[color])
   {
      if(myidlocal == 0)
         mfem::out << "\033[1;33mWarning: Mesh " << domain_names[color] << " has fewer ranks (" << numproclocal << ") than expected (" << np_list[color] << ").\033[0m" << std::endl;
   }

   MPI_Barrier(MPI_COMM_WORLD);

   // 2. Load serial mesh
   if (Mpi::Root())
      mfem::out << "\033[34mLoading serial mesh... \033[0m";

   Mesh *serial_mesh = new Mesh("../../data/three-domains.msh");
   int sdim = serial_mesh->SpaceDimension();

   for (int l = 0; l < serial_ref_levels[color]; l++)
   {
      serial_mesh->UniformRefinement();
   }

   serial_mesh->EnsureNodes();

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   
   int *world_ranks = new int[numproclocal];
   MPI_Allgather(&myid, 1, MPI_INT, world_ranks, 1, MPI_INT, *comml);

   
   // 3. Generate serial submeshes
   if (Mpi::Root())
   {
      mfem::out << "\033[34mCreating sub-meshes... \033[0m";
   }

   // Create the sub-domains for the cylinder, solid and fluid domains
   AttributeSets &attr_sets = serial_mesh->attribute_sets;

   Array<int> domain_attribute = attr_sets.GetAttributeSet(domain_names[color]);
   auto submesh = SubMesh::CreateFromDomain(*serial_mesh, domain_attribute);


   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;


   // 4. Generate partitioning and create parallel mesh
      // Generate mesh partitioning
   MPI_Barrier(MPI_COMM_WORLD);
   if (Mpi::Root())
      mfem::out << "\033[34mGenerating partitioning and creating parallel mesh... \033[0m";

   // Partition type:
   // 0) METIS_PartGraphRecursive (sorted neighbor lists)
   // 1) METIS_PartGraphKway      (sorted neighbor lists) (default)
   // 2) METIS_PartGraphVKway     (sorted neighbor lists)
   // 3) METIS_PartGraphRecursive
   // 4) METIS_PartGraphKway
   // 5) METIS_PartGraphVKway
   int partition_type = 1;
   int *partitioning = submesh.GeneratePartitioning(numproclocal, partition_type);

   // Create parallel mesh
   ParMesh parent_mesh = ParMesh(*comml, submesh, partitioning);

   ExportMeshwithPartitioning(outfolder, domain_names[color], submesh, partitioning, world_ranks); 
   //ExportMeshwithPartitioning(outfolder, *serial_mesh, partitioning);
   delete[] partitioning;
   delete serial_mesh;
   
   for (int l = 0; l < parallel_ref_levels[color]; l++)
   {
      parent_mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      mfem::out << "\033[34mdone.\033[0m" << std::endl;
   }

   return 0;
}

void ExportMeshwithPartitioning(const std::string &outfolder, const std::string &filename, Mesh &mesh, const int *partitioning_, const int *world_ranks)
{
   // Extract the partitioning
   Array<int> partitioning;
   partitioning.MakeRef(const_cast<int *>(partitioning_), mesh.GetNE(), false);

   // Assign partitioning to the mesh
   FiniteElementCollection *attr_fec = new L2_FECollection(0, mesh.Dimension());
   FiniteElementSpace *attr_fespace = new FiniteElementSpace(&mesh, attr_fec);
   GridFunction attr(attr_fespace);
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      attr(i) = world_ranks[partitioning[i]] + 1;
   }

   // Create paraview datacollection
   ParaViewDataCollection paraview_dc(filename, &mesh);
   paraview_dc.SetPrefixPath(outfolder);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetCompressionLevel(9);
   paraview_dc.RegisterField("partitioning", &attr);
   paraview_dc.Save();

   delete attr_fespace;
   delete attr_fec;
}