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
//            -------------------------------
//            Mesh Boundary/Domain idexporter
//            -------------------------------
//
// Sample runs:

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <limits>
#include <cstdlib>
#include "FilesystemHelper.hpp"  

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   //
   /// 1. Parse command-line options.
   //
   int n = 10; // custom mesh
   int dim = 2;
   int elem = 0;

   int ser_ref_levels = 0; // mesh refinement
   int par_ref_levels = 0;
   const char *mesh_file = "";

   const char *folderPath = "./";

   // TODO: check parsing and assign variables
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file,
                  "-m",
                  "--mesh",
                  "Mesh file to use.");
   args.AddOption(&dim,
                  "-d",
                  "--dimension",
                  "Dimension of the problem (2 = 2d, 3 = 3d)");
   args.AddOption(&elem,
                  "-e",
                  "--element-type",
                  "Type of elements used (0: Quad/Hex, 1: Tri/Tet)");
   args.AddOption(&n,
                  "-n",
                  "--num-elements",
                  "Number of elements in uniform mesh.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&folderPath,
                  "-o",
                  "--output-folder",
                  "Output folder.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
   }

   args.PrintOptions(std::cout);

   //
   /// 2. Read the (serial) mesh from the given mesh file on all processors.
   //

   Mesh mesh;

   if (mesh_file[0] != '\0')
   {
      mesh = Mesh::LoadFromFile(mesh_file);
   }
   else // Create square mesh
   {
      Element::Type type;
      switch (elem)
      {
      case 0: // quad
         type = (dim == 2) ? Element::QUADRILATERAL : Element::HEXAHEDRON;
         break;
      case 1: // tri
         type = (dim == 2) ? Element::TRIANGLE : Element::TETRAHEDRON;
         break;
      }

      switch (dim)
      {
      case 2: // 2d
         mesh = Mesh::MakeCartesian2D(n, n, type, true);
         break;
      case 3: // 3d
         mesh = Mesh::MakeCartesian3D(n, n, n, type, true);
         break;
      }
   }

   // Serial Refinement
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   //
   /// 3. Save the refined mesh 
   //
   if (!fs::is_directory(folderPath) || !fs::exists(folderPath)) { // Check if folder exists
      fs::create_directories(folderPath); // create folder
   }

   std::ostringstream mesh_name;
   std::ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);

   std::ostringstream omesh_file, omesh_file_bdr;
   omesh_file << folderPath << "/mesh.vtk";
   omesh_file_bdr << folderPath << "/mesh_bdr";
   std::fstream omesh(omesh_file.str().c_str(), std::ios::out);
   omesh.precision(14);
   mesh.PrintVTK(omesh);
   mesh.PrintBdrVTU(omesh_file_bdr.str());

   // Free used memory.

   return 0;
}
