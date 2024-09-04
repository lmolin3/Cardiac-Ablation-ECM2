//
// Export mesh boundaries
//

#include "mfem.hpp"

// Include for mkdir
#include <sys/stat.h>

#ifdef M_PI
#define PI M_PI
#else
#define M_PI 3.14159265358979
#endif


using namespace mfem;

// Test
int main(int argc, char *argv[])
{
   //
   /// 1. Initialize MPI and HYPRE.
   //
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   Hypre::Init();


   //
   /// 2. Parse command-line options. 
   //
   int n = 10;                // custom mesh
   int dim = 2;
   int elem = 0;

   int ser_ref_levels = 0;    // mesh refinement
   int par_ref_levels = 0;
   const char *mesh_file = "";

   const char *folderPath = "./";

   // TODO: check parsing and assign variables
   mfem::OptionsParser args(argc, argv);
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
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0)
   {
       args.PrintOptions(std::cout);
   }


   //
   /// 3. Read the (serial) mesh from the given mesh file on all processors.
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
            type = (dim == 2) ? Element::QUADRILATERAL: Element::HEXAHEDRON;
            break;
         case 1: // tri
            type = (dim == 2) ? Element::TRIANGLE: Element::TETRAHEDRON;
            break;
      }

      switch (dim)
      {
         case 2: // 2d
            mesh = Mesh::MakeCartesian2D(n,n,type,true);	
            break;
         case 3: // 3d
            mesh = Mesh::MakeCartesian3D(n,n,n,type,true);	
            break;
      }
   }

   // Serial Refinement
   for (int l = 0; l < ser_ref_levels; l++)
   {
       mesh.UniformRefinement();
   }


   //
   /// 4. Define a parallel mesh by a partitioning of the serial mesh. 
   // Refine this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   //
   ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh->UniformRefinement();
       }
   }

   
   //
   /// 5 Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol_*".

   std::ostringstream mesh_name, v_name, p_name;
   mesh_name << folderPath << "/mesh." << std::setfill('0') << std::setw(6) << myrank; 

   v_name << folderPath << "/sol_v." << std::setfill('0') << std::setw(6) << myrank;
   p_name << folderPath << "/sol_p." << std::setfill('0') << std::setw(6) << myrank;

   std::ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   pmesh->Print(mesh_ofs);

   std::ostringstream omesh_file, omesh_file_bdr;
   omesh_file << folderPath << "/mesh.vtk";
   omesh_file_bdr << folderPath << "/mesh_bdr.vtu";
   std::ofstream omesh(omesh_file.str().c_str());
   omesh.precision(14);
   pmesh->PrintVTK(omesh);
   pmesh->PrintBdrVTU(omesh_file_bdr.str());


   // Free used memory.
   delete pmesh; pmesh = nullptr;
   
   // Finalize Hypre and MPI
   HYPRE_Finalize();
   MPI_Finalize();

   return 0;
}