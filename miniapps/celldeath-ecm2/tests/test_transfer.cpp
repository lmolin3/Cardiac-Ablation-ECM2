//
// Sample runs:
// 1. Transfer temperature from fine to coarse space
//    ./test_transfer -d 2 -e 0 -n 10 -ot 4 -oc 1 -paraview -of ./output 
// 2. Transfer temperature from coarse to fine space
//    ./test_transfer -d 2 -e 0 -n 10 -ot 1 -oc 4 -paraview -of ./output

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static Vector center;
static real_t Tmax = 100.0;
static real_t circle_radius = 0.2;



struct s_MeshContext // mesh
{
   int n = 10;                
   int dim = 2;
   int elem = 0;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
} Mesh_ctx;

// Forward declaration

real_t temperature_function(const Vector &x, real_t t);

int main(int argc, char *argv[])
{
   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////
   Mpi::Init(argc, argv);
   Hypre::Init();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options.
   ///////////////////////////////////////////////////////////////////////////////////////////////

   int order_original = 1;
   int order_transferred = 1;
   bool paraview = true;
   string outfolder = "./Output/";

   OptionsParser args(argc, argv);
    args.AddOption(&Mesh_ctx.dim,
                   "-d",
                   "--dimension",
                   "Dimension of the problem (2 = 2d, 3 = 3d)");
    args.AddOption(&Mesh_ctx.elem,
                   "-e",
                   "--element-type",
                   "Type of elements used (0: Quad/Hex, 1: Tri/Tet)");
    args.AddOption(&Mesh_ctx.n,
                   "-n",
                   "--num-elements",
                   "Number of elements in uniform mesh.");
   args.AddOption(&Mesh_ctx.ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&Mesh_ctx.par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order_original, "-ot", "--order-temperature", "Finite element polynomial degree for temperature");
   args.AddOption(&order_transferred, "-oc", "--order-celldeath", "Finite element polynomial degree for cell-death");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable Paraview visualization.");
   args.AddOption(&outfolder, "-of", "--out-folder", "Output folder.");
   args.AddOption(&Tmax, "-Tmax", "--max-temperature", "Maximum temperature.");
   args.ParseCheck();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////
   Mesh serial_mesh;
   switch (Mesh_ctx.dim)
   {
   case 2: // 2d
{      Element::Type type = (Mesh_ctx.elem == 0) ? Element::QUADRILATERAL: Element::TRIANGLE;
      serial_mesh = Mesh::MakeCartesian2D(Mesh_ctx.n,Mesh_ctx.n,type,true);
      break;}
   case 3: // 3d
{      Element::Type type = (Mesh_ctx.elem == 0) ? Element::HEXAHEDRON: Element::TETRAHEDRON;
      serial_mesh = Mesh::MakeCartesian3D(Mesh_ctx.n,Mesh_ctx.n,Mesh_ctx.n,type,true);
      break;}
      default:
      MFEM_ABORT("Unsupported dimension");
   }
   serial_mesh.EnsureNodes();

   for (int l = 0; l < Mesh_ctx.ser_ref_levels; l++)
   {
      serial_mesh.UniformRefinement();
   }


   auto mesh = std::make_shared<ParMesh>(ParMesh(MPI_COMM_WORLD, serial_mesh));
   serial_mesh.Clear(); // the serial mesh is no longer needed
   for (int l = 0; l < Mesh_ctx.par_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   if (mesh->Dimension() == 2)
   {
      center.SetSize(2);
      center(0) = 0.5;
      center(1) = 0.5;
   }
   else if (mesh->Dimension() == 3)
   {
      center.SetSize(3);
      center(0) = 0.5;
      center(1) = 0.5;
      center(2) = 0.5;
   }


   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Set up grid functions and transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Original space   
   H1_FECollection fec_original(order_original, mesh->Dimension());
   ParFiniteElementSpace fes_original(mesh.get(), &fec_original);

   // Transferred space
   H1_FECollection fec_transferred(order_transferred, mesh->Dimension());
   ParFiniteElementSpace fes_transferred(mesh.get(), &fec_transferred);
   
   // Create the TransferOperator
   TrueTransferOperator *transferOp = new TrueTransferOperator(fes_original, fes_transferred);

   // Define grid functions
   ParGridFunction *T_gf_original = new ParGridFunction(&fes_original);
   ParGridFunction *T_gf_transferred = new ParGridFunction(&fes_transferred);

   int original_truevsize = fes_original.GetTrueVSize();
   int transferred_truevsize = fes_transferred.GetTrueVSize();

   int global_truevsize = fes_original.GlobalTrueVSize();
   int global_transferred_truevsize = fes_transferred.GlobalTrueVSize();

   if (Mpi::Root())
   {
      mfem::out << "Original dofs: " << global_truevsize << std::endl;
      mfem::out << "Transferred dofs: " << global_transferred_truevsize << std::endl;
   }

   Vector T_original, T_transferred;
   T_original.SetSize(original_truevsize);   T_original = 0.0;
   T_transferred.SetSize(transferred_truevsize); T_transferred = 0.0;

   // Initialize Paraview visualization
   ParaViewDataCollection paraview_dc_original("Results-Original", mesh.get());
   paraview_dc_original.SetPrefixPath(outfolder);  
   paraview_dc_original.SetDataFormat(VTKFormat::BINARY);
   if (order_original > 1)
   {
      paraview_dc_original.SetHighOrderOutput(true);
      paraview_dc_original.SetLevelsOfDetail(order_original);
   }
   paraview_dc_original.RegisterField("Temperature", T_gf_original);

   ParaViewDataCollection paraview_dc_transferred("Results-Transferred", mesh.get());
   paraview_dc_transferred.SetPrefixPath(outfolder);
   paraview_dc_transferred.SetDataFormat(VTKFormat::BINARY);
   if (order_transferred > 1)
   {
      paraview_dc_transferred.SetHighOrderOutput(true);
      paraview_dc_transferred.SetLevelsOfDetail(order_transferred);
   }
   paraview_dc_transferred.RegisterField("Temperature", T_gf_transferred);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Define analytic temperature profile 
   FunctionCoefficient T0(temperature_function);
   T_gf_original->ProjectCoefficient(T0);
   T_gf_original->GetTrueDofs(T_original);

   // Export before transfer
   if (paraview)
   {
      paraview_dc_original.SetCycle(0); paraview_dc_original.SetTime(0.0); paraview_dc_original.Save();
      paraview_dc_transferred.SetCycle(0); paraview_dc_transferred.SetTime(0.0); paraview_dc_transferred.Save();
   }


   /// Transfer
   if ( order_original == order_transferred )
   {
      T_transferred = T_original;
   }
   else
   {  // Coarse to fine prolongation
      transferOp->Mult(T_original, T_transferred);
   }

   T_gf_transferred->SetFromTrueDofs(T_transferred);

   // Export after transfer
   if (paraview)
   {
      paraview_dc_original.SetCycle(1); paraview_dc_original.SetTime(0.1); paraview_dc_original.Save();
      paraview_dc_transferred.SetCycle(1); paraview_dc_transferred.SetTime(0.1); paraview_dc_transferred.Save();
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   delete T_gf_original;
   delete T_gf_transferred;
   delete transferOp;

   return 0;
}


real_t temperature_function(const Vector &x, real_t t)
{
   // Define the temperature profile
   // T = 1 / (1 + r^2) + t, where r is the distance from the center
   real_t xc = x(0) - center(0);
   real_t yc = x(1) - center(1);
   real_t r_squared =( xc * xc + yc * yc ) - circle_radius * circle_radius;

   return Tmax * std::exp(-25*r_squared) + t;
}