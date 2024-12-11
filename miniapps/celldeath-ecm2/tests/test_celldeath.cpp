//                       MFEM Cell-Death-ecm2 Example
//
// Sample runs:  
//
// Sample runs:
//
// 1. 2D Problem with Quad Elements:
//    ./test_celldeath -d 2 -e 0 -n 10 -paraview -of ./output -Tmax 100
//
// 2. 2D Problem with Tri Elements:
//    ./test_celldeath -d 2 -e 1 -n 10 -paraview -of ./output -Tmax 100
//
// 3. 3D Problem with Hex Elements:
//    ./test_celldeath -d 3 -e 0 -n 10 -paraview -of ./output -Tmax 100
//
// 4. 2D Problem with higher order for temperature field:
//    ./test_celldeath -d 2 -e 0 -n 10 -ot 4 -paraview -of ./output -Tmax 100
//
// 5. 2D Problem with same order for temperature and cell-death fields:
//    ./test_celldeath -d 2 -e 0 -n 10 -ot 2 -oc 2 -paraview -of ./output -Tmax 100
//

#include "mfem.hpp"
#include "lib/celldeath_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static Vector center;
static real_t Tmax = 1.0;
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
   int order_temperature = 1;
   int order_celldeath = 1;
   int type = 0;
   int meshDim = 2;
   bool paraview = true;
   string outfolder = "./Output/CellDeath/";

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
   args.AddOption(&order_temperature, "-ot", "--order-temperature", "Finite element polynomial degree for temperature");
   args.AddOption(&order_celldeath, "-oc", "--order-celldeath", "Finite element polynomial degree for cell-death");
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
   /// 4. Set up coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   /*
   ConstantCoefficient *A1 = new ConstantCoefficient(1.0);
   ConstantCoefficient *A2 = new ConstantCoefficient(1.0);
   ConstantCoefficient *A3 = new ConstantCoefficient(1.0);
   ConstantCoefficient *deltaE1 = new ConstantCoefficient(1.0);
   ConstantCoefficient *deltaE2 = new ConstantCoefficient(1.0);
   ConstantCoefficient *deltaE3 = new ConstantCoefficient(1.0);
   */

   real_t A1 = 1.0;
   real_t A2 = 1.0;
   real_t A3 = 1.0;
   real_t deltaE1 = 1.0;
   real_t deltaE2 = 1.0;
   real_t deltaE3 = 1.0;

   // Define analytic temperature profile 
   FunctionCoefficient T0(temperature_function);
   
   H1_FECollection fec(order_temperature, mesh->Dimension());
   ParFiniteElementSpace fespace(mesh.get(), &fec);
   ParGridFunction *T_gf = new ParGridFunction(&fespace);
   T_gf->ProjectCoefficient(T0);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create the CellDeath Solver and DataCollections
   ///////////////////////////////////////////////////////////////////////////////////////////////

   celldeath::CellDeathSolverEigen solver(mesh, order_celldeath, T_gf, A1, A2, A3, deltaE1, deltaE2, deltaE3); 
   celldeath::CellDeathSolverGotran solver_gotran(mesh, order_celldeath, T_gf, A1, A2, A3, deltaE1, deltaE2, deltaE3); 

   // Initialize Paraview visualization
   ParaViewDataCollection paraview_dc("CellDeathEigen", mesh.get());
   ParaViewDataCollection paraview_dc_gotran("CellDeathGotran", mesh.get());

   if (paraview)
   {
      paraview_dc.SetPrefixPath(outfolder);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetCompressionLevel(9);
      solver.RegisterParaviewFields(paraview_dc);
      solver.AddParaviewField("Temperature", T_gf);

      paraview_dc_gotran.SetPrefixPath(outfolder);
      paraview_dc_gotran.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_gotran.SetCompressionLevel(9);
      solver_gotran.RegisterParaviewFields(paraview_dc_gotran);
      solver_gotran.AddParaviewField("Temperature", T_gf);
   }

   // Export initial state
   if (paraview)
   {
      solver.WriteFields(0, 0.0);
      solver_gotran.WriteFields(0, 0.0);
   }

   int temp_truevsize = fespace.GlobalTrueVSize();
   int celldeath_truevsize = solver.GetProblemSize();

   if (Mpi::Root())
   {
      mfem::out << "Temperature dofs: " << temp_truevsize << std::endl;
      mfem::out << "Cell-death dofs: " << celldeath_truevsize << std::endl;
   }


   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Perform time-integration (looping over the time iterations, step, with a time-step dt).
   ///////////////////////////////////////////////////////////////////////////////////////////////

   real_t t = 0.0;
   real_t dt = 1.0e-2;
   real_t t_final = 1.0;
   bool last_step = false;

   // Solving problem with numerical method
   if (Mpi::Root())
      mfem::out << "Solving problem with numerical approximation..." << endl;
   StopWatch chronoGotran;
   chronoGotran.Start();
   for (int step = 1; !last_step; step++)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      // Update temperature profile
      T0.SetTime(t + dt);
      T_gf->ProjectCoefficient(T0);

      // Solve
      int method = 0;
      int substeps = 1;
      solver_gotran.Solve(t, dt, method, substeps);
      t += dt;

      // Output of time steps
      if (paraview)
      {
         solver_gotran.WriteFields(step, t);
      }

   }

   chronoGotran.Stop();
   if (Mpi::Root())
   {
      mfem::out << "Done. time: ";
      mfem::out << chronoGotran.RealTime() << " seconds" << endl;
   }


   // Solving problem with Eigen method
   if (Mpi::Root())
      mfem::out << "Solving problem with Eigen method..." << endl;
   t = 0.0;
   last_step = false;
   StopWatch chronoEigen;
   chronoEigen.Start();
   for (int step = 1; !last_step; step++)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      // Update temperature profile
      T0.SetTime(t + dt);
      T_gf->ProjectCoefficient(T0);

      // Solve
      solver.Solve(t, dt);
      t += dt;

      // Output of time steps
      if (paraview)
      {
         solver.WriteFields(step, t);
      }

   }

   chronoEigen.Stop();
   if (Mpi::Root())
   {
      mfem::out << "Done. time: ";
      mfem::out << chronoEigen.RealTime() << " seconds" << endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   //delete A1;
   //delete A2;
   //delete A3;
   //delete deltaE1;
   //delete deltaE2;
   //delete deltaE3;
   delete T_gf;

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