//                       MFEM Cell-Death-ecm2 Example
//
// Sample runs:  
//                mpirun -np 4 ./celldeath_ecm2 -of ./Output/CellDeath  
//
// Description: 
// 

#include "mfem.hpp"
#include "lib/celldeath_solver.hpp"
#include "lib/celldeath_solver_gotran.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static Vector center(2);
static real_t Tmax = 1.0;
static real_t circle_radius = 0.2;

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
   string mesh_file = "../../data/square-disc.mesh";
   int order = 1;
   bool paraview = true;
   string outfolder = "./Output/CellDeath/";

   OptionsParser args(argc, argv);
   //args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable Paraview visualization.");
   args.AddOption(&outfolder, "-of", "--out-folder", "Output folder.");
   args.AddOption(&Tmax, "-Tmax", "--max-temperature", "Maximum temperature.");
   args.ParseCheck();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////
   Mesh serial_mesh(mesh_file);

   auto mesh = std::make_shared<ParMesh>(ParMesh(MPI_COMM_WORLD, serial_mesh));
   serial_mesh.Clear(); // the serial mesh is no longer needed

   mesh->UniformRefinement();

   if (mesh->Dimension() == 2)
   {
      center(0) = 0.5;
      center(1) = 0.5;
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
   
   H1_FECollection fec(order, mesh->Dimension());
   ParFiniteElementSpace fespace(mesh.get(), &fec);
   ParGridFunction *T_gf = new ParGridFunction(&fespace);
   T_gf->ProjectCoefficient(T0);

   int local_dofs = fespace.GetTrueVSize();
   int total_dofs = 0;
   MPI_Reduce(&local_dofs, &total_dofs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
   if (Mpi::Root())
      mfem::out << "Temperature dofs: " << total_dofs << endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create the CellDeath Solver and DataCollections
   ///////////////////////////////////////////////////////////////////////////////////////////////

   int order_celldeath = 1;
   celldeath::CellDeathSolver solver(mesh, T_gf, A1, A2, A3, deltaE1, deltaE2, deltaE3); 
   celldeathgotran::CellDeathSolverGotran solver_gotran(mesh, T_gf, A1, A2, A3, deltaE1, deltaE2, deltaE3); 

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