//                       Poisson Equation
//
// This example is part of the MFEM tutorial for the UCBM Biomechanics of Solids course.
// Adapted from ex1p.cpp from the MFEM examples.
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Poisson problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order.
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions.
//
// Compile with: make tutorial_ucbm_01
//
// Sample runs:  mpirun -np 4 tutorial_ucbm_01 -m ../data/square-disc.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/star.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/star-mixed.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/escher.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/fichera.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/fichera-mixed.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/toroid-wedge.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/octahedron.mesh -o 1
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/periodic-annulus-sector.msh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/periodic-torus-sector.msh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/square-disc-nurbs.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/disc-nurbs.mesh 
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/pipe-nurbs.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/star-surf.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/square-disc-surf.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/inline-segment.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/amr-quad.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/amr-hex.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/mobius-strip.mesh
//               mpirun -np 4 tutorial_ucbm_01 -m ../data/mobius-strip.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#pragma GCC diagnostic ignored "-Wunused-variable"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   /////////////////////////////////////////////////////////////////////////////
   //------     1. Initialize MPI and HYPRE.     
   /////////////////////////////////////////////////////////////////////////////

   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();


   /////////////////////////////////////////////////////////////////////////////
   //------     2. Parse command-line options.         
   /////////////////////////////////////////////////////////////////////////////

   // Mesh parameters
   const char *mesh_file = "../data/star.mesh";
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   // Finite element space parameters
   int order = 1;
   // Output parameters
   const char *outfolder = "./Output/";

   StopWatch chrono;
   real_t t_assemble, t_solve;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&serial_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&parallel_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");
   args.ParseCheck();


   /////////////////////////////////////////////////////////////////////////////
   //------     3. Create serial and parallel mesh
   /////////////////////////////////////////////////////////////////////////////

   //<--- Load serial mesh
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //<--- Refine the serial mesh on all processors to increase the resolution. I
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   //<--- Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int l = 0; l < parallel_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }


   /////////////////////////////////////////////////////////////////////////////
   //------     4. Define finite element space
   /////////////////////////////////////////////////////////////////////////////

   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   fec = new H1_FECollection(order, dim);
   fespace = new ParFiniteElementSpace(pmesh, fec);
   

   auto size = fespace->GlobalTrueVSize();
   if (Mpi::Root())
   {
      out << "Number of finite element unknowns: " << size << endl
          << "Assembling: " << std::flush;
   }


   /////////////////////////////////////////////////////////////////////////////
   //------     5. Boundary conditions
   /////////////////////////////////////////////////////////////////////////////

   //<--- Determine the list of true (i.e. parallel conforming) essential boundary dofs. 
   //     The essential boundary conditions are defined by marking boundary attributes and
   //     converting it to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      // Apply boundary conditions on all external boundaries:
      pmesh->MarkExternalBoundaries(ess_bdr);
      // Boundary conditions can also be applied based on named attributes:
      // pmesh->MarkNamedBoundaries(set_name, ess_bdr)

      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }


   /////////////////////////////////////////////////////////////////////////////
   //------     6. Create the Linear and Bilinear Forms
   /////////////////////////////////////////////////////////////////////////////

   chrono.Clear();
   chrono.Start();

   //<--- Set up the parallel linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system. In this case, b_i equals the
   //     boundary integral of f*phi_i where f represents a "pull down" force on
   //     the Neumann part of the boundary and phi_i are the basis functions in
   //     the finite element fespace. 
   ParLinearForm *b = new ParLinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   //<--- Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   //<--- Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the
   //     Diffusion domain integrator.

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   //<--- Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, etc.
   if (Mpi::Root()) { out << "matrix ... " << std::flush; }
   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   if (Mpi::Root())
   {
      out << "done." << endl;
      out << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   chrono.Stop();
   t_assemble = chrono.RealTime();

   /////////////////////////////////////////////////////////////////////////////
   //------     7. Define solver/preconditioner and solve the linear system
   /////////////////////////////////////////////////////////////////////////////

   //<--- Define a parallel PCG solver for A X = B with the BoomerAMG
   //     preconditioner from hypre.
   HypreBoomerAMG *prec = new HypreBoomerAMG(A);
   prec->SetSystemsOptions(dim);

   prec->SetPrintLevel(0);

   CGSolver *solver = new CGSolver(MPI_COMM_WORLD);
   solver->SetPreconditioner(*prec);
   solver->SetOperator(A);
   solver->SetRelTol(1e-12);
   solver->SetMaxIter(1000);
   solver->SetPrintLevel(1);


   /////////////////////////////////////////////////////////////////////////////
   //------     8. Define output and visualization
   /////////////////////////////////////////////////////////////////////////////

   ParaViewDataCollection pvdc(mesh_file, pmesh);
   pvdc.SetPrefixPath(outfolder);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   if (order > 1)
   {
      pvdc.SetHighOrderOutput(true);
      pvdc.SetLevelsOfDetail(order);
   }
   pvdc.RegisterField("x", &x);

   /////////////////////////////////////////////////////////////////////////////
   //------     9. Solve the problem and save
   /////////////////////////////////////////////////////////////////////////////

   //<--- Solve the linear system A X = B
   chrono.Clear();
   chrono.Start();

   solver->Mult(B, X);

   chrono.Stop();
   t_solve = chrono.RealTime();

   //<--- Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   //<--- Save results
   pvdc.SetCycle(0);
   pvdc.SetTime(0.0);
   pvdc.Save();


   //<--- Print timing information

   // Compute global times
   real_t t_assemble_g, t_solve_g;
   MPI_Allreduce(&t_assemble, &t_assemble_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(&t_solve, &t_solve_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);

   if (Mpi::Root())
   {
      // Print timing summary
      out << std::endl;
      out << "-----------------------------------------------" << std::endl;
      out << "Timing Summary" << std::endl;
      out << "-----------------------------------------------" << std::endl;

      // Helper lambda to format time with appropriate units
      auto format_time = [](real_t time_s) -> std::string
      {
         if (time_s < 0.1)
         {
            return std::to_string(time_s * 1000.0) + " ms";
         }
         else
         {
            return std::to_string(time_s) + " s ";
         }
      };

      out << std::fixed << std::setprecision(3);
      out << std::setw(30) << std::left << "Assembly time:"
          << std::setw(15) << std::right << format_time(t_assemble_g) << std::endl;
      out << std::setw(30) << std::left << "Solution time:"
          << std::setw(15) << std::right << format_time(t_solve_g) << std::endl;
      out << "-----------------------------------------------" << std::endl;
   }


   /////////////////////////////////////////////////////////////////////////////
   //------     8. Cleanup and free memory
   /////////////////////////////////////////////////////////////////////////////

   //<--- Free the used memory.
   delete solver;
   delete prec;
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete pmesh;

   return 0;
}
