//                       Linear Elasticity on reconstructed Femur 
//                             with Partial Assembly
//
// This example is part of the MFEM tutorial for the UCBM Biomechanics of Solids course.
// Adapted from ex2p.cpp from the MFEM examples.
//
// Description:  This example code solves a linear elasticity problem on a
//               3D femur mesh using finite element methods with partial assembly.
//               The femur is fixed at the bottom (essential BCs) and a downward
//               force is applied on the femoral head (natural BCs).
//               The code demonstrates the use of parallel meshes, finite element
//               spaces, linear and bilinear forms, boundary conditions, and
//               solving the resulting linear system using the PCG solver with
//               the BoomerAMG preconditioner from hypre. When partial assembly
//               is enabled, the code uses matrix-free operator evaluation and a 
//               Low Order Refined (LOR) preconditioner.
//
// Compile with: make tutorial_ucbm_02_pa
//
// Sample runs:  
// No refinement, second order elements:
//                mpirun -np 8 tutorial_ucbm_02 -rs 0 -o 2 -of ./Output/Tutorial02/rs0_o2
//
// 1 serial refinement, first order elements:
//                mpirun -np 8 tutorial_ucbm_02 -rs 1 -o 1 -of ./Output/Tutorial02/rs1_o1
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#pragma GCC diagnostic ignored "-Wunused-variable"

using namespace std;
using namespace mfem;

void f_func(const Vector &x, Vector &f);

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
   const char *mesh_file = "../data/femur.e";
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   // Finite element space parameters
   int order = 1;
   // Output parameters
   const char *outfolder = "./Output/";

   StopWatch chrono;
   real_t t_assemble, t_solve;

   OptionsParser args(argc, argv);
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
   fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
   

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
   int bottom_attr = 1; 
   Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[bottom_attr] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);


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
   VectorFunctionCoefficient f_coeff(dim, f_func);
      
   ParLinearForm *b = new ParLinearForm(fespace);

   int head_attr = 0; // Boundary attribute for femoral head
   Array<int> head_bdr(pmesh->bdr_attributes.Max());
   head_bdr = 0;
   head_bdr[head_attr] = 1;
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f_coeff), head_bdr);
   if (Mpi::Root())
   {
      out << "r.h.s. ... " << std::flush;
   }
   b->Assemble();

   //<--- Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   //<--- Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator.
   // Typical cortical bone properties: (Note: geometric units are in mm, so you need to convert)
   // Young's modulus E = 20 GPa = 20000 MPa = 20000 N/mm² (conservative for cortical bone)
   // Poisson's ratio ν = 0.3
   // Lamé parameters: λ = Eν/((1+ν)(1-2ν)), μ = E/(2(1+ν))
   double E = 20000.0;                                       // Young's modulus in N/mm² (increased from 17000)
   double nu = 0.3;                                          // Poisson's ratio
   double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)); // ≈ 11538 N/mm²
   double mu = E / (2.0 * (1.0 + nu)); 

   ConstantCoefficient lambda_func(lambda);
   ConstantCoefficient mu_func(mu);

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

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

   ParaViewDataCollection pvdc("Femur", pmesh);
   pvdc.SetPrefixPath(outfolder);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   if (order > 1)
   {
      pvdc.SetHighOrderOutput(true);
      pvdc.SetLevelsOfDetail(order);
   }
   pvdc.RegisterField("displacement", &x);

   // Save initial condition
   pvdc.SetCycle(0);
   pvdc.SetTime(0.0);
   pvdc.Save();

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
   pvdc.SetCycle(1);
   pvdc.SetTime(1.0);
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

void f_func(const Vector &x, Vector &f)
{
   // Realistic physiological loading for femur:
   // - Body weight ~700 N (typical adult ~70 kg)
   // - Static standing: ~1x body weight on femur head
   // - Load during standing: ~700 N
   // - Applied over femoral head contact area ~500-1000 mm²
   // - Pressure = Force/Area ≈ 0.7-1.4 N/mm²
   //
   // Current load: 0.17 N/mm² traction applied over ~2950 mm² boundary
   // Total force: ~500 N (equivalent to bodyweight during standing)
   
   f(0) = 0.0;   // No load in x-direction
   f(1) = 0.0;   // No load in y-direction
   f(2) = -0.17; // 0.17 N/mm² downward compression (~500 N total)
}