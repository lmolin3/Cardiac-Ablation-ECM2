//                       Poisson Equation
//
// This example is part of the MFEM tutorial for the UCBM Biomechanics of Solids course.
// Adapted from ex1p.cpp from the MFEM examples.
//
// Description:  This example code is based on tutorial_ucbm_01.cpp serves to run
//               a convergence study for the Poisson problem.
//
// Compile with: make tutorial_ucbm_01
//
// Sample runs:
// You can run any of the cases from tutorial_ucbm_01.cpp.
//
// 1. Convergence study on linear finite elements (should observe rate ~2 in L2 and ~1 in H1):
//               mpirun -np 4 tutorial_ucbm_01_convergence -rs 3 -o 1
//
// 2. Convergence study on cubic finite elements (should observe rate ~4 in L2 and ~3 in H1):
//               mpirun -np 4 tutorial_ucbm_01_convergence -rs 3 -o 3
//
// 3. Modified frequency for the exact solution (default is freq=2.0):
//               mpirun -np 4 tutorial_ucbm_01_convergence -rs 3 -o 2 -k 4.0
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#pragma GCC diagnostic ignored "-Wunused-variable"

using namespace std;
using namespace mfem;

// Setting the frequency for the exact solution
real_t freq = 2.0;
real_t kappa;

// Exact smooth analytic solution for convergence study
real_t u_exact(const Vector &, real_t);
void u_grad_exact(const Vector &, Vector &);
real_t f_exact(const Vector &);

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
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   // Finite element space parameters
   int order = 1;
   // Output parameters
   const char *outfolder = "./Output/";

   OptionsParser args(argc, argv);
   args.AddOption(&serial_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&parallel_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-k", "--frequency",
                  "Frequency for the exact solution.");
   args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");

   args.ParseCheck();

   kappa = freq * M_PI;

   if (Mpi::Root())
   {
      out << "----------------------------------------------------------------------------------------"
          << std::endl;
      out << std::left << std::setw(16) << "DOFs " << std::setw(16) << "h " << std::setw(16) << "L^2 error " << std::setw(16);
      out << "L^2 rate " << std::setw(16) << "H^1 error " << std::setw(16) << "H^1 rate" << std::endl;
      out << "----------------------------------------------------------------------------------------"
          << std::endl;
   }

   real_t l2_err_prev = 0.0;
   real_t h1_err_prev = 0.0;
   real_t h_prev = 0.0;

   for (int ref_level = 0; ref_level <= 4; ref_level++)
   {

      /////////////////////////////////////////////////////////////////////////////
      //------     3. Create serial and parallel mesh
      /////////////////////////////////////////////////////////////////////////////

      //<--- Load serial mesh
      Mesh *mesh = new Mesh("../data/inline-quad.mesh");
      mesh->EnsureNodes();
      int dim = mesh->Dimension();

      //<--- Refine the serial mesh on all processors to increase the resolution. I
      for (int l = 0; l < ref_level; l++)
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
      ParFiniteElementSpace *fespace, *fespace_grad;
      fec = new H1_FECollection(order, dim);
      fespace = new ParFiniteElementSpace(pmesh, fec);
      fespace_grad = new ParFiniteElementSpace(pmesh, fec, dim);

      auto size = fespace->GlobalTrueVSize();

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

      //<--- Set up the parallel linear form b(.) which corresponds to the
      //     right-hand side of the FEM linear system. In this case, b_i equals the
      //     boundary integral of f*phi_i where f represents a "pull down" force on
      //     the Neumann part of the boundary and phi_i are the basis functions in
      //     the finite element fespace.
      ParLinearForm *b = new ParLinearForm(fespace);
      FunctionCoefficient f_exact_coeff(f_exact);
      b->AddDomainIntegrator(new DomainLFIntegrator(f_exact_coeff));
      b->Assemble();

      //<--- Define the solution vector x as a parallel finite element grid
      //     function corresponding to fespace. Initialize x with initial guess of
      //     zero, which satisfies the boundary conditions.

      FunctionCoefficient u_ex_coeff(u_exact);
      VectorFunctionCoefficient grad_u_ex_coeff(dim, u_grad_exact);

      ParGridFunction x(fespace);
      ParGridFunction x_exact(fespace);
      ParGridFunction grad_x(fespace_grad);

      x.ProjectCoefficient(u_ex_coeff);
      x_exact.ProjectCoefficient(u_ex_coeff);
      grad_x.ProjectCoefficient(grad_u_ex_coeff);

      //<--- Set up the parallel bilinear form a(.,.) on the finite element space
      //     corresponding to the Laplacian operator -Delta, by adding the
      //     Diffusion domain integrator.

      ConstantCoefficient one(1.0);
      ParBilinearForm *a = new ParBilinearForm(fespace);
      a->AddDomainIntegrator(new DiffusionIntegrator(one));

      //<--- Assemble the parallel bilinear form and the corresponding linear
      //     system, applying any necessary transformations such as: parallel
      //     assembly, eliminating boundary conditions, etc.
      a->Assemble();

      HypreParMatrix A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

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
      solver->SetPrintLevel(0);

      /////////////////////////////////////////////////////////////////////////////
      //------     8. Define output and visualization
      /////////////////////////////////////////////////////////////////////////////

      // Initialize Paraview visualization
      std::string name = "MMS-ref" + std::to_string(ref_level);
      ParaViewDataCollection pvdc(name, pmesh);
      pvdc.SetPrefixPath(outfolder);
      pvdc.SetDataFormat(VTKFormat::BINARY32);
      if (order > 1)
      {
         pvdc.SetHighOrderOutput(true);
         pvdc.SetLevelsOfDetail(order);
      }
      pvdc.RegisterField("x", &x);
      pvdc.RegisterField("x_exact", &x_exact);

      /////////////////////////////////////////////////////////////////////////////
      //------     8. Solve the problem and save
      /////////////////////////////////////////////////////////////////////////////

      //<--- Solve the linear system A X = B
      solver->Mult(B, X);

      //<--- Recover the parallel grid function corresponding to X. This is the
      //     local finite element solution on each processor.
      a->RecoverFEMSolution(X, *b, x);

      //<--- Save the solution in parallel using ParaView Data Collection
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.Save();

      ///////////////////////////////////////////////////////////////////////////////////////////////
      //------ 9. Compute and print the L^2 and H^1 norms of the error.
      ///////////////////////////////////////////////////////////////////////////////////////////////

      real_t l2_err = 0.0;
      real_t h1_err = 0.0;
      real_t l2_rate = 0.0;
      real_t h1_rate = 0.0;
      l2_err = x.ComputeL2Error(u_ex_coeff);
      h1_err = x.ComputeH1Error(&u_ex_coeff, &grad_u_ex_coeff);

      real_t h_min = 0.0;
      real_t h_max = 0.0;
      real_t kappa_min = 0.0;
      real_t kappa_max = 0.0;
      pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

      if (serial_ref_levels != 0)
      {
         l2_rate = log(l2_err / l2_err_prev) / log(h_min / h_prev);
         h1_rate = log(h1_err / h1_err_prev) / log(h_min / h_prev);
      }
      else
      {
         l2_rate = 0.0;
         h1_rate = 0.0;
      }

      l2_err_prev = l2_err;
      h1_err_prev = h1_err;
      h_prev = h_min;

      if (Mpi::Root())
      {
         out << setw(16) << size << setw(16) << h_min << setw(16) << l2_err << setw(16) << l2_rate;
         out << setw(16) << h1_err << setw(16) << h1_rate << endl;
      }

      /////////////////////////////////////////////////////////////////////////////
      //------     10. Cleanup and free memory
      /////////////////////////////////////////////////////////////////////////////

      //<--- Free the used memory.
      delete solver;
      delete prec;
      delete a;
      delete b;
      if (fespace_grad)
      {
         delete fespace_grad;
         fespace_grad = nullptr;
      }
      if (fespace)
      {
         delete fespace;
         fespace = nullptr;
      }
      if (fec)
      {
         delete fec;
         fec = nullptr;
      }
      delete pmesh;
   }

   return 0;
}

real_t u_exact(const Vector &x, real_t t)
{
   real_t phi = 0.0;
   if (x.Size() == 2)
   {
      phi = sin(kappa * x(0)) * sin(kappa * x(1));
   }
   else
   {
      phi = sin(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2));
   }

   return phi;
}

void u_grad_exact(const Vector &x, Vector &gradPhi)
{
   if (x.Size() == 2)
   {
      gradPhi(0) = kappa * cos(kappa * x(0)) * sin(kappa * x(1));
      gradPhi(1) = kappa * sin(kappa * x(0)) * cos(kappa * x(1));
   }
   else
   {
      gradPhi(0) = kappa * cos(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2));
      gradPhi(1) = kappa * sin(kappa * x(0)) * cos(kappa * x(1)) * sin(kappa * x(2));
      gradPhi(2) = kappa * sin(kappa * x(0)) * sin(kappa * x(1)) * cos(kappa * x(2));
   }
}

real_t f_exact(const Vector &x)
{
   real_t f = 0.0;
   if (x.Size() == 2)
   {
      f = 2.0 * kappa * kappa * (sin(kappa * x(0)) * sin(kappa * x(1)));
   }
   else
   {
      f = 3.0 * kappa * kappa * (sin(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2)));
   }

   return f;
}