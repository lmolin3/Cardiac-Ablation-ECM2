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
//
// Bidomain model
//
// Compile with: g++  -O3 -std=c++11 -fopenmp -I.. test_bidomain.cpp -o test_bidomain -L.. -lmfem -lrt
//
// Boundary markers for the square mesh are:
// Bottom=1, Right=2, Top=3, Left=4
//
// Run with:
// mpirun -np 4 ./test_bidomain 
//


#include "utils.hpp"
#include <fstream>
#include <sys/stat.h>  // Include for mkdir
#include <omp.h>

#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979
#endif

using namespace mfem;


struct s_BidomainContext // Bidomain params
{
   int order = 1;
   double dt = 1e-3;
   double t_final = 10 * dt;
   double lambda = 1.0;
   bool verbose = true;
   bool paraview = false;
   bool checkres = false;
   const char *outfolder = "./Output/Test/";
   int fun = 1;
   int bdf = 3;
} B_ctx;

struct s_MeshContext // mesh
{
   int n = 10;                
   int dim = 2;
   int elem = 0;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
} Mesh_ctx;


// Forward declarations of functions
void EulerAngles(const Vector &x, Vector &e);
std::function<void(const Vector &, DenseMatrix &)> DiffusionMatrix(const Array<double> &d);

int main(int argc, char *argv[])
{

   //
   /// 1. Initialize MPI and Hypre.
   //
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   //
   /// 2. Parse command-line options.
   //

   OptionsParser args(argc, argv);
   args.AddOption(&B_ctx.order,
                  "-ou",
                  "--order-velocity",
                  "Order (degree) of the finite elements for velocity.");
   args.AddOption(&B_ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&B_ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&B_ctx.verbose,
                  "-v",
                  "--verbose",
                  "-no-v",
                  "--no-verbose",
                  "Enable verbosity.");
   args.AddOption(&B_ctx.paraview,
                  "-pv",
                  "--paraview",
                  "-no-pv",
                  "--no-paraview",
                  "Enable or disable Paraview output.");
   args.AddOption(&B_ctx.checkres,
                  "-cr",
                  "--checkresult",
                  "-no-cr",
                  "--no-checkresult",
                  "Enable or disable checking of the result. Returns -1 on failure.");
    args.AddOption(&B_ctx.outfolder,
                   "-o",
                   "--output-folder",
                   "Output folder.");

    args.AddOption(&B_ctx.bdf,
                   "-bdf",
                   "--bdf-order",
                   "Maximum bdf order (1<=bdf<=3)");
   
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
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }


   //
   /// 3. Read the (serial) mesh from the given mesh file on all processors.
   //
   Element::Type type;
   switch (Mesh_ctx.elem)
   {
   case 0: // quad
      type = (Mesh_ctx.dim == 2) ? Element::QUADRILATERAL: Element::HEXAHEDRON;
      break;
   case 1: // tri
      type = (Mesh_ctx.dim == 2) ? Element::TRIANGLE: Element::TETRAHEDRON;
      break;
   }

   Mesh mesh;
   switch (Mesh_ctx.dim)
   {
   case 2: // 2d
      mesh = Mesh::MakeCartesian2D(Mesh_ctx.n,Mesh_ctx.n,type,true);
      break;
   case 3: // 3d
      mesh = Mesh::MakeCartesian3D(Mesh_ctx.n,Mesh_ctx.n,Mesh_ctx.n,type,true);
      break;
   }
   mesh.EnsureNodes();

   for (int l = 0; l < Mesh_ctx.ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }


   //
   /// 4. Define a parallel mesh by a partitioning of the serial mesh.
   // Refine this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   //
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < Mesh_ctx.par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }


   //
   // 5. Define a parallel finite element space on the parallel mesh. Here we
   //    H1 continuous high-order Lagrange finite elements of the given order.
   //
   FiniteElementCollection *fec = new H1_FECollection(B_ctx.order, Mesh_ctx.dim);
   ParFiniteElementSpace   *fes = new ParFiniteElementSpace(pmesh, fec);

   HYPRE_BigInt dimU = fes->GlobalTrueVSize();

   if (B_ctx.verbose)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(U) = " << dimU << "\n";
      std::cout << "dim(U+Ue) = " << dimU + dimU << "\n";
      std::cout << "***********************************************************\n";
   }


   //
   // 6. Define the two BlockStructure of the problem. The offsets computed
   //    here are local to the processor.
   //
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = fes->TrueVSize();
   block_offsets[2] = block_offsets[1];
   block_offsets.PartialSum();

   //
   // 7. Define the coefficients.
   //
   Array<double> di = Array<double>({1.0, 1.0, 1.0});
   Array<double> de = Array<double>({1.0, 1.0, 1.0});

   MatrixFunctionCoefficient Di(Mesh_ctx.dim, DiffusionMatrix(di));
   MatrixFunctionCoefficient De(Mesh_ctx.dim, DiffusionMatrix(de));


   //
   // 8. Assemble the finite element matrices for the Bidomain operator and Monodomain preconditioner
   //
   //                            B = [ Buu  Bue ]
   //                                [ Beu  Bee ]
   //     where:
   //     Buu = chi Cm alpha/dt M + lambda/(lambda+1) Ai
   //     Bue = lambda/(lambda+1) Ai - 1/(lambda+1) Ae
   //     Beu = Ai
   //     Bee = Ai + Ae
   //
   //     M  = \int_\Omega u_h \cdot v_h d\Omega                u_h, v_h \in U_h  
   //     Ai = \int_\Omega (\grad u_h)^T Di \grad v_h d\Omega   u_h, v_h \in U_h
   //

   int skip_zeros = 0;

   HypreParMatrix  *M = nullptr;
   HypreParMatrix *Ai = nullptr;
   HypreParMatrix *Ae = nullptr;

   ParBilinearForm *M_form  = new ParBilinearForm(fes);  // Mass 
   ParBilinearForm *Ai_form = new ParBilinearForm(fes);  // Diffusion (intracellular) 
   ParBilinearForm *Ae_form = new ParBilinearForm(fes);  // Diffusion (extracellular)

   M_form->AddDomainIntegrator(new MassIntegrator());
   M_form->Assemble(skip_zeros); M_form->Finalize(skip_zeros);
   M = M_form->ParallelAssemble();

   Ai_form->AddDomainIntegrator(new DiffusionIntegrator(Di));
   Ai_form->Assemble(skip_zeros); Ai_form->Finalize(skip_zeros);
   Ai = Ai_form->ParallelAssemble();

   Ae_form->AddDomainIntegrator(new DiffusionIntegrator(De));
   Ae_form->Assemble(skip_zeros); Ae_form->Finalize(skip_zeros);
   Ae = Ae_form->ParallelAssemble();


   double a1 = B_ctx.lambda/(B_ctx.lambda+1.0);
   double a2 = - 1.0 / (B_ctx.lambda + 1.0);
   HypreParMatrix *Buu = nullptr;
   HypreParMatrix *Bue = Add(a1, *Ai, a2, *Ae);
   HypreParMatrix *Beu = new HypreParMatrix(*Ai);
   HypreParMatrix *Bee = Add(1.0, *Ai, 1.0, *Ae);

   BlockOperator *bidomainOp = new BlockOperator(block_offsets);
   //bidomainOp->SetBlock(0, 0, Buu);
   bidomainOp->SetBlock(0, 1, Bue);
   bidomainOp->SetBlock(1, 0, Beu);
   bidomainOp->SetBlock(1, 1, Bee);

   //
   // 9. Construct the Monodomain preconditioner.
   //
   //                            M = [ invBuu        ]
   //                                [   Beu  invBee ]
   //
   // The diagonal blocks are approximated using CG preconditioned with ILU (HypreEuclid)
   //

   double rtol = 1e-6;
   double atol = 1e-10;
   int maxIter = 1000;
   int      pl = 0;

   // Solver/Preconditioner for diagonal block Buu
   HypreILU *invBuu_pc = new HypreILU();
   //invBuu_pc->SetOperator();
   invBuu_pc->SetLevelOfFill(4); // fill level of 4

   CGSolver *invBuu = new CGSolver(fes->GetComm());
   invBuu->iterative_mode = false;      
   invBuu->SetAbsTol(atol);
   invBuu->SetRelTol(rtol);
   invBuu->SetMaxIter(maxIter);
   invBuu->SetPreconditioner(*invBuu_pc);
   //invBuu->SetOperator();
   invBuu->SetPrintLevel(pl);

   // Solver/Preconditioner for diagonal block Bee
   HypreILU *invBee_pc = new HypreILU();
   invBee_pc->SetOperator(*Bee);
   invBee_pc->SetLevelOfFill(4); // fill level of 4

   CGSolver *invBee = new CGSolver(fes->GetComm());
   invBee->iterative_mode = false;      
   invBee->SetAbsTol(atol);
   invBee->SetRelTol(rtol);
   invBee->SetMaxIter(maxIter);
   invBee->SetPreconditioner(*invBee_pc);
   invBee->SetOperator(*Bee);
   invBee->SetPrintLevel(pl);

   // Create block lower triangular Monodomain Preconditioner
   BlockLowerTriangularPreconditioner *monodomainPrec = new BlockLowerTriangularPreconditioner(block_offsets);
   monodomainPrec->SetDiagonalBlock(0,invBuu);
   monodomainPrec->SetDiagonalBlock(0,invBee);
   monodomainPrec->SetBlock(1,0,Beu);


   //
   // 10. Construct the Bidomain Solver.
   //
   // FGMRES preconditioned with Monodomain
   //
   HypreFGMRES fgmres(fes->GetComm());
   fgmres.SetTol(atol);
   fgmres.SetMaxIter(maxIter);
   fgmres.SetPrintLevel(pl);
   fgmres.SetKDim(100);
   fgmres.SetPreconditioner(*monodomainPrec);
   fgmres.SetOperator(*bidomainOp);


   //
   // 11. Define solver for Ionic Model
   //


   //
   // 12. Assemble rhs
   //



   //
   /// 11. Set initial condition and boundary conditions
   //


   //
   /// 12. Solve problem
   //



   // Free memory
   delete pmesh;

   return 0;
}



// Define coefficients for conductivity

std::function<void(const Vector &, DenseMatrix &)> DiffusionMatrix(const Array<double> &d)
{
   return [d](const Vector &x, DenseMatrix &m)
   {
      // Define dimension of problem
      const int dim = x.Size();

      // Compute Euler angles
      Vector e(3);
      EulerAngles(x,e);
      double e1 = e(0);
      double e2 = e(1);
      double e3 = e(2);

      // Compute rotated matrix
      if (dim == 3)
      {
         // Compute cosine and sine of the angles e1, e2, e3
         const double c1 = cos(e1);
         const double s1 = sin(e1);
         const double c2 = cos(e2);
         const double s2 = sin(e2);
         const double c3 = cos(e3);
         const double s3 = sin(e3);

         // Fill the rotation matrix R with the Euler angles.
         DenseMatrix R(3, 3);
         R(0, 0) = c1 * c3 - c2 * s1 * s3;
         R(0, 1) = -c1 * s3 - c2 * c3 * s1;
         R(0, 2) = s1 * s2;
         R(1, 0) = c3 * s1 + c1 * c2 * s3;
         R(1, 1) = c1 * c2 * c3 - s1 * s3;
         R(1, 2) = -c1 * s2;
         R(2, 0) = s2 * s3;
         R(2, 1) = c3 * s2;
         R(2, 2) = c2;

         // Multiply the rotation matrix R with the diffusivity vector.
         Vector l(3);
         l(0) = d[0];
         l(1) = d[1];
         l(2) = d[2];

         // Compute m = R^t diag(l) R
         DenseMatrix m(3, 3);
         R.Transpose();
         MultADBt(R, l, R, m);
         return m;
      }
      else if (dim == 2)
      {
         const double c1 = cos(e1);
         const double s1 = sin(e1);
         DenseMatrix Rt(2, 2);
         Rt(0, 0) =  c1;
         Rt(0, 1) =  s1;
         Rt(1, 0) = -s1;
         Rt(1, 1) =  c1;
         Vector l(2);
         l(0) = d[0];
         l(1) = d[1];
         DenseMatrix m(2, 2);
         MultADAt(Rt,l,m);
         return m;
      }
      else
      {
         DenseMatrix m(1, 1);
         m(0, 0) = d[0];
         return m;
      }

   };
}

void EulerAngles(const Vector &x, Vector &e)
{
    const int dim = x.Size();

    e(0) = 0.0;
    e(1) = 0.0;
    if( dim == 3)
    {
        e(2) = 0.0;
    }
}