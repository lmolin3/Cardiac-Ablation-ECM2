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
// Test for P2 mass lumping
//
// Run with:
// mpirun -np 4 ./test_lumping_global_scal -d 2 -e 1 -n 10 -ou 2 -f 1 --output-folder ./Output/TestLumping/ -pm 

// Include mfem and I/O
#include "mfem.hpp"
#include <fstream>
#include <iostream>

// Include for mkdir
#include <sys/stat.h>

// Include for defining exact solution
#include <math.h>

// Include for std::bind and std::function
#include <functional>

// Include ns-ecm2 miniapp
#include "custom_bilinteg.hpp"


using namespace mfem;

// Forward declarations of functions
void PrintMatrices( const char* outFolder );
void LumpingDiagScaling( HypreParMatrix *M, double dim,  Vector &M_lump);
double x_lin(const Vector &X);
double x_const(const Vector &X);

// Declare the global matrices
HypreParMatrix *M = nullptr;
HypreParMatrix *M_lump1= nullptr;
Vector M_lump2;
Vector *x= nullptr;
Vector *b= nullptr;
Vector *x1= nullptr;
Vector *x2= nullptr;
ParMesh *pmesh;
const char *outFolder = "./";

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
    int uorder = 2;            // fe

    int n = 10;                // mesh
    int dim = 2;
    int elem = 0;
    int ser_ref_levels = 0;
    int par_ref_levels = 0;

    bool printMat = false;

    int fun = 1;

    mfem::OptionsParser args(argc, argv);
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
    args.AddOption(&uorder, "-ou", "--order_vel",
                   "Finite element order for velocity (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&outFolder,
                   "-o",
                   "--output-folder",
                   "Output folder.");
   args.AddOption(&printMat, "-pm", "--print-mat", "-no-pm", "--no-print-mat",
                  "Enable or output of matrices.");
    args.AddOption(&fun, "-f", "--test-function",
                   "Analytic function to test");

    args.Parse();
    if (!args.Good())
    {
        if (myrank == 0)
        {
            args.PrintUsage(mfem::out);
        }
        MPI_Finalize();
        return 1;
    }
    if (myrank == 0)
    {
        args.PrintOptions(mfem::out);
    }


    //
    /// 3. Read the (serial) mesh from the given mesh file on all processors.
    //
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

   if (myrank == 0)
   {
    mfem::out<<"Creating serial mesh."<<std::endl;
   }
    Mesh mesh;
    switch (dim)
    {
    case 2: // 2d
        mesh = Mesh::MakeCartesian2D(n,n,type,true);
        break;
    case 3: // 3d
        mesh = Mesh::MakeCartesian3D(n,n,n,type,true);
        break;
    }

    for (int l = 0; l < ser_ref_levels; l++)
    {
        mesh.UniformRefinement();
    }
 

    //
    /// 4. Define a parallel mesh by a partitioning of the serial mesh.
    // Refine this mesh further in parallel to increase the resolution. Once the
    // parallel mesh is defined, the serial mesh can be deleted.
    //
   if (myrank == 0)
   {
    mfem::out<<"Creating serial mesh."<<std::endl;
   }
    pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();
    {
        for (int l = 0; l < par_ref_levels; l++)
        {
            pmesh->UniformRefinement();
        }
    }


    //
    /// 5. Create Consistent and Lumped Mass matrices
    //
    H1_FECollection ufec(uorder,dim);
    ParFiniteElementSpace ufes(pmesh, &ufec);
    int vdim = ufes.GetTrueVSize();

    //
    /// 6. Create Consistent and Lumped Mass matrices
    //
    ParBilinearForm M_form(&ufes);
    ParBilinearForm M_lump1_form(&ufes);

    ConstantCoefficient coeff(1.0);

    StopWatch sw_assembly, sw_print;
    sw_assembly.Start();

   if (pmesh->GetMyRank() == 0)
   {
    mfem::out<<"Adding integrators."<<std::endl;
   }
    M_form.AddDomainIntegrator(new MassIntegrator( coeff ));
    M_lump1_form.AddDomainIntegrator( new LumpedIntegrator( new MassIntegrator( coeff ) ) );

   if (pmesh->GetMyRank() == 0)
   {
    mfem::out<<"Assembling and finalizing."<<std::endl;
   }
    int skip_zeros = 0;  // Maintain sparsity pattern
    M_form.Assemble(skip_zeros);  M_form.Finalize(skip_zeros); 
    M_lump1_form.Assemble(skip_zeros);  M_lump1_form.Finalize(skip_zeros); 

   if (pmesh->GetMyRank() == 0)
   {
    mfem::out<<"Parallel Assemble."<<std::endl;
   }
    
    M = M_form.ParallelAssemble();
    M_lump1 = M_lump1_form.ParallelAssemble();
    Vector *M_lump1_vec = new Vector(vdim);
    M_lump1->GetDiag(*M_lump1_vec);
    
    M_lump2.SetSize(vdim);
    LumpingDiagScaling(M, dim, M_lump2);

    sw_assembly.Stop();

    // Solve system with mass matrix
    x = new Vector(vdim); // solution vec
    ParGridFunction x_gf(&ufes);

    FunctionCoefficient*    x_fun = nullptr;
    switch (fun)
    {
    case 1:
    {
        x_fun = new FunctionCoefficient(x_const);
        break;
    }
    case 2:
    {
        x_fun = new FunctionCoefficient(x_lin);
        break;
    }
    default:
        break;
    }

    x_gf.ProjectCoefficient(*x_fun);
    x_gf.GetTrueDofs(*x);

    b = new Vector(vdim); //b.Randomize();        // rhs
    M->Mult(*x, *b);
    
    x1 = new Vector(*b);  // solution vec
    x2 = new Vector(*b);  // solution vec

#ifndef MFEM_USE_MUMPS
    if (pmesh->GetMyRank() == 0)
    {
        mfem::out<<"Using CG solvers."<<std::endl;
    }
    CGSolver invM(MPI_COMM_WORLD);
    invM.SetRelTol(1e-12);
    invM.SetMaxIter(1000);
    invM.SetPrintLevel(0);
    invM.SetOperator(*M);

    invM.Mult(*b, *x);
#else
    if (pmesh->GetMyRank() == 0)
    {
        mfem::out<<"Using MUMPS solvers."<<std::endl;
    }
    MUMPSSolver mumps(pmesh->GetComm());
    mumps.SetPrintLevel(0);
    mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
    mumps.SetOperator(*M);
    mumps.Mult(*b, *x);
#endif

    *x1 /= *M_lump1_vec;
    *x2 /= M_lump2;


    // Compute error
    Vector diff1(*x), diff2(*x), diff(*x);
    diff1 -= *x1;
    diff2 -= *x2;
    diff -= *x;
    double err = diff.Norml2();
    double err1 = diff1.Norml2();
    double err2 = diff2.Norml2();
  
    if ( pmesh->GetMyRank() == 0 )
    {
        mfem::out << "   " << "|| x - x ||  " << std::setw(3)
            << std::setprecision(2) << std::scientific << err
            << "   " << std::endl;
        mfem::out << "   " << "|| x - x1 ||  " << std::setw(3)
            << std::setprecision(2) << std::scientific << err1
            << "   " << std::endl;
        mfem::out << "   " << "|| x - x2 ||  " << std::setw(3)
            << std::setprecision(2) << std::scientific << err2
            << "   " << std::endl;
        mfem::out << std::endl;
      }

    // Print matrices
   if (pmesh->GetMyRank() == 0 && printMat)
   {
    mfem::out<<"Printing matrices."<<std::endl;
    sw_print.Start();
    PrintMatrices( outFolder );
    sw_print.Stop();
    }
    double my_rt[2], rt_max[2];

   my_rt[0] = sw_assembly.RealTime();
   my_rt[1] = sw_print.RealTime();

   MPI_Reduce(my_rt, rt_max, 2, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(10) << "ASSEMBLY" << std::setw(10) << "PRINT" << "\n";
      mfem::out << std::setprecision(3) << std::setw(10) << my_rt[0] << std::setw(10) << my_rt[1] << "\n";
   }

    // Free memory
    delete pmesh; pmesh = nullptr;
    delete M; M = nullptr;
    delete M_lump1; M_lump1 = nullptr;


    return 0;
}


void PrintMatrices( const char* outFolder )
{
    if (pmesh->GetMyRank() == 0)
    {
   
        // Create folder
        std::string folderName(outFolder);
        if (mkdir(folderName.c_str(), 0777) == -1   &&  (pmesh->GetMyRank() == 0) )
        { mfem::err << "Error :  " << strerror(errno) << std::endl; };

        //Create files
        std::ofstream M_file(std::string(folderName) + '/' + "M.dat");
        std::ofstream M_lump1_file(std::string(folderName) + '/' + "M_lump1.dat");
        std::ofstream M_lump2_file(std::string(folderName) + '/' + "M_lump2.dat");
        std::ofstream b_file(std::string(folderName) + '/' + "b.dat");
        std::ofstream x_file(std::string(folderName) + '/' + "x.dat");
        std::ofstream x1_lump1_file(std::string(folderName) + '/' + "x_lump1.dat");
        std::ofstream x2_lump2_file(std::string(folderName) + '/' + "x_lump2.dat");       

        SparseMatrix M_lump2_mat(M_lump2);

        // Print matrices in matlab format
        M->PrintMatlab(M_file);
        M_lump1->PrintMatlab(M_lump1_file);
        M_lump2_mat.PrintMatlab(M_lump2_file);
        b->Print(b_file);
        x->Print(x_file);
        x1->Print(x1_lump1_file);
        x2->Print(x2_lump2_file);

   }

}


void LumpingDiagScaling( HypreParMatrix *M, double dim, Vector &M_lump)
{
    M->GetDiag(M_lump);
    double Trace = M_lump.Sum();
    M_lump *= 1.0 / Trace;
}



double x_lin(const Vector &X)
{
    const int dim = X.Size();

    double x = X[0];
    double y = X[1];
    if( dim == 3)
    {
        double z = X[2];
    }

    return y;
}

double x_const(const Vector &X)
{
    return 1.0;
}