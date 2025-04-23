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
//            -----------------------------------------------------
//            Volta Miniapp:  Simple Electrostatics Simulation Code
//            -----------------------------------------------------
//
// This miniapp solves a simple 2D or 3D electrostatic problem (Quasi-static Maxwell).
//
//                            Div sigma Grad Phi = 0
//
// Boundary conditions consist in:
// * Dirichlet: user defined potential
// * Dirichlet: potential leading to a user defined uniform electric field.
// * Neumann: selected current density
//
// We discretize the electric potential with H1 finite elements.
// The electric field E is discretized with HCurl finite elements (just for visualization/postprocessing).
//
// Sample runs:
//
//   A cylinder at constant voltage in a square, grounded metal pipe:
//      mpirun -np 4 ./electrostatics_test3D_aniso_rotated -m ../multidomain/multidomain-hex.mesh -rs 1 -o 4 -sattr '1 2' -sval '1.0 1.0' -ar '0.0 3.0' -dbcs '6 7 8' -dbcv '1.0 0.0 0.0' -pa
//

#include "lib/electrostatics_solver.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/stat.h> // Include for mkdir

using namespace std;
using namespace mfem;
using namespace mfem::electrostatics;


// Forward declarations of functions

// Conductivity Matrix
void EulerAngles(const Vector &x, Vector &e);
std::function<void(const Vector &, DenseMatrix &)> ConductivityMatrix( const Vector &d);

// Boundary Conditions
void SetupBCHandler(BCHandler *bcs, Array<int> &dbcs, Vector &dbcv, Array<int> &dbce, Array<int> &nbcs, Vector &nbcv, std::vector<Vector> &e_uniform);

static Vector pw_sigma(0);    // Piecewise conductivity values
static Vector aniso_ratio(0); // Piecewise anisotropy ratio for conductivity
static Vector sigma_attr(0);  // Domain attributes associated to piecewise Conductivity

// Phi Boundary Condition
std::vector<Vector> e_uniform(0);
static Vector dbce_val(0);
bool uebc_const = true;


int main(int argc, char *argv[])
{
   /// 1. Initialize MPI and Hypre
   Mpi::Init(argc, argv);
   Hypre::Init();

   /// 2. Parse command-line options.
   const char *mesh_file = "../multidomain/multidomain-hex.mesh";
   int order = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   int prec_type = 1;
   bool visualization = false;
   bool visit = false;
   bool paraview = true;
   const char *outfolder = "./Output/Test/";
   bool pa = false;


   Array<int> dbcs;
   Array<int> dbce;
   Array<int> nbcs;

   Vector dbcv;
   Vector nbcv;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&sigma_attr, "-sattr", "--sigma-attributes",
                  "Domain attributes associated to piecewise Conductivity");
   args.AddOption(&pw_sigma, "-sval", "--piecewise-sigma",
                  "Piecewise values of Conductivity");
   args.AddOption(&aniso_ratio, "-ar", "--anisotropy-ratio",
                  "Piecewise values of Anisotropy Ratio");
   args.AddOption(&dbcs, "-dbcs", "--dirichlet-bc-surf",
                  "Dirichlet Boundary Condition Surfaces");
   args.AddOption(&dbcv, "-dbcv", "--dirichlet-bc-vals",
                  "Dirichlet Boundary Condition Values");
   args.AddOption(&dbce, "-dbce", "--dirichlet-bc-efield",
                  "Dirichlet Boundary Condition Gradient (phi = -z) Surfaces");
   args.AddOption(&dbce_val, "-uebc", "--uniform-e-bc",
                  "Specify the three components of the constant "
                  "electric field (if more dbce specified, write all the components in order)");
   args.AddOption(&uebc_const, "-uebc-const", "--uniform-e-bc-const",
                  "-no-uebc-const", "--no-uniform-e-bc-const",
                  "Specify only one electric field for all boundaries listed in dbce ");
   args.AddOption(&nbcs, "-nbcs", "--neumann-bc-surf",
                  "Neumann Boundary Condition Surfaces");
   args.AddOption(&nbcv, "-nbcv", "--neumann-bc-vals",
                  "Neumann Boundary Condition Values");
   args.AddOption(&prec_type, "-prec", "--preconditioner",
                  "Preconditioner type (full assembly): 0 - BoomerAMG, 1 - LOR, \n"
                  "Preconditioner type (partial assembly): 0 - Jacobi smoother, 1 - LOR");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&outfolder,
                  "-of",
                  "--output-folder",
                  "Output folder.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   
   /// 3. Read Mesh and create parallel
   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   StopWatch sw_initialization;
   sw_initialization.Start();
   if (Mpi::Root())
   {
      cout << "Starting initialization." << endl;
   } 
   
   Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
   for (int l = 0; l < serial_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }


   auto pmesh = make_shared<ParMesh>(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   int sdim = pmesh->SpaceDimension();



   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = parallel_ref_levels;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }
   // Make sure tet-only meshes are marked for local refinement.
   pmesh->Finalize(true);

   /// 4. Set up conductivity coefficient

   int d = pmesh->Dimension();

   Array<int> attr(0);
   Array<MatrixCoefficient *> coefs(0);

   MFEM_ASSERT(pw_sigma.Size() == sigma_attr.Size(), "Size mismatch between conductivity values and attributes");

   for (int i = 0; i < pw_sigma.Size(); i++)
   {
      MFEM_ASSERT(sigma_attr[i] <= pmesh->attributes.Max(), "Attribute value out of range");

      real_t sigma_val = pw_sigma[i];
      real_t sigma_ratio = aniso_ratio[i];
      // Create vector of conductivity values
      Vector sigma_vec(d);
      sigma_vec(0) = sigma_val;
      sigma_vec(1) = sigma_val / sigma_ratio;
      sigma_vec(2) = sigma_val / sigma_ratio;
   
      MatrixCoefficient *tmp = pw_sigma[i] != 0 ? new MatrixFunctionCoefficient(3, ConductivityMatrix(sigma_vec)) : NULL;

      coefs.Append(tmp);
      attr.Append(sigma_attr[i]);
   }

   PWMatrixCoefficient *sigmaCoeff = new PWMatrixCoefficient(d, attr, coefs);

   /// 5. Set up boundary conditions

   // Default values for Dirichlet BCs
   // If the gradient bc was selected but the E field was not specified
   // set a default vector value.

   // Unpack the vector of uniform electric fields: dbce_val has values of electric field (size determined by sze of the mesh) for each bc applied in dbce, so we should unpack them into vectors in e_uniform
   if (dbce)
   {
      int n = dbce.Size();
      int nval = dbce_val.Size();
      int s = pmesh->SpaceDimension();
      int k = 0;

      for (int i = 0; i < dbce.Size(); i++)
      {
         Vector e_tmp(s);
         e_tmp = 0.0;
         for (int j = 0; j < s; j++)
         {
            k = k + j;
            e_tmp(j) = (k) < nval ? static_cast<real_t>(dbce_val(k)) : 0.0;
         }
         e_uniform.push_back(e_tmp);

         if (uebc_const) // If the user wants to use the same electric field for all boundaries
         {
            break;
         }
         k++;
      }
   }

   // If values for Dirichlet BCs were not set assume they are zero
   if (dbcv.Size() < dbcs.Size() && !dbce)
   {
      dbcv.SetSize(dbcs.Size());
      dbcv = 0.0;
   }

   // If values for Neumann BCs were not set assume they are zero
   if (nbcv.Size() < nbcs.Size())
   {
      nbcv.SetSize(nbcs.Size());
      nbcv = 0.0;
   }

   // Create BCHandler and parse bcs
   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool verbose = true;
   BCHandler *bcs = new BCHandler(pmesh, verbose); // Boundary conditions handler
   SetupBCHandler(bcs, dbcs, dbcv, dbce, nbcs, nbcv, e_uniform);

   /// 6. Create the Electrostatics Solver
   // Create the Electrostatic solver
   ElectrostaticsSolver Volta(pmesh, order, bcs, sigmaCoeff, verbose);
Volta.display_banner(std::cout);

   // Initialize GLVis visualization
   if (visualization)
   {
      Volta.InitializeGLVis();
   }

   if ((mkdir(outfolder, 0777) == -1) && Mpi::Root())
   {
      mfem::err << "Error :  " << strerror(errno) << std::endl;
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Volta-Parallel", pmesh.get());
   visit_dc.SetPrefixPath(outfolder);

   if (visit)
   {
      Volta.RegisterVisItFields(visit_dc);
   }

   // Initialize Paraview visualization
   ParaViewDataCollection paraview_dc("Volta-Parallel", pmesh.get());

   if (paraview)
   {
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetPrefixPath(outfolder);
         paraview_dc.SetLevelsOfDetail(order);
         Volta.RegisterParaviewFields(paraview_dc);
   }

   sw_initialization.Stop();
   real_t my_rt[1], rt_max[1];
   my_rt[0] = sw_initialization.RealTime();
   MPI_Reduce(my_rt, rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   if (Mpi::Root())
   {
      cout << "Initialization done.  Elapsed time " << my_rt[0] << " s." << endl;
   }

   /// 7. Solve the problem
   if (Mpi::Root())
   {
      cout << "\nSolving... " << endl;
   }

   // Display the current number of DoFs in each finite element space
   Volta.PrintSizes();

   // Setup solver and Assemble all forms
   int pl = 1;
   Volta.EnablePA(pa);
   Volta.Setup(prec_type, pl);

   // Solve the system and compute any auxiliary fields
   Volta.Solve();

   // Determine the current size of the linear system
   int prob_size = Volta.GetProblemSize();

   // Write fields to disk for VisIt
   if (visit || paraview)
   {
      Volta.WriteFields();
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      Volta.DisplayToGLVis();
   }

   Volta.PrintTimingData();


   /// 8. Cleanup
   // Delete the MatrixCoefficient objects at the end of main
   for (int i = 0; i < coefs.Size(); i++)
   {
      delete coefs[i];
   }

   delete sigmaCoeff;

   return 0;
}



void SetupBCHandler(BCHandler *bcs, Array<int> &dbcs, Vector &dbcv, Array<int> &dbce, Array<int> &nbcs, Vector &nbcv, std::vector<Vector> &e_uniform)
{

   // Parse all BCs
   for (int i = 0; i < dbcs.Size(); i++) // Dirichlet potential
   {
      bcs->AddDirichletBC(dbcv[i], dbcs[i]);
   }

   for (int i = 0; i < dbce.Size(); i++) // Dirichlet E field
   {
      Vector e_uniform_tmp = uebc_const ? e_uniform[0] : e_uniform[i];
      bcs->AddDirichletEFieldBC(e_uniform_tmp, dbce[i]);
   }

   for (int i = 0; i < nbcs.Size(); i++) // Neumann BC
   {
      bcs->AddNeumannBC(nbcv[i], nbcs[i]);
   }
}


std::function<void(const Vector &, DenseMatrix &)> ConductivityMatrix( const Vector &d)
{
   return [d](const Vector &x, DenseMatrix &m)
   {
      // Define dimension of problem
      const int dim = x.Size();

      // Compute Euler angles
      Vector e(3);
      EulerAngles(x,e);
      real_t e1 = e(0);
      real_t e2 = e(1);
      real_t e3 = e(2);

      // Compute rotated matrix
      if (dim == 3)
      {
         // Compute cosine and sine of the angles e1, e2, e3
         const real_t c1 = cos(e1);
         const real_t s1 = sin(e1);
         const real_t c2 = cos(e2);
         const real_t s2 = sin(e2);
         const real_t c3 = cos(e3);
         const real_t s3 = sin(e3);

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
         R.Transpose();
         MultADBt(R, l, R, m);
      }
      else if (dim == 2)
      {
         const real_t c1 = cos(e1);
         const real_t s1 = sin(e1);
         DenseMatrix Rt(2, 2);
         Rt(0, 0) =  c1;
         Rt(0, 1) =  s1;
         Rt(1, 0) = -s1;
         Rt(1, 1) =  c1;
         Vector l(2);
         l(0) = d[0];
         l(1) = d[1];
         MultADAt(Rt,l,m);
      }
      else
      {
         m(0, 0) = d[0];
      }

   };
}

void EulerAngles(const Vector &x, Vector &e)
{
    const int dim = x.Size();

    e(0) = -60.0 * M_PI / 180.0; // convert to radians
    e(1) = 0.0;
    if( dim == 3)
    {
        e(2) = 0.0;
    }
}
