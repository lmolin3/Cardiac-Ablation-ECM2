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
//            RF Miniapp:  Simple Electrostatics Simulation Code
//            -----------------------------------------------------
//
// This miniapp solves a sconvergence analysis for the 2D electrostatic problem (Quasi-static Maxwell).
//
//                            Div sigma Grad Phi = 0
//
// Boundary conditions consist in:
// * Dirichlet: manufactured solution
//
// We discretize the electric potential with H1 finite elements.
// The electric field E is discretized with H1 finite elements (just for visualization/postprocessing).
//
// Sample runs:
//
//   A cylinder at constant RF_solverge in a square, grounded metal pipe:
//      mpirun -np 4 ./convergence -o 1 -sattr '1' -sval '1.0' -rs 0
//

#include <mfem.hpp>

#include "../lib/electrostatics_solver.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/stat.h> // Include for mkdir

using namespace std;
using namespace mfem;
using namespace mfem::electrostatics;

static Vector pw_sigma(0);   // Piecewise conductivity values
static Vector sigma_attr(0); // Domain attributes associated to piecewise Conductivity

IdentityMatrixCoefficient *Id = NULL;


// Setting the frequency for the exact solution
real_t freq = 1.0;
real_t kappa;

// Exact smooth analytic solution for convergence study
real_t Phi_exact(const Vector &, real_t );
void Phi_grad_exact(const Vector &, Vector &);
real_t f_exact(const Vector &);


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

   int element = 0;
   int order = 1;
   int serial_ref_levels = 0;
   bool paraview = false;
   const char *outfolder = "./Output/Convergence/2D";

   OptionsParser args(argc, argv);
   args.AddOption(&element, "-e", "--element-type",
                  "Element type (quads=1 or triangles=0).");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&sigma_attr, "-sattr", "--sigma-attributes",
                  "Domain attributes associated to piecewise Conductivity");
   args.AddOption(&pw_sigma, "-sval", "--piecewise-sigma",
                  "Piecewise values of Conductivity");
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

   kappa = freq * M_PI;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Read Mesh and create parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   Mesh *mesh = nullptr;

   if (element == 0)
   {
      mesh = new Mesh("../../data/inline-tri.mesh");
   }
   else
   {
      mesh = new Mesh("../../data/inline-quad.mesh");
   }
   mesh->EnsureNodes();
   int sdim = mesh->SpaceDimension();

   StopWatch sw_initialization;
   sw_initialization.Start();
   if (Mpi::Root())
   {
      cout << "Starting initialization." << endl;
   }

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. NURBS meshes are
   // refined at least twice, as they are typically coarse.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   auto pmesh = make_shared<ParMesh>(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Make sure tet-only meshes are marked for local refinement.
   pmesh->Finalize(true);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Set up coefficients
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Id = new IdentityMatrixCoefficient(sdim);

   int d = pmesh->Dimension();

   Array<int> attr(0);
   Array<MatrixCoefficient *> coefs(0);

   MFEM_ASSERT(pw_sigma.Size() == sigma_attr.Size(), "Size mismatch between conductivity values and attributes");

   for (int i = 0; i < pw_sigma.Size(); i++)
   {
      MFEM_ASSERT(sigma_attr[i] <= pmesh->attributes.Max(), "Attribute value out of range");

      MatrixCoefficient *tmp = pw_sigma[i] != 0 ? new ScalarMatrixProductCoefficient(pw_sigma[i], *Id) : NULL;
      coefs.Append(tmp);
      attr.Append(sigma_attr[i]);
   }

   PWMatrixCoefficient *sigmaCoeff = new PWMatrixCoefficient(d, attr, coefs);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Set up boundary conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Array<int> dbcs(pmesh->bdr_attributes.Max());
   dbcs = 1;

   // Create BCHandler and parse bcs
   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool verbose = true;
   BCHandler *bcs = new BCHandler(pmesh, verbose); // Boundary conditions handler
   bcs->AddDirichletBC(Phi_exact, dbcs);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Create the Electrostatics Solver
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Create the Electrostatic solver
   ElectrostaticsSolver RF_solver(pmesh, order, bcs, sigmaCoeff, verbose);
   RF_solver.display_banner(std::cout);

   if ((mkdir(outfolder, 0777) == -1) && Mpi::Root())
   {
      mfem::err << "Error :  " << strerror(errno) << std::endl;
   }

   // Get reference to the potential vector and gridfunction internal to RF_solver
   ParFiniteElementSpace *fespace = RF_solver.GetFESpace();

   FunctionCoefficient Phiex_coeff(Phi_exact);
   VectorFunctionCoefficient gradPhiex_coeff(sdim, Phi_grad_exact);

   ParGridFunction &Phi_gf = RF_solver.GetPotential();
   ParGridFunction *Phi_exact_gf = new ParGridFunction(fespace);

   Phi_gf.ProjectCoefficient(Phiex_coeff);
   Phi_exact_gf->ProjectCoefficient(Phiex_coeff);

   // Initialize Paraview visualization
   ParaViewDataCollection paraview_dc("RF_solver-MMS", pmesh.get());

   if (paraview)
   {
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetPrefixPath(outfolder);
         paraview_dc.SetLevelsOfDetail(order);
         RF_solver.RegisterParaviewFields(paraview_dc);
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
   RF_solver.PrintSizes();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Add the volumetric term to the right-hand side
   auto *f_exact_coeff = new FunctionCoefficient(f_exact); // Note: RF_solverSolver uses CoeffContainer which takes ownership of f_exact_coeff, so it should be created with new()
   int attr_volumetric = 1;
   RF_solver.AddVolumetricTerm(f_exact_coeff, attr_volumetric);  

   // Setup solver and Assemble all forms
   RF_solver.Setup();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Solve
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Solve the system and compute any auxiliary fields
   RF_solver.Solve();

   ParGridFunction E_gf = RF_solver.GetElectricField();
   real_t el = RF_solver.ElectricLosses(E_gf);

   // Determine the current size of the linear system
   int prob_size = RF_solver.GetProblemSize();

   // Write fields to disk for VisIt
   if (paraview)
   {
      RF_solver.WriteFields();
   }

   // Send the solution by socket to a GLVis server.
   RF_solver.PrintTimingData();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Compute and print the L^2 and H^1 norms of the error.
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Set output options and print header
   cout.precision(4);

   if (Mpi::Root())
   {
      cout << "----------------------------------------------------------------------------------------"
           << endl;
      cout << left << setw(16) << "DOFs " << setw(16) << "h " << setw(16) << "L^2 error " << setw(16);
      cout << "L^2 rate " << setw(16) << "H^1 error " << setw(16) << "H^1 rate" << endl;
      cout << "----------------------------------------------------------------------------------------"
           << endl;
   }

   real_t l2_err = 0.0;
   real_t h1_err = 0.0;
   real_t l2_rate = 0.0;
   real_t h1_rate = 0.0;
   l2_err = Phi_gf.ComputeL2Error(Phiex_coeff);
   h1_err = Phi_gf.ComputeH1Error(&Phiex_coeff, &gradPhiex_coeff);

   real_t h_min = 0.0;
   real_t h_max = 0.0;
   real_t kappa_min = 0.0;
   real_t kappa_max = 0.0;
   pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   ///*if (serial_ref_levels != 0)
   //{
   //  l2_rate = log(l2_err/l2_err_prev) / log(h_min/h_prev);
   //   h1_rate = log(h1_err/h1_err_prev) / log(h_min/h_prev);
   // }
   // else
   //{
   //  l2_rate = 0.0;
   //   h1_rate = 0.0;
   //}

   // l2_err_prev = l2_err;
   // h1_err_prev = h1_err;
   // h_prev = h_min;

   if (Mpi::Root())
   {
      cout << setw(16) << RF_solver.GetProblemSize() << setw(16) << h_min << setw(16) << l2_err << setw(16) << l2_rate;
      cout << setw(16) << h1_err << setw(16) << h1_rate << endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 10. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Delete the MatrixCoefficient objects at the end of main
   for (int i = 0; i < coefs.Size(); i++)
   {
      delete coefs[i];
   }

   delete sigmaCoeff;
   delete Id;

   delete Phi_exact_gf;

   return 0;
}


real_t Phi_exact(const Vector &x, real_t t)
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

void Phi_grad_exact(const Vector &x, Vector &gradPhi)
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
      f = 3.0 * kappa * kappa * (sin(kappa * x(0)) * sin(kappa * x(1)) * sin(
                                    kappa * x(2)));
   }

   return f;
}
