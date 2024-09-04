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
//            ------------------------------------------
//            Heat Miniapp:  Fourier Heat transfer in 2D
//            ------------------------------------------
//
// This example solves a 2D Heat Transfer problem
//
//                            rho c du/dt = Div k Grad T + Q
//
// rho and c are piecewise coefficients,
// k is a piecewise matrix coefficient (isotropic conductivity)
//
//
//
// Boundary conditions consist in:
// * Dirichlet: temperature
// * Neumann: heat flux
// * Robin: convective bc
//
// We discretize the temperature with H1 finite elements.
//
// Sample runs:
//
//    mpirun -np 4 ./convergence -ode 1 -n 20 -a 3 -b 1.2 -kattr 1 -kval 1 -cattr 1 -cval 1 -rhoattr 1 -rhoval 1 --paraview -o 1 -r 5 -fun 2 -d 2
//

#include "../lib/heat_solver.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/stat.h> // Include for mkdir

using namespace std;
using namespace mfem;
using namespace mfem::heat;

// Exact smooth analytic solution for convergence study
double T_exact1(const Vector &x, double t);
void T_grad_exact1(const Vector &x, Vector &gradT);
double f_exact1(const Vector &x);

double T_exact2(const Vector &x, double t);
void T_grad_exact2(const Vector &x, Vector &gradT);
double f_exact2(const Vector &x);

double T_exact3(const Vector &x, double t);
void T_grad_exact3(const Vector &x, Vector &gradT);
double f_exact3(const Vector &x, double t);

// Setting the frequency for the exact solution
double alpha = 3.0;
double beta = 1.2;
double freq = 1.0;
double kappa;

static Vector pw_k(0);     // Piecewise conductivity values
static Vector k_attr(0);   // Domain attributes associated to piecewise Conductivity
static Vector pw_c(0);     // Piecewise heat capacity values
static Vector c_attr(0);   // Domain attributes associated to piecewise heat capacity
static Vector pw_rho(0);   // Piecewise density values
static Vector rho_attr(0); // Domain attributes associated to piecewise density

IdentityMatrixCoefficient *Id = NULL;

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

   // Analytic problem
   int fun = 1;
   int dim = 2;
   // Mesh
   int element = 0;
   int order = 1;
   int total_refinements = 0;
   // Time integrator
   int ode_solver_type = -1;
   int num_steps = 20;
   // Postprocessing
   bool paraview = true;
   const char *outfolder = "./Output/Convergence/";

   OptionsParser args(argc, argv);
   args.AddOption(&element, "-e", "--element-type",
                  "Element type (quads=1 or triangles=0).");
   args.AddOption(&dim, "-d", "--dimension",
                  "Dimension of the problem (2 or 3).");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&total_refinements, "-r", "--refinements",
                  "Number of total uniform refinements");
   args.AddOption(&ode_solver_type, "-ode", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   4 - Implicit Midpoint, 5 - SDIRK23, 6 - SDIRK34,\n\t"
                  "\t   7 - Forward Euler, 8 - RK2, 9 - RK3 SSP, 10 - RK4.");
   args.AddOption(&num_steps, "-n", "--num-steps",
                  "Number of time steps.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "First parameter for the exact solution.");
   args.AddOption(&beta, "-b", "--beta",
                  "Second parameter for the exact solution.");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency for the exact solution.");
   args.AddOption(&k_attr, "-kattr", "--kappa-attributes",
                  "Domain attributes associated to piecewise Thermal Conductivity");
   args.AddOption(&pw_k, "-kval", "--piecewise-kappa",
                  "Piecewise values of Thermal Conductivity");
   args.AddOption(&c_attr, "-cattr", "--c-attributes",
                  "Domain attributes associated to piecewise Heat Capacity");
   args.AddOption(&pw_c, "-cval", "--piecewise-c",
                  "Piecewise values of Heat Capacity");
   args.AddOption(&rho_attr, "-rhoattr", "--rho-attributes",
                  "Domain attributes associated to piecewise Density");
   args.AddOption(&pw_rho, "-rhoval", "--piecewise-rho",
                  "Piecewise values of Density");
   args.AddOption(&paraview, "-paraview", "--paraview",
                  "-no-paraview", "--no-paraview",
                  "Enable/Disable Paraview output.");
   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder for Paraview files.");
   args.AddOption(&fun, "-fun", "--function",
                  "Function to use for the exact solution: 1 - 2nd ord Polynomial, 2 - Trigonometric");

   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }

   kappa = freq * M_PI;

   // Determine final time
   double t_initial = 0.0;
   double t_final = 2.0;
   double dt = (t_final - t_initial) / num_steps;

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

   double l2_err_prev = 0.0;
   double h1_err_prev = 0.0;
   double h_prev = 0.0;

   // Refinement loop
   for (int serial_ref_levels = 0; serial_ref_levels < total_refinements; serial_ref_levels++)
   {

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 3. Read Mesh and create parallel
      ///////////////////////////////////////////////////////////////////////////////////////////////

      // Read the (serial) mesh from the given mesh file on all processors.  We
      // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
      // and volume meshes with the same code.

      Mesh *mesh = nullptr;

      switch (dim)
      {
      case 2:
      {
         if (element == 0)
         {
            mesh = new Mesh("../../data/inline-tri.mesh");
         }
         else
         {
            mesh = new Mesh("../../data/inline-quad.mesh");
         }
         break;
      }
      case 3:
      {
         if (element == 0)
         {
            mesh = new Mesh("../../data/inline-tet.mesh");
         }
         else
         {
            mesh = new Mesh("../../data/inline-hex.mesh");
         }
         break;
      }
      default:
         break;
      }
      mesh->EnsureNodes();
      int sdim = mesh->SpaceDimension();

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

      // Parse the analytic problem
      // Set the initial condition.
      FunctionCoefficient *Tex_coeff = nullptr;
      FunctionCoefficient *f_exact_coeff = nullptr;
      VectorFunctionCoefficient *gradTex_coeff = nullptr;

      switch (fun)
      {
      case 1:
      {
         Tex_coeff = new FunctionCoefficient(T_exact1);
         gradTex_coeff = new VectorFunctionCoefficient(sdim, T_grad_exact1);
         f_exact_coeff = new FunctionCoefficient(f_exact1);
         break;
      }
      case 2:
      {
         Tex_coeff = new FunctionCoefficient(T_exact2);
         gradTex_coeff = new VectorFunctionCoefficient(sdim, T_grad_exact2);
         f_exact_coeff = new FunctionCoefficient(f_exact2);
         break;
      }
      case 3:
      {
         Tex_coeff = new FunctionCoefficient(T_exact3);
         gradTex_coeff = new VectorFunctionCoefficient(sdim, T_grad_exact3);
         f_exact_coeff = new FunctionCoefficient(f_exact3);
         break;
      }
      default:
         break;
      }

      // Conductivity
      Id = new IdentityMatrixCoefficient(sdim);

      Array<int> attr(0);
      Array<MatrixCoefficient *> coefs(0);

      MFEM_ASSERT(pw_k.Size() == k_attr.Size(), "Size mismatch between conductivity values and attributes");

      for (int i = 0; i < pw_k.Size(); i++)
      {
         MFEM_ASSERT(k_attr[i] <= pmesh->attributes.Max(), "Attribute value out of range");

         MatrixCoefficient *tmp = pw_k[i] != 0 ? new ScalarMatrixProductCoefficient(pw_k[i], *Id) : NULL;
         coefs.Append(tmp);
         attr.Append(k_attr[i]);
      }

      PWMatrixCoefficient *Kappa = new PWMatrixCoefficient(sdim, attr, coefs);

      // Heat Capacity
      attr.SetSize(0);
      Array<Coefficient *> coefs_c(0);

      MFEM_ASSERT(pw_c.Size() == c_attr.Size(), "Size mismatch between heat capacity values and attributes");

      for (int i = 0; i < pw_c.Size(); i++)
      {
         MFEM_ASSERT(c_attr[i] <= pmesh->attributes.Max(), "Attribute value out of range");

         ConstantCoefficient *tmp = pw_c[i] != 0 ? new ConstantCoefficient(pw_c[i]) : NULL;
         coefs_c.Append(tmp);
         attr.Append(c_attr[i]);
      }

      PWCoefficient *c = new PWCoefficient(attr, coefs_c);

      // Density
      attr.SetSize(0);
      Array<Coefficient *> coefs_rho(0);

      MFEM_ASSERT(pw_rho.Size() == rho_attr.Size(), "Size mismatch between density values and attributes");

      for (int i = 0; i < pw_rho.Size(); i++)
      {
         MFEM_ASSERT(rho_attr[i] <= pmesh->attributes.Max(), "Attribute value out of range");

         ConstantCoefficient *tmp = pw_rho[i] != 0 ? new ConstantCoefficient(pw_rho[i]) : NULL;
         coefs_rho.Append(tmp);
         attr.Append(rho_attr[i]);
      }

      PWCoefficient *rho = new PWCoefficient(attr, coefs_rho);

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 5. Set up boundary conditions
      ///////////////////////////////////////////////////////////////////////////////////////////////

      // Default values for Dirichlet BCs
      // If values for Dirichlet BCs were not set assume they are zero
      Array<int> dbcs(pmesh->bdr_attributes.Max());
      dbcs = 1;

      // Create BCHandler and parse bcs
      // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
      bool verbose = false;
      BCHandler *bcs = new BCHandler(pmesh, verbose); // Boundary conditions handler

      switch (fun)
      {
      case 1:
      {
         bcs->AddDirichletBC(T_exact1, dbcs);
         break;
      }
      case 2:
      {
         bcs->AddDirichletBC(T_exact2, dbcs);
         break;
      }
      case 3:
      {
         bcs->AddDirichletBC(T_exact3, dbcs);
         break;
      }
      default:
         break;
      }

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 6. Create the Heat Solver
      ///////////////////////////////////////////////////////////////////////////////////////////////

      // Create the Heat solver
      verbose = false;
      HeatSolver Heat(pmesh, order, bcs, Kappa, c, rho, ode_solver_type, verbose);

      int attr_volumetric = 1;
      Heat.AddVolumetricTerm(f_exact_coeff, attr_volumetric);

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 7. Setup solver and Assemble forms
      ///////////////////////////////////////////////////////////////////////////////////////////////

      Heat.EnablePA(false);
      Heat.Setup();

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 8. Perform time-integration (looping over the time iterations, step, with a
      //     time-step dt).
      ///////////////////////////////////////////////////////////////////////////////////////////////

      // Get reference to the temperature vector and gridfunction internal to Heat
      ParFiniteElementSpace *fespace = Heat.GetFESpace();

      Vector &T = Heat.GetTemperature();
      ParGridFunction &T_gf = Heat.GetTemperatureGf();
      ParGridFunction *T_exact_gf = new ParGridFunction(fespace);

      Tex_coeff->SetTime(0.0);
      T_gf.ProjectCoefficient(*Tex_coeff);
      T_exact_gf->ProjectCoefficient(*Tex_coeff);

      // Initialize Paraview visualization
      std::string name = "Heat-MMS-ref" + std::to_string(serial_ref_levels);
      ParaViewDataCollection paraview_dc(name, pmesh.get());

      if (paraview)
      {
         paraview_dc.SetPrefixPath(outfolder);
         Heat.RegisterParaviewFields(paraview_dc);
         Heat.AddParaviewField("T_exact", T_exact_gf);
         Heat.WriteFields(0, 0.0);
      }

      // Set output options and print header
      cout.precision(4);

      // Set initial temperature
      Heat.SetInitialTemperature(T_gf);

      // Time-stepping loop
      double t = 0.0;
      bool last_step = false;
      for (int step = 1; !last_step; step++)
      {
         if (t + dt >= t_final - dt / 2)
         {
            last_step = true;
         }

         Heat.Step(t, dt, step);

         if (paraview)
         {
            Tex_coeff->SetTime(t);
            gradTex_coeff->SetTime(t);
            T_exact_gf->ProjectCoefficient(*Tex_coeff);
            Heat.WriteFields(step, t);
         }
      }

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 9. Compute and print the L^2 and H^1 norms of the error.
      ///////////////////////////////////////////////////////////////////////////////////////////////

      double l2_err = 0.0;
      double h1_err = 0.0;
      double l2_rate = 0.0;
      double h1_rate = 0.0;
      l2_err = T_gf.ComputeL2Error(*Tex_coeff);
      h1_err = T_gf.ComputeH1Error(Tex_coeff, gradTex_coeff);

      double h_min = 0.0;
      double h_max = 0.0;
      double kappa_min = 0.0;
      double kappa_max = 0.0;
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
         cout << setw(16) << Heat.GetProblemSize() << setw(16) << h_min << setw(16) << l2_err << setw(16) << l2_rate;
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

      for (int i = 0; i < coefs_c.Size(); i++)
      {
         delete coefs_c[i];
      }

      for (int i = 0; i < coefs_rho.Size(); i++)
      {
         delete coefs_rho[i];
      }

      delete Kappa;
      delete c;
      delete rho;
      delete Id;

      delete gradTex_coeff;
      delete Tex_coeff;

      delete T_exact_gf;
   }
}

double T_exact1(const Vector &x, double t)
{
   double T = 0.0;
   if (x.Size() == 2)
   {
      // T = 1 + x^2 + alpha y^2 + beta t
      T = 1.0 + x(0) * x(0) + alpha * x(1) * x(1) + beta * t;
   }
   else if (x.Size() == 3)
   {
      // T = 1 + x^2 + alpha y^2 + beta z^2 + beta t
      T = 1.0 + x(0) * x(0) + alpha * x(1) * x(1) + alpha * x(2) * x(2) + beta * t;
   }
   else
   {
      mfem_error("This test works only for 2D or 3D");
   }

   return T;
}

void T_grad_exact1(const Vector &x, Vector &gradT)
{
   if (x.Size() == 2)
   {
      // gradT = [2x, 2alpha y]
      gradT(0) = 2.0 * x(0);
      gradT(1) = 2.0 * alpha * x(1);
   }
   else if (x.Size() == 3)
   {
      // gradT = [2x, 2alpha y, 2alpha z]
      gradT(0) = 2.0 * x(0);
      gradT(1) = 2.0 * alpha * x(1);
      gradT(2) = 2.0 * alpha * x(2);
   }
   else
   {
      mfem_error("This test works only for 2D or 3D");
   }
}

double f_exact1(const Vector &x)
{
   double f = 0.0;
   if (x.Size() == 2)
   {
      // beta - 2 - 2 alpha
      f = beta - 2.0 - 2.0 * alpha;
   }
   else if (x.Size() == 3)
   {
      // beta - 2 - 2 alpha - 2 alpha
      f = beta - 2.0 - 2.0 * alpha - 2.0 * alpha;
   }
   else
   {
      mfem_error("This test works only for 2D or 3D");
   }

   return f;
}

double T_exact2(const Vector &x, double t)
{
   double T = 0.0;
   if (x.Size() == 2)
   {
      // T = sin(kappa x) sin(kappa y) + beta t
      T = sin(kappa * x(0)) * sin(kappa * x(1)) + beta * t;
   }
   else if (x.Size() == 3)
   {
      // T = sin(kappa x) sin(kappa y) sin(kappa z) + beta t
      T = sin(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2)) + beta * t;
   }
   else
   {
      mfem_error("This test works only for 2D or 3D");
   }

   return T;
}

void T_grad_exact2(const Vector &x, Vector &gradT)
{
   if (x.Size() == 2)
   {
      // gradT = [kappa cos(kappa x) sin(kappa y), kappa sin(kappa x) cos(kappa y)]
      gradT(0) = kappa * cos(kappa * x(0)) * sin(kappa * x(1));
      gradT(1) = kappa * sin(kappa * x(0)) * cos(kappa * x(1));
   }
   else if (x.Size() == 3)
   {
      // gradT = [kappa cos(kappa x) sin(kappa y) sin(kappa z), kappa sin(kappa x) cos(kappa y) sin(kappa z), kappa sin(kappa x) sin(kappa y) cos(kappa z)]
      gradT(0) = kappa * cos(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2));
      gradT(1) = kappa * sin(kappa * x(0)) * cos(kappa * x(1)) * sin(kappa * x(2));
      gradT(2) = kappa * sin(kappa * x(0)) * sin(kappa * x(1)) * cos(kappa * x(2));
   }
   else
   {
      mfem_error("This test works only for 2D or 3D");
   }
}

double f_exact2(const Vector &x)
{
   double f = 0.0;
   if (x.Size() == 2)
   {
      // beta + 2 kappa^2 sin(kappa x) sin(kappa y)
      f = beta + 2.0 * kappa * kappa * (sin(kappa * x(0)) * sin(kappa * x(1)));
   }
   else if (x.Size() == 3)
   {
      // beta + 3 kappa^2 sin(kappa x) sin(kappa y) sin(kappa z)
      f = beta + 3.0 * kappa * kappa * (sin(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2)));
   }
   else
   {
      mfem_error("This test works only for 2D or 3D");
   }

   return f;
}

double T_exact3(const Vector &x, double t)
{
   double T = 0.0;
   if (x.Size() == 2)
   {
      // T = sin(kappa x) sin(kappa y) + beta t^2
      T = sin(kappa * x(0)) * sin(kappa * x(1)) + beta * pow(t, 2);
   }
   else if (x.Size() == 3)
   {
      // T = sin(kappa x) sin(kappa y) sin(kappa z) + beta t^2
      T = sin(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2)) + beta * pow(t, 2);
   }
   else
   {
      mfem_error("This test works only for 2D or 3D");
   }

   return T;
}

void T_grad_exact3(const Vector &x, Vector &gradT)
{
   if (x.Size() == 2)
   {
      // gradT = [kappa cos(kappa x) sin(kappa y), kappa sin(kappa x) cos(kappa y)]
      gradT(0) = kappa * cos(kappa * x(0)) * sin(kappa * x(1));
      gradT(1) = kappa * sin(kappa * x(0)) * cos(kappa * x(1));
   }
   else if (x.Size() == 3)
   {
      // gradT = [kappa cos(kappa x) sin(kappa y) sin(kappa z), kappa sin(kappa x) cos(kappa y) sin(kappa z), kappa sin(kappa x) sin(kappa y) cos(kappa z)]
      gradT(0) = kappa * cos(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2));
      gradT(1) = kappa * sin(kappa * x(0)) * cos(kappa * x(1)) * sin(kappa * x(2));
      gradT(2) = kappa * sin(kappa * x(0)) * sin(kappa * x(1)) * cos(kappa * x(2));
   }
   else
   {
      mfem_error("This test works only for 2D or 3D");
   }
}

double f_exact3(const Vector &x, double t)
{
   double f = 0.0;
   if (x.Size() == 2)
   {
      // beta + 2 kappa^2 sin(kappa x) sin(kappa y)
      f = 2.0 * beta * t + 2.0 * kappa * kappa * (sin(kappa * x(0)) * sin(kappa * x(1)));
   }
   else if (x.Size() == 3)
   {
      // 2 beta t + 3 kappa^2 sin(kappa x) sin(kappa y) sin(kappa z)
      f = 2.0 * beta * t + 3.0 * kappa * kappa * (sin(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2)));
   }
   else
   {
      mfem_error("This test works only for 2D or 3D");
   }

   return f;
}
