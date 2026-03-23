//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Poisson
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../miniapps/multidomain/multidomain-hex.mesh";
   int order = 1;
   int serial_ref_levels = 0;
   bool monolithic = true;
   real_t scaling = 1e5;
   string filename = "A.mtx";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels", "Number of serial refinement levels.");
   args.AddOption(&filename, "-f", "--filename", "Filename for the output matrix.");
   args.AddOption(&scaling, "-s", "--scaling", "Scaling factor for the conductivity coefficient.");
   args.AddOption(&monolithic, "-m", "--monolithic", "-nm", "--no-monolithic", "Monolithic piecewise problem or separate problems.");
   args.ParseCheck();

   if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".mtx")
   {
      filename += ".mtx";
   }

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }
   
   if (monolithic)
   {
      //    Define a finite element space on the mesh. Here we use H1 continuous
      //    high-order Lagrange finite elements of the given order.
      H1_FECollection fec(order, mesh.Dimension());
      FiniteElementSpace fespace(&mesh, &fec);
      cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

      // Define parameters for the piecewise conductivity coefficient
      Vector sigma(mesh.attributes.Max());
      sigma = 1.0;
      sigma(0) = sigma(1) * scaling;
      PWConstCoefficient sigma_func(sigma);

      //    Set up the bilinear form a(.,.) corresponding to the -Delta operator.
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(sigma_func));
      a.Assemble();

      // Assemble matrix
      SparseMatrix A;
      Array<int> ess_dofs(1);
      ess_dofs[0] = 1; // Eliminate the first DOF to make the matrix non-singular
      a.FormSystemMatrix(ess_dofs, A);

      // Export A
      ofstream A_ofs(filename);
      A_ofs.precision(16);
      A.PrintMatlab(A_ofs);
      A_ofs.close();
   }

   return 0;
}
