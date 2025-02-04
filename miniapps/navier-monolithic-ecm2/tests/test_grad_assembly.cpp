#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   // 2. Parse command line options.
   string mesh_file = "../../data/star.mesh";
   int order = 1;

   // 3. Read the serial mesh from the given mesh file.
   Mesh serial_mesh(mesh_file);

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh once in parallel to increase the resolution.
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear(); // the serial mesh is no longer needed
   mesh.UniformRefinement();

   // 5. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection pfec(order, mesh.Dimension());
   H1_FECollection ufec(order+1, mesh.Dimension());
   ParFiniteElementSpace pfes(&mesh, &pfec);
   ParFiniteElementSpace ufes(&mesh, &ufec, mesh.Dimension());
   HYPRE_BigInt total_num_dofs_vel = ufes.GlobalTrueVSize();
   HYPRE_BigInt total_num_dofs_pres = pfes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << total_num_dofs_vel << endl;
      cout << "Number of pressure unknowns: " << total_num_dofs_pres << endl;
   }


   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   ParMixedBilinearForm gform1(&pfes, &ufes);
   ParMixedBilinearForm gform2(&pfes, &ufes);
   ParMixedBilinearForm dform(&ufes, &pfes);

   OperatorHandle opG1, opG2, opDiff, opD, opG3;

   Array<int> empty;
   int skip_zeros = 0;
   gform1.AddDomainIntegrator(new GradientIntegrator);
   gform1.Assemble(skip_zeros);
   gform1.Finalize();
   gform1.FormRectangularSystemMatrix(empty, empty, opG1);

   ConstantCoefficient negone(-1.0);
   gform2.AddDomainIntegrator(new TransposeIntegrator(new VectorDivergenceIntegrator(negone)));
   gform2.Assemble(skip_zeros);
   gform2.Finalize();
   gform2.FormRectangularSystemMatrix(empty, empty, opG2);

   ConstantCoefficient one(1.0);
   dform.AddDomainIntegrator(new VectorDivergenceIntegrator);
   dform.Assemble(skip_zeros);
   dform.Finalize();
   dform.FormRectangularSystemMatrix(empty, empty, opD);
   opG3.Reset(opD.As<HypreParMatrix>()->Transpose());
   *opG3.As<HypreParMatrix>() *= -1.0;

   // 8. Check that the two operators are the same.
   opDiff.Reset(Add(1.0, *opG1.As<HypreParMatrix>(), -1.0, *opG2.As<HypreParMatrix>()),  true);
   real_t norm = opDiff.As<HypreParMatrix>()->FNorm();

   Vector x(total_num_dofs_pres);
   Vector y1(total_num_dofs_vel);
   Vector y2(total_num_dofs_vel);
   x.Randomize(1);

   opG1.As<HypreParMatrix>()->Mult(x, y1);
   opG2.As<HypreParMatrix>()->Mult(x, y2);
   y1 -= y2;
   real_t loc_norm2 = y1.Norml2();
   real_t glob_norm2;
   MPI_Allreduce(&loc_norm2, &glob_norm2, 1, MPI_DOUBLE, MPI_SUM, pfes.GetComm());


   if (Mpi::Root())
   {
      cout << "|| G1 - G2 ||_F: " << norm << endl;
      cout << "|| G1 x - G2 x ||_2: " << glob_norm2 << endl;
   }  

  // 9. Print matrices to file
   if (num_procs == 1)
   {
      ofstream opD_file("D.dat");
      ofstream G1_file("G1.dat");
      ofstream G2_file("G2.dat");
      ofstream G3_file("G3.dat");
      ofstream y1_file("y1.dat");
      ofstream y2_file("y2.dat");

      opG1.As<HypreParMatrix>()->PrintMatlab(G1_file);
      opG2.As<HypreParMatrix>()->PrintMatlab(G2_file);
      opD.As<HypreParMatrix>()->PrintMatlab(opD_file);
      opG3.As<HypreParMatrix>()->PrintMatlab(G3_file);
      y1.Print(y1_file);
      y2.Print(y2_file);
   }

   return 0;
}
