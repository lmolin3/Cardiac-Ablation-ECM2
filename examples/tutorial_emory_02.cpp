//                       MFEM Example 16 - Parallel Version
//
// Compile with: make tutorial_emory_02
//
// Sample runs:  mpirun -np 4 tutorial_emory_02
//               mpirun -np 4 tutorial_emory_02 -m ../data/inline-tri.mesh
//               mpirun -np 4 tutorial_emory_02 -m ../data/disc-nurbs.mesh -tf 2
//               mpirun -np 4 tutorial_emory_02 -s 21 -a 0.0 -k 1.0
//               mpirun -np 4 tutorial_emory_02 -s 22 -a 1.0 -k 0.0
//               mpirun -np 8 tutorial_emory_02 -s 23 -a 0.5 -k 0.5 -o 4
//               mpirun -np 4 tutorial_emory_02 -s 4 -dt 1.0e-4 -tf 4.0e-2 -vs 40
//               mpirun -np 16 tutorial_emory_02 -m ../data/fichera-q2.mesh
//               mpirun -np 16 tutorial_emory_02 -m ../data/fichera-mixed.mesh
//               mpirun -np 16 tutorial_emory_02 -m ../data/escher-p2.mesh
//               mpirun -np 8 tutorial_emory_02 -m ../data/beam-tet.mesh -tf 10 -dt 0.1
//               mpirun -np 4 tutorial_emory_02 -m ../data/amr-quad.mesh -o 4 -rs 0 -rp 0
//               mpirun -np 4 tutorial_emory_02 -m ../data/amr-hex.mesh -o 2 -rs 0 -rp 0
//
// Description:  This example solves a time dependent linear heat equation
//               problem of the form du/dt = K(u) = \nabla \cdot \kappa \nabla u.
//
//               The example demonstrates the use of linear operators (the
//               class ConductionOperator defining K(u)), as well as their
//               implicit time integration. Note that implementing the method
//               ConductionOperator::ImplicitSolve is the only requirement for
//               high-order implicit (SDIRK) time integration.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity k
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   ParBilinearForm *M;
   ParBilinearForm *K;

   HypreParMatrix Mmat;
   HypreParMatrix Kmat;
   HypreParMatrix *T; // T = M + dt K
   real_t current_dt;

   CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec; // Preconditioner for the mass matrix M

   CGSolver T_solver;    // Implicit solver for T = M + dt K
   HypreSmoother T_prec; // Preconditioner for the implicit solver

   std::unique_ptr<Coefficient> kappa = nullptr;

   mutable Vector z; // auxiliary vector

public:
   ConductionOperator(ParFiniteElementSpace &f, real_t kappa_);

   void Mult(const Vector &u, Vector &du_dt) const override;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override;

   ~ConductionOperator() override;
};

real_t InitialTemperature(const Vector &x);

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

   const char *mesh_file = "../data/star.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 2;

   int ode_solver_type = 23;  // SDIRK33Solver
   real_t t_final = 0.5;
   real_t dt = 1.0e-2;
   real_t kappa = 0.5;

   bool paraview = true;
   int vis_steps = 5;
   const char *outfolder = "./Output/";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "Kappa coefficient offset.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for paraview (paraview.llnl.gov) visualization.");
   args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }



   /////////////////////////////////////////////////////////////////////////////
   //------     3. Create serial and parallel mesh
   /////////////////////////////////////////////////////////////////////////////

   //<--- Read the serial mesh from the given mesh file on all processors. We can
   //     handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //     with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //<--- Define the ODE solver used for time integration. Several implicit
   //     singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //     explicit Runge-Kutta methods are available.
   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

   //<--- Refine the mesh in serial to increase the resolution. In this example
   //     we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //     a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   //<--- Define a parallel mesh by a partitioning of the serial mesh. Refine
   //     this mesh further in parallel to increase the resolution. Once the
   //     parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }



   ///////////////////////////////////////////////////////////////////////////////////////////
   //------     4. FE setup: Define finite element space, solution vector, and the operator.
   ///////////////////////////////////////////////////////////////////////////////////////////

   //<--- Define the vector finite element space representing the current and the
   //     initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   HYPRE_BigInt fe_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of temperature unknowns: " << fe_size << endl;
   }

   ParGridFunction u_gf(&fespace);

   //<--- Set the initial conditions for u. All boundaries are considered
   //     natural.
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   //<--- Initialize the conduction operator and the paraview visualization.
   ConductionOperator oper(fespace, kappa);
   u_gf.SetFromTrueDofs(u);



   /////////////////////////////////////////////////////////////////////////////
   //------     5. Define output and visualization
   /////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      out << "Saving the initial temperature field to disk...";

   ParaViewDataCollection pvdc(mesh_file, pmesh);
   pvdc.SetPrefixPath(outfolder);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   if (order > 1)
   {
      pvdc.SetHighOrderOutput(true);
      pvdc.SetLevelsOfDetail(order);
   }   
   pvdc.RegisterField("temperature", &u_gf);
   if (paraview)
   {
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.Save();
   }

   if (Mpi::Root())
      out << " done." << endl;


   ////////////////////////////////////////////////////////////////////////////////////////////////////////
   //------     6. Perform time-integration (looping over the time iterations, ti, with a time-step dt).
   ////////////////////////////////////////////////////////////////////////////////////////////////////////

   ode_solver->Init(oper);
   real_t t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      //--- Advance one time step with the ODE solver. 
      ode_solver->Step(u, t, dt);
      u_gf.SetFromTrueDofs(u);

      //--- Postprocessing and visualization. 
      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t << endl;
         }

         if (paraview)
         {
            pvdc.SetCycle(ti);
            pvdc.SetTime(t);
            pvdc.Save();
         }

      }
   }

   
   
   /////////////////////////////////////////////////////////////////////////////
   //------     7. Cleanup and free memory
   /////////////////////////////////////////////////////////////////////////////
   
   delete pmesh;

   return 0;
}

ConductionOperator::ConductionOperator(ParFiniteElementSpace &f, real_t kappa_)
   : TimeDependentOperator(f.GetTrueVSize(), (real_t) 0.0), fespace(f),
     M(NULL), K(NULL), T(NULL), current_dt(0.0),
     M_solver(f.GetComm()), T_solver(f.GetComm()), z(height)
{
   const real_t rel_tol = 1e-8;

   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble(0); // keep sparsity pattern of M and K the same
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   kappa = std::make_unique<ConstantCoefficient>(kappa_);
   K = new ParBilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
   K->Assemble(0); // keep sparsity pattern of M and K the same
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(100);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);
}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   // Compute:
   //    du_dt = M^{-1}*-Ku
   // for du_dt, where K is linearized by using u from the previous timestep
   Kmat.Mult(u, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, du_dt);
}

void ConductionOperator::ImplicitSolve(const real_t dt,
                                       const Vector &u, Vector &k)
{
   // Solve the equation:
   //    M*k = -K(u + dt*k) for the unknown k=du/dt
   if (!T)
   {
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
   }
   MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt

   // Compute the right-hand side: z = -K(u)
   Kmat.Mult(u, z);
   z.Neg();

   // Solve for k: T k = z
   T_solver.Mult(z, k);
}

ConductionOperator::~ConductionOperator()
{
   delete T;
   delete M;
   delete K;
}

real_t InitialTemperature(const Vector &x)
{
   if (x.Norml2() < 0.5)
   {
      return 2.0;
   }
   else
   {
      return 1.0;
   }
}
