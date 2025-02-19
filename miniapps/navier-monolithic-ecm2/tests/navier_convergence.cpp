// Navier MMS example
//
// A manufactured solution is defined as
//
// u = [pi * sin(t) * sin(pi * x)^2 * sin(2 * pi * y),
//      -(pi * sin(t) * sin(2 * pi * x)) * sin(pi * y)^2].
//
// p = cos(pi * x) * sin(t) * sin(pi * y)
//
// The solution is used to compute the symbolic forcing term (right hand side),
// of the equation. Then the numerical solution is computed and compared to the
// exact manufactured solution to determine the error.
//
// Boundary markers for the square mesh are:
// Bottom=1, Right=2, Top=3, Left=4
//
// Sample run:
//
// 1. Yosida block preconditioner + ApproximateDiscreteLaplacian Schur complement preconditioner
// mpirun -np 4 ./navier-convergence-monolithic -d 2 -e 1 -n 10 -r 3 -ou 2 -op 1 -f 1 -bcs 1 --preconditioner 4 --schur-preconditioner 5
// 
// 2. Mass lumping
// mpirun -np 4 ./navier-convergence-monolithic -d 2 -e 1 -n 10 -r 3 -ou 2 -op 1 -f 1 -bcs 1 --preconditioner 4 --schur-preconditioner 5 --mass-lumping 
//
// 3. Stiff strain
// mpirun -np 4 ./navier-convergence-monolithic -d 2 -e 1 -n 10 -r 3 -ou 2 -op 1 -f 1 -bcs 1 --preconditioner 4 --schur-preconditioner 5 --stiff-strain
//

#include "lib/navier_solver.hpp"
#include <fstream>
#include <sys/stat.h>  // Include for mkdir

#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979
#endif

using namespace std;
using namespace mfem;
using namespace navier;


struct s_NavierContext // Navier Stokes params
{
   int uorder = 2;
   int porder = 2;
   real_t kinvis = 0.01;
   real_t a = 2.0;
   real_t dt = 1e-3;
   real_t t_final = 10 * dt;
   bool verbose = false;
   int fun = 1;
   int bdf = 1;
   int bcs = 0; // 0 = FullyDirichlet, 1 = FullyNeumann, 2 = Mixed
   // int splitting_type = 0;  // 0 = Chorin-Temam, 1 = Yosida, 2 = High-Order Yosida 
   // int correction_order = 1; // Correction order for High-Order Yosida   
   BlockPreconditionerType pc_type = BlockPreconditionerType::BLOCK_DIAGONAL;       // 0: Block Diagonal, 1: BlowLowerTri, 2: BlockUpperTri, 3: Chorin-Temam, 4: Yosida, 5: Chorin-Temam Pressure Corrected, 6: Yosida Pressure Corrected
   SchurPreconditionerType schur_pc_type = SchurPreconditionerType::APPROXIMATE_DISCRETE_LAPLACIAN; // 0: Pressure Mass, 1: Pressure Laplacian, 2: PCD, 3: Cahouet-Chabard, 4: LSC, 5: Approximate Inverse
   TimeAdaptivityType time_adaptivity_type = TimeAdaptivityType::NONE; // Time adaptivity type (NONE, CFL, HOPC)
   int pressure_correction_order = 2; // Order of the pressure correction
   bool mass_lumping = false; // Enable mass lumping
   bool stiff_strain = false; // false: viscous stress (Δu), true: stiff strain ( ∇u + ∇u^T )
} NS_ctx;

struct s_MeshContext // mesh
{
   int n = 10;                
   int dim = 2;
   int elem = 0;
} Mesh_ctx;


// Forward declarations of functions
void vel1(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);

   u(0) = M_PI * sin(t) * pow(sin(M_PI * xi), 2.0) * sin(2.0 * M_PI * yi);
   u(1) = -(M_PI * sin(t) * sin(2.0 * M_PI * xi) * pow(sin(M_PI * yi), 2.0));
}

real_t p1(const Vector &x, real_t t)
{
   real_t xi = x(0);
   real_t yi = x(1);

   return sin(t) * cos(M_PI * xi) * sin(M_PI * yi);
}

void accel1(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);

   u(0) =   M_PI * pow(sin(M_PI * xi), 2.0) * sin(2 * M_PI * yi) * cos(t)                                                        // dudt
          - 2.0 * NS_ctx.kinvis * pow(M_PI, 3.0) * sin(2.0 * M_PI * yi) * sin(t) * ( 2 * cos(2.0 * M_PI * xi) - 1.0 )            // - nu lap u
          - M_PI * sin( M_PI * xi) * sin( M_PI * yi) * sin(t)                                                                    // grad p 
          + 4.0 * pow(M_PI, 3.0) * cos( M_PI * xi) * pow(sin(M_PI * xi), 3.0) * pow(sin(M_PI * yi), 2.0) * pow(sin(t), 2.0);     // u grad u

   u(1) = - M_PI * pow(sin(M_PI * yi), 2.0) * sin(2 * M_PI * xi) * cos(t)                                                        // dudt
          + 2.0 * NS_ctx.kinvis * pow(M_PI, 3.0) * sin(2.0 * M_PI * xi) * sin(t) * ( 2 * cos(2.0 * M_PI * yi) - 1.0 )            // - nu lap u
          + M_PI * cos( M_PI * xi) * cos( M_PI * yi) * sin(t)                                                                    // grad p 
          + 4.0 * pow(M_PI, 3.0) * cos( M_PI * yi) * pow(sin(M_PI * yi), 3.0) * pow(sin(M_PI * xi), 2.0) * pow(sin(t), 2.0);     // u grad u 
}


// MMS2
void vel2(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);

   u(0) = sin(xi) * sin(yi+t) ;
   u(1) = cos(xi) * cos(yi+t) ;
}

real_t p2(const Vector &x, real_t t)
{
   real_t xi = x(0);
   real_t yi = x(1);

   return cos(xi) * sin(yi+t);
}

void accel2(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);

   u(0) =   cos(t + yi)*sin(xi)                // dudt
          + 2.0 * NS_ctx.kinvis * sin(t + yi)  // - nu lap u
          - sin(t + yi)*sin(xi)                // grad p 
          + sin(2.0*xi)/2.0;                   // u grad u

   u(1) =   cos(t + yi)*cos(xi)                        // dudt
          + 2.0 * NS_ctx.kinvis * cos(t + yi)*cos(xi)  // - nu lap u
          - sin(t + yi)*cos(xi)                        // grad p 
          - sin(2.0*t + 2.0*yi)/2.0;                   // u grad u
}

// Kim & Moin
void vel3(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);

   u(0) = -cos(NS_ctx.a*M_PI*xi) * sin(NS_ctx.a*M_PI*yi) * std::exp(-2.0*pow(NS_ctx.a,2.0)*pow(M_PI, 2.0)* NS_ctx.kinvis*t) ;
   u(1) =  sin(NS_ctx.a*M_PI*xi) * cos(NS_ctx.a*M_PI*yi) * std::exp(-2.0*pow(NS_ctx.a,2.0)*pow(M_PI, 2.0)* NS_ctx.kinvis*t) ;
}

real_t p3(const Vector &x, real_t t)
{
   real_t xi = x(0);
   real_t yi = x(1);

   return -1.0/4.0 * ( cos(2.0*NS_ctx.a*M_PI*xi) + cos(2.0*NS_ctx.a*M_PI*yi) ) * std::exp(-4.0*pow(NS_ctx.a,2.0)*pow(M_PI, 2.0)* NS_ctx.kinvis*t);
}

void accel3(const Vector &x, real_t t, Vector &u)
{
   u = 0.0;
}

/*void accel3(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);

   u(0) =   2.0*std::exp(-2.0*t)*cos(xi)*sin(yi)                    // dudt
          - 2.0 * NS_ctx.kinvis * std::exp(-2.0*t)*cos(xi)*sin(yi)  // - nu lap u
          + (sin(2.0*xi) * std::exp(-2.0*t))/2.0                    // grad p 
          - (sin(2.0*xi) * std::exp(-4.0*t))/2.0;                   // u grad u

   u(1) =  -2.0*std::exp(-2.0*t)*cos(yi)*sin(xi)                    // dudt
          + 2.0 * NS_ctx.kinvis * std::exp(-2.0*t)*cos(yi)*sin(xi)  // - nu lap u
          + (sin(2.0*yi) * std::exp(-2.0*t))/2.0                    // grad p 
          - (sin(2.0*yi) * std::exp(-4.0*t))/2.0;                   // u grad u
} */



int main(int argc, char *argv[])
{

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int total_refinements = 0;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options.
   ///////////////////////////////////////////////////////////////////////////////////////////////

   SolverParams sParams(1e-8, 1e-10, 1000, 0); // rtol, atol, maxiter, print-level


   OptionsParser args(argc, argv);
   args.AddOption(&NS_ctx.uorder,
                  "-ou",
                  "--order-velocity",
                  "Order (degree) of the finite elements for velocity.");
   args.AddOption(&NS_ctx.porder,
                  "-op",
                  "--order-pressure",
                  "Order (degree) of the finite elements for pressure.");
   args.AddOption(&NS_ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&NS_ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&NS_ctx.kinvis, "-kv", "--kinematic-viscosity", "Kinematic Viscosity.");
    args.AddOption(&NS_ctx.fun, "-f", "--test-function",
                   "Analytic function to test");
    args.AddOption(&NS_ctx.bcs, "-bcs", "--boundary-conditions",
                  "ODE solver: 0 - Fully dirichlet,\n\t"
                  "            1 - Fully Neumann,\n\t"
                  "            2 - Mixed (top/bottom Dirichlet, left/right Neumann)");
    args.AddOption(&NS_ctx.bdf,
                   "-bdf",
                   "--bdf-order",
                   "Maximum bdf order (1<=bdf<=3)");
   args.AddOption((int *)&NS_ctx.pc_type,
                   "-pc",
                   "--preconditioner",
                   "Preconditioner type (0: Block Diagonal, 1: BlowLowerTri, 2: BlockUpperTri, 3: Chorin-Temam, 4: Yosida, 5: Chorin-Temam Pressure Corrected, 6: Yosida Pressure Corrected, 7: Yosida High Order Pressure Correction)");                
   args.AddOption((int *)&NS_ctx.schur_pc_type,
                   "-schur-pc",
                   "--schur-preconditioner",
                   "Preconditioner type (0: Pressure Mass, 1: Pressure Laplacian, 2: PCD, 3: Cahouet-Chabard, 4: LSC, 5: Approximate Discrete Laplacian)");
   args.AddOption((int *)&NS_ctx.time_adaptivity_type,
                   "-ta",
                   "--time-adaptivity",
                   "Time adaptivity type (0: None, 1: CFL, 2: HOPC)");
   args.AddOption(&NS_ctx.pressure_correction_order,
                   "-pco",
                   "--pressure-correction-order",
                   "Order of the pressure correction >= 1");
   args.AddOption(&NS_ctx.mass_lumping,
                     "-ml",
                     "--mass-lumping",
                     "-no-ml",
                     "--no-mass-lumping",
                     "Enable or disable mass lumping. (default = false)");
   args.AddOption(&NS_ctx.stiff_strain,
                     "-ss",
                     "--stiff-strain",
                     "-no-ss",
                     "--no-stiff-strain",
                     "Enable or disable stiff strain. (default = false)");
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
    args.AddOption(&total_refinements, "-r", "--refinements",
                     "Number of total uniform refinements");
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


   switch (NS_ctx.bcs)
   {
   case 0: // Fully dirichlet
   {
      if (myid == 0) {mfem::out << "Fully Dirichlet problem."<<std::endl;};
      break;
   }
   case 1: // Fully neumann 
   {
      if (myid == 0) {mfem::out << "Fully Neumann problem."<<std::endl;};
      break;
   }
   case 2: // Mixed
   {  
      if (myid == 0) {mfem::out << "Mixed Neumann/Dirichlet problem."<<std::endl;};
      break;
   }
   default:
      break;
   }

   if (Mpi::Root())
   {
      cout << "------------------------------------------------------------------------------------------------------------"
           << endl;
      cout << left << setw(16) << "DOFs u " << setw(16) << "DOFs p " << setw(16) << "h " << setw(16) << "L^2 error (u) " << setw(16) << "L^2 error (p) " << setw(16);
      cout << "L^2 rate (u)" << setw(16) << "L^2 rate (p)" << setw(16) << endl;
      cout << "------------------------------------------------------------------------------------------------------------"
           << endl;
   }

   real_t l2_err_u_prev = 0.0;
   real_t l2_err_p_prev = 0.0;
   real_t h1_err_u_prev = 0.0;
   real_t h1_err_p_prev = 0.0;
   real_t h_prev = 0.0;

   // Refinement loop
   for (int serial_ref_levels = 0; serial_ref_levels < total_refinements; serial_ref_levels++)
   {

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Read Mesh and create parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

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

   if(NS_ctx.fun==1)
   {
      GridFunction *nodes = mesh.GetNodes();
      *nodes *= 2.0;
      *nodes -= 1.0;
   }

   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   auto pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Create the NS Solver and BCHandler
   ///////////////////////////////////////////////////////////////////////////////////////////////
   
   // Create the BC handler (bcs need to be setup before calling Solver::Setup() )
   bool verbose = false;
   navier::BCHandler *bcs = new navier::BCHandler(pmesh, verbose); // Boundary conditions handler
   navier::MonolithicNavierSolver *naviersolver = new navier::MonolithicNavierSolver(pmesh, bcs, NS_ctx.kinvis, NS_ctx.uorder, NS_ctx.porder, NS_ctx.verbose);

   naviersolver->SetSolver(sParams);
   naviersolver->SetMaxBDFOrder(NS_ctx.bdf);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Set up boundary conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Set the initial condition.
   VectorFunctionCoefficient *u_excoeff = nullptr;
   VectorFunctionCoefficient *accel_excoeff = nullptr;
   FunctionCoefficient *p_excoeff = nullptr;
   
   switch (NS_ctx.fun)
   {
   case 1:
   {
      u_excoeff = new VectorFunctionCoefficient(pmesh->Dimension(), vel1);
      p_excoeff = new FunctionCoefficient(p1);
      accel_excoeff = new VectorFunctionCoefficient(pmesh->Dimension(), accel1);
      break;
   }
   case 2:
   {
      u_excoeff = new VectorFunctionCoefficient(pmesh->Dimension(), vel2);
      p_excoeff = new FunctionCoefficient(p2);
      accel_excoeff = new VectorFunctionCoefficient(pmesh->Dimension(), accel2);
      break;
   }
   case 3:
   {
      u_excoeff = new VectorFunctionCoefficient(pmesh->Dimension(), vel3);
      p_excoeff = new FunctionCoefficient(p3);
      accel_excoeff = new VectorFunctionCoefficient(pmesh->Dimension(), accel3);
      break;
   }
   default:
      break;
   }

   u_excoeff->SetTime( 0.0 );
   p_excoeff->SetTime( 0.0 );

   naviersolver->SetInitialConditionVel( *u_excoeff );
   naviersolver->SetInitialConditionPrevVel( *u_excoeff );
   naviersolver->SetInitialConditionPres( *p_excoeff );

   ParGridFunction* u_gf = naviersolver->GetVelocity();
   ParGridFunction* p_gf = naviersolver->GetPressure();

   ParGridFunction u_ex_gf(naviersolver->GetFESpaceVelocity());
   ParGridFunction p_ex_gf(naviersolver->GetFESpacePressure());
   ParGridFunction rhs_gf(naviersolver->GetFESpaceVelocity());

   ParGridFunction err_u_gf(naviersolver->GetFESpaceVelocity());
   ParGridFunction err_p_gf(naviersolver->GetFESpacePressure());

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   // Bottom=1, Right=2, Top=3, Left=4
   int bottom_attr = 1;
   int right_attr  = 2;
   int top_attr    = 3;
   int left_attr   = 4;

   u_ex_gf.ProjectCoefficient(*u_excoeff);
   p_ex_gf.ProjectCoefficient(*p_excoeff);

   ConstantCoefficient *alpha = nullptr;
   ConstantCoefficient *beta = nullptr;

   switch (NS_ctx.bcs)
   {
   case 0: // Fully dirichlet
   {
      Array<int> ess_attr(pmesh->bdr_attributes.Max());
      ess_attr = 1;
      bcs->AddVelDirichletBC(u_excoeff, ess_attr);
      break;
   }
   case 1: // Fully neumann 
   {
      Array<int> traction_attr(pmesh->bdr_attributes.Max());
      traction_attr = 1;
      alpha = new ConstantCoefficient(NS_ctx.kinvis);
      beta = new ConstantCoefficient(-1.0);
      bcs->AddCustomTractionBC(alpha, &u_ex_gf, beta, &p_ex_gf, traction_attr, false);   // (psi,v) = (kinvis * n.grad(u_ex) - p_ex.n,v) 
      break;
   }
   case 2: // Mixed
   {  
      Array<int> ess_attr(pmesh->bdr_attributes.Max());
      ess_attr = 0;
      ess_attr[bottom_attr - 1]  = 1;
      ess_attr[top_attr - 1] = 1;
      bcs->AddVelDirichletBC(u_excoeff, ess_attr);

      Array<int> traction_attr(pmesh->bdr_attributes.Max());
      traction_attr = 0;
      traction_attr[right_attr - 1]  = 1;
      traction_attr[left_attr - 1] = 1;
      alpha = new ConstantCoefficient(NS_ctx.kinvis);
      beta = new ConstantCoefficient(-1.0);
      bcs->AddCustomTractionBC(alpha, &u_ex_gf, beta, &p_ex_gf, traction_attr, false);   // (psi,v) = (kinvis * n.grad(u_ex) - p_ex.n,v) 
      break;
   }
   default:
      break;
   }

   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1;
   naviersolver->AddAccelTerm(accel_excoeff, domain_attr);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   naviersolver->Setup(NS_ctx.dt, NS_ctx.pc_type, NS_ctx.schur_pc_type, NS_ctx.time_adaptivity_type, NS_ctx.mass_lumping, NS_ctx.stiff_strain);

   if (NS_ctx.pc_type == navier::BlockPreconditionerType::YOSIDA_HIGH_ORDER_PRESSURE_CORRECTED)
   {
      naviersolver->SetPressureCorrectionOrder(NS_ctx.pressure_correction_order);
   }
   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Solve unsteady problem
   ///////////////////////////////////////////////////////////////////////////////////////////////
   
   real_t err_u = 0.0;
   real_t err_p = 0.0;
   real_t t = 0.0;
   real_t dt = NS_ctx.dt;
   bool last_step = false;
   bool accept_step = false;
   for (int step = 1; !last_step; ++step)
   {
      if (t + dt >= NS_ctx.t_final - dt / 2)
      {
         last_step = true;
      }

      // Update fields u/p for CustomNeumannBC
      u_excoeff->SetTime(t + dt);
      p_excoeff->SetTime(t + dt);

      u_ex_gf.ProjectCoefficient(*u_excoeff);
      p_ex_gf.ProjectCoefficient(*p_excoeff);

      // solve current step
      accept_step = naviersolver->Step(t, dt, step);
   }
   
      ////////////////////////////////////////////////////////////////////////////////////
      /// 9. Compute and print the L^2 and H^1 norms of the error.                     ///
      ////////////////////////////////////////////////////////////////////////////////////

      real_t l2_err_u = 0.0;
      real_t l2_err_p = 0.0;
      real_t h1_err_u = 0.0; // Not used for now
      real_t h1_err_p = 0.0; // Not used for now
      real_t l2_rate_u = 0.0;
      real_t l2_rate_p = 0.0;
      real_t h1_rate_u = 0.0; // Not used for now
      real_t h1_rate_p = 0.0; // Not used for now
      l2_err_u = u_gf->ComputeL2Error(*u_excoeff);
      l2_err_p = p_gf->ComputeL2Error(*p_excoeff);
      //h1_err = T_gf.ComputeH1Error(Tex_coeff, gradTex_coeff);
      h1_err_u = 0.0;
      h1_err_p = 0.0;

      real_t h_min = 0.0;
      real_t h_max = 0.0;
      real_t kappa_min = 0.0;
      real_t kappa_max = 0.0;
      pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

      if (serial_ref_levels != 0)
      {
         l2_rate_u = log(l2_err_u / l2_err_u_prev) / log(h_min / h_prev);
         l2_rate_p = log(l2_err_p / l2_err_p_prev) / log(h_min / h_prev);
         //h1_rate = log(h1_err / h1_err_prev) / log(h_min / h_prev);
         h1_rate_u = 0.0;
         h1_rate_p = 0.0;
      }
      else
      {
         l2_rate_u = 0.0;
         l2_rate_p = 0.0;
         h1_rate_u = 0.0;
         h1_rate_p = 0.0;
      }

      l2_err_u_prev = l2_err_u;
      l2_err_p_prev = l2_err_p;
      h1_err_u_prev = h1_err_u;
      h1_err_p_prev = h1_err_p;
      h_prev = h_min;

      if (Mpi::Root())
      {
         cout << setw(16) << naviersolver->GetVelocitySize() << setw(16) << naviersolver->GetPressureSize()  << setw(16) << h_min << setw(16) << l2_err_u << setw(16) << l2_err_p;
         cout << setw(16) << l2_rate_u << setw(16) << l2_rate_p << endl;
      }


   //////////////////////////////////////////////////////////////
   ///                           Cleanup                       //
   //////////////////////////////////////////////////////////////

   delete naviersolver; 

   if (alpha) { delete alpha; }
   if (beta) { delete beta; }

   if (NS_ctx.bcs == 0 ) 
      delete p_excoeff;
   else if (NS_ctx.bcs == 1)
      delete u_excoeff;

   }   
   
   return 0; 

}
