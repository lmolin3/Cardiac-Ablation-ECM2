// Flow around cylinder in 2D/3D
//
// The problem domain is set up like this
//
//                                  no slip
//              |\    + ---------------------------------- +
// Parabolic -->  |   |                                    |
//  inflow      |/    + ------ +                           |  Traction free (outflow)
//                             |                           |
//                             + ------------------------- +
//                                      no slip
//
// Mesh attributes for 2D/3D are:
// inflow = 1, outflow = 2, wall = 3
//
// Sample run:
//
// 1. Yosida block preconditioner + ApproximateDiscreteLaplacian Schur complement preconditioner 
// mpirun -np 4 ./navier-backstep-monolithic -d 2 -rs 0 -rp 0 -ou 2 -op 1 -dt 1e-3 -tf 1e-1 -tp 1e-2 -kv 0.1 -u 1.0 --preconditioner 4 --schur-preconditioner 5
//
// 2. Mass lumping
// mpirun -np 4 ./navier-backstep-monolithic -d 2 -rs 0 -rp 0 -ou 2 -op 1 -dt 1e-3 -tf 1e-1 -tp 1e-2 -kv 0.1 -u 1.0 --preconditioner 4 --schur-preconditioner 5 --mass-lumping
// 
// 3. Stiff strain
// mpirun -np 4 ./navier-backstep-monolithic -d 2 -rs 0 -rp 0 -ou 2 -op 1 -dt 1e-3 -tf 1e-1 -tp 1e-2 -kv 0.1 -u 1.0 --preconditioner 4 --schur-preconditioner 5 --stiff-strain
//

#include "../lib/navier_solver.hpp"
#include <fstream>
#include <sys/stat.h>  // Include for mkdir

#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979
#endif

using namespace mfem;
using namespace navier;

struct s_MeshContext // mesh
{
   int dim = 2;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int D = 1.0;  // Channel diameter
} Mesh_ctx;


struct s_NavierContext // Navier Stokes params
{
   int uorder = 2;
   int porder = 1;
   double kinvis = 0.01;
   double Umax =  1.0;
   double dt = 1e-3;
   double t_final = 10 * dt;
   double preloadT = 0.1 * t_final;
   bool verbose = true;
   bool paraview = false;
   const char *outfolder = "./Output/BackFacingStep/2D/Test/";
   bool ExportData = false;
   int bdf = 3;
   // int splitting_type = 0;  // 0 = Chorin-Temam, 1 = Yosida, 2 = High-Order Yosida 
   // int correction_order = 1; // Correction order for High-Order Yosida   
   BlockPreconditionerType pc_type = BlockPreconditionerType::BLOCK_DIAGONAL;       // 0: Block Diagonal, 1: BlowLowerTri, 2: BlockUpperTri, 3: Chorin-Temam, 4: Yosida, 5: Chorin-Temam Pressure Corrected, 6: Yosida Pressure Corrected
   SchurPreconditionerType schur_pc_type = SchurPreconditionerType::APPROXIMATE_DISCRETE_LAPLACIAN; // 0: Pressure Mass, 1: Pressure Laplacian, 2: PCD, 3: Cahouet-Chabard, 4: LSC, 5: Approximate Inverse   int correction_order = 1; // Correction order for High-Order Yosida
   TimeAdaptivityType time_adaptivity_type = TimeAdaptivityType::NONE; // Time adaptivity type (NONE, CFL, HOPC)
   int pressure_correction_order = 2; // Order of the pressure correction
   bool mass_lumping = false; // Enable mass lumping
   bool stiff_strain = false; // false: viscous stress (Δu), true: stiff strain ( ∇u + ∇u^T )
} NS_ctx;


// Forward declarations of functions
void inflow(const Vector &x, double t, Vector &u);
void noSlip(const Vector &x, double t, Vector &u);
double pZero(const Vector &x, double t);

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

   SolverParams sParams(1e-6, 1e-8, 1000, 0); // rtol, atol, maxiter, print-level


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
   args.AddOption(&NS_ctx.preloadT, "-tp", "--preload-time", "Preload time.");
   args.AddOption(&NS_ctx.kinvis, "-kv", "--kinematic-viscosity", "Kinematic Viscosity.");
   args.AddOption(&NS_ctx.Umax, "-u", "--inflow-velocity",
                   "Inflow velocity");
   args.AddOption(&NS_ctx.verbose,
                  "-v",
                  "--verbose",
                  "-no-v",
                  "--no-verbose",
                  "Enable verbosity.");
   args.AddOption(&NS_ctx.paraview,
                  "-pv",
                  "--paraview",
                  "-no-pv",
                  "--no-paraview",
                  "Enable or disable Paraview output.");
   args.AddOption(&NS_ctx.outfolder,
                   "-of",
                   "--output-folder",
                   "Output folder.");
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


   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Read Mesh and create parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mesh mesh;

   switch (Mesh_ctx.dim)
   {
   case 2:
   {
      mesh = Mesh::LoadFromFile("./Mesh/back_facing_step_2D.msh");
      break;
   }
   case 3:
   {
      mesh = Mesh::LoadFromFile("./Mesh/back_facing_step_3D.msh");
      break;
   }
   default:
      break;
   }

   for (int l = 0; l < Mesh_ctx.ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }


   auto pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < Mesh_ctx.par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }


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


   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.

   if (NS_ctx.verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Kinematic viscosity: " << NS_ctx.kinvis << std::endl;
      mfem::out << "Max velocity: " << NS_ctx.Umax << std::endl;
      mfem::out << std::endl;
   }

   int inflow_attr = 1;
   int outflow_attr = 2;
   Array<int> ess_attr(pmesh->bdr_attributes.Max());
   ess_attr = 1;                    // Mark walls/cylinder for no-slip condition
   ess_attr[inflow_attr - 1]  = 0;
   ess_attr[outflow_attr - 1] = 0;

   // Inflow
   bcs->AddVelDirichletBC(inflow,inflow_attr);

   // No Slip
   bcs->AddVelDirichletBC(noSlip,ess_attr);

   // Outflow
   //FunctionCoefficient *p_out = new FunctionCoefficient(pZero);
   //bcs->AddPresDirichletBC(p_out,outflow_attr);


   // initial condition
   //naviersolver->SetInitialConditionVel( *u_in );
   //naviersolver->SetInitialConditionPrevVel( *u_in );

   ParGridFunction *u_gf = naviersolver->GetVelocity();
   ParGridFunction *p_gf = naviersolver->GetPressure();

   // Creating output directory if not existent
   ParaViewDataCollection *paraview_dc = nullptr;
   
   if( NS_ctx.paraview )
   {
      if ( (mkdir(NS_ctx.outfolder, 0777) == -1) && (pmesh->GetMyRank() == 0) ) {mfem::err << "Error :  " << strerror(errno) << std::endl;}

      paraview_dc = new ParaViewDataCollection("Results-Paraview", pmesh.get());
      paraview_dc->SetPrefixPath(NS_ctx.outfolder);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetCompressionLevel(9);
      naviersolver->RegisterParaviewFields(*paraview_dc);

      naviersolver->WriteFields(0, 0.0);

   }   


   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Setup solver and Assemble forms
   ///////////////////////////////////////////////////////////////////////////////////////////////

   navier::QuantitiesOfInterest qoi(pmesh.get());
   naviersolver->Setup(NS_ctx.dt, NS_ctx.pc_type, NS_ctx.schur_pc_type, NS_ctx.time_adaptivity_type, NS_ctx.mass_lumping, NS_ctx.stiff_strain);

   if (NS_ctx.pc_type == navier::BlockPreconditionerType::YOSIDA_HIGH_ORDER_PRESSURE_CORRECTED)
   {
      naviersolver->SetPressureCorrectionOrder(NS_ctx.pressure_correction_order);
   }
   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Solve unsteady problem
   ///////////////////////////////////////////////////////////////////////////////////////////////


   double t = 0.0;
   double dt = NS_ctx.dt;
   bool last_step = false;
   bool accept_step = false;

   real_t CFL = 0.0;
   
   for (int step = 1; !last_step; ++step)
   {
      if (t + dt >= NS_ctx.t_final - dt / 2)
      {
         last_step = true;
      }

      accept_step = naviersolver->Step(t, dt, step);

      CFL = qoi.ComputeCFL(*u_gf, dt);

      if( NS_ctx.paraview && accept_step)
      {
         naviersolver->WriteFields(step, t);
      }

      if (pmesh->GetMyRank() == 0)
         mfem::out << "CFL: " << CFL << std::endl;

   }

   

   delete paraview_dc;
   delete naviersolver; 

   return 0;
}


void inflow(const Vector &x, double t, Vector &u)
{

   const int dim = x.Size();
   double xi = x[0];
   double yi = x[1];

   u = 0.0;

   u(1) = 0.0;

   // Preload
   double preload = 0.0;
   if( t < NS_ctx.preloadT )
   {
      preload = 0.5* (1.0 - cos(M_PI*t/NS_ctx.preloadT));   
   }
   else
   {
      preload = 1.0;
   }

   if( dim == 3)
   {
      double zi = x[2];
      u(0) = preload * NS_ctx.Umax * 32.0 / Mesh_ctx.D *( 1.0 - yi/Mesh_ctx.D) * ( 1.0 - 2.0 * yi / Mesh_ctx.D) * ( zi/Mesh_ctx.D - 1.0) * zi;
      u(2) = 0.0;
   }
   else
   {
      u(0) = preload * 8.0 * NS_ctx.Umax * ( 1.0 - yi/Mesh_ctx.D ) * ( 2.0 * yi/Mesh_ctx.D - 1.0 );
   }

}

void noSlip(const Vector &x, double t, Vector &u)
{
   u = 0.0;
}

double pZero(const Vector &x, double t)
{
   return 0.0;
}