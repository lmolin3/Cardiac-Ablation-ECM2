// Navier Poiseulle example
//
// The problem domain is set up like this
//
//                            no slip
//             |\     + ----------------------------------+
//             | \    |                                   |
// Parabolic -->  \   |                                   |   Traction free (outflow)
//  inflow   -->  /   |                                   |
//             | /    |                                   |
//             |/     + ----------------------------------+
//                            no slip
//
// Inflow profile: u_in = 1/4kinvis dpdx (R^2 - r^2)    (Poiseulle flow)
// 
// Mesh attributes for 2D are:
// inflow = 1, outflow = 2, cylinder = 3, wall = 4
//
// Mesh attributes for 3D are:
// inflow = 1, outflow = 2, sphere = 3, wall = 4
//
//
// Sample run:
//
// 1. Yosida block preconditioner + ApproximateDiscreteLaplacian Schur complement preconditioner
// mpirun -np 4 ./navier-stented-channel-monolithic -rs 0 -ou 2 -op 1 -dt 1e-3 -tf 1e-1 -kv 1.0 -dp 1.0 --preconditioner 4 --schur-preconditioner 5
//
// 2. Mass lumping
// mpirun -np 4 ./navier-stented-channel-monolithic -rs 0 -ou 2 -op 1 -dt 1e-3 -tf 1e-1 -kv 1.0 -dp 1.0 --preconditioner 4 --schur-preconditioner 5 --mass-lumping
//
// 3. Stiff strain
// mpirun -np 4 ./navier-stented-channel-monolithic -rs 0 -ou 2 -op 1 -dt 1e-3 -tf 1e-1 -kv 1.0 -dp 1.0 --preconditioner 4 --schur-preconditioner 5 --stiff-strain
//
   
#include "lib/navier_solver.hpp"
#include <fstream>
#include <sys/stat.h>  // Include for mkdir

#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979
#endif

using namespace mfem;
using namespace navier;


struct s_NavierContext // Navier Stokes params
{
   int uorder = 2;
   int porder = 1;
   real_t R = 0.5;           
   real_t kinvis = 1.0;
   real_t dpdx = 4.0;
   real_t dt = 1e-3;
   real_t t_final = 10 * dt;
   real_t preloadT = 0.0;
   real_t gamma = 1.0;
   bool verbose = true;
   bool paraview = false;
   int save_freq = 1;
   const char *outfolder = "./Output/Poiseulle/Test/";
   int bdf = 3;
   BlockPreconditionerType pc_type = BlockPreconditionerType::BLOCK_DIAGONAL;       // 0: Block Diagonal, 1: BlowLowerTri, 2: BlockUpperTri, 3: Chorin-Temam, 4: Yosida, 5: Chorin-Temam Pressure Corrected, 6: Yosida Pressure Corrected
   SchurPreconditionerType schur_pc_type = SchurPreconditionerType::APPROXIMATE_DISCRETE_LAPLACIAN; // 0: Pressure Mass, 1: Pressure Laplacian, 2: PCD, 3: Cahouet-Chabard, 4: LSC, 5: Approximate Inverse
   TimeAdaptivityType time_adaptivity_type = TimeAdaptivityType::NONE; // Time adaptivity type (NONE, CFL, HOPC)
   int pressure_correction_order = 2; // Order of the pressure correction
   bool mass_lumping = false; // Enable mass lumping
   bool stiff_strain = false; // false: viscous stress (Δu), true: stiff strain ( ∇u + ∇u^T )
} NS_ctx;

struct s_MeshContext // mesh
{
   int dim = 2;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
} Mesh_ctx;


// Forward declarations of functions
void inflow(const Vector &x, real_t t, Vector &u);
void noSlip(const Vector &x, real_t t, Vector &u);
real_t pZero(const Vector &x, real_t t);

int main(int argc, char *argv[])
{

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
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
   args.AddOption(&NS_ctx.dpdx, "-dp", "--pressure-drop", "Pressure difference driving flow.");
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
    args.AddOption(&NS_ctx.save_freq,
                   "-sf",
                   "--save-freq",
                   "Frequency to save output.");
    args.AddOption(&NS_ctx.bdf,
                   "-bdf",
                   "--bdf-order",
                   "Maximum bdf order (1<=bdf<=3)");
   args.AddOption((int *)&NS_ctx.pc_type,
                   "-pc",
                   "--preconditioner",
                   "Preconditioner type (0: Block Diagonal, 1: BlowLowerTri, 2: BlockUpperTri, 3: Chorin-Temam, 4: Yosida, 5: Chorin-Temam Pressure Corrected, 6: Yosida Pressure Corrected)");                 
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
   mesh = Mesh::LoadFromFile("./Mesh/stented-channel.msh");


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
   int inflow_attr = 1;
   int outflow_attr = 2;
   int wall_attr = 3;

   // Inflow
   VectorFunctionCoefficient *u_in = new VectorFunctionCoefficient(pmesh->Dimension(), inflow);
   bcs->AddVelDirichletBC(u_in,inflow_attr);

   // Outflow
   //FunctionCoefficient *p_out = new FunctionCoefficient(pZero);
   //bcs->AddPresDirichletBC(p_out,outflow_attr);

   // No Slip
   bcs->AddVelDirichletBC(noSlip,wall_attr);

   // Initial condition
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

   naviersolver->Setup(NS_ctx.dt, NS_ctx.pc_type, NS_ctx.schur_pc_type, NS_ctx.time_adaptivity_type, NS_ctx.mass_lumping, NS_ctx.stiff_strain);

   if (NS_ctx.pc_type == navier::BlockPreconditionerType::YOSIDA_HIGH_ORDER_PRESSURE_CORRECTED)
   {
      naviersolver->SetPressureCorrectionOrder(NS_ctx.pressure_correction_order);
   }

   // Get problem size
   int vel_size = naviersolver->GetVelocitySize();
   int pres_size = naviersolver->GetPressureSize();

   if (Mpi::Root())
   {
      mfem::out << "Velocity dofs: " << vel_size << std::endl;
      mfem::out << "Pressure dofs: " << pres_size << std::endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Solve unsteady problem
   ///////////////////////////////////////////////////////////////////////////////////////////////
 
   real_t t = 0.0;
   real_t dt = NS_ctx.dt;
   bool last_step = false;
   bool accept_step = false;

   real_t last_save_time = 0.0;

   for (int step = 1; !last_step; ++step)
   {
      if (t + dt >= NS_ctx.t_final - dt / 2)
      {
         last_step = true;
      }

      // Solve
      accept_step = naviersolver->Step(t, dt, step);
      if (NS_ctx.paraview && accept_step && 
         (step % NS_ctx.save_freq == 0 || (t - last_save_time) >= 1e-1))
     {
        naviersolver->WriteFields(step, t);
        last_save_time = t;
     }
   }

   

   delete paraview_dc;
   delete naviersolver;

   return 0;
}



void inflow(const Vector &x, real_t t, Vector &u)
{

   const int dim = x.Size();

   real_t r = x[1]*x[1] + x[2]*x[2];

      // Preload
      real_t preload = 0.0;
      if( t < NS_ctx.preloadT )
      {
         preload = 0.5* (1.0 - cos(M_PI*t/NS_ctx.preloadT));   
      }
      else
      {
         preload = 1.0;
      }

   u = 0.0;
   u(0) = preload * ( 1.0 / (4.0 * NS_ctx.kinvis ) * NS_ctx.dpdx * ( NS_ctx.R*NS_ctx.R - r*r) );   // 1/ 4kinvis dpdx (R^2 - r^2)
}

void noSlip(const Vector &x, real_t t, Vector &u)
{
   u = 0.0;
}

real_t pZero(const Vector &x, real_t t)
{
   return 0.0;
}