// This miniapp solves a quasistatic solid mechanics problem assuming an hyperelastic
// material and no body forces.
//
// The equation
//                   ∇⋅σ(∇u) = 0
//
// with stress σ is solved for displacement u.
//
//             +----------+----------+
//   fixed --->|                     |<--- constant displacement
//             |                     |
//             +----------+----------+
//
// This miniapp uses an elasticity operator that allows for a custom material,
// defined through the DifferentiableOperator interface.
// Available materials include:
//   - Linear elastic
//   - Saint-Venant-Kirchoff
//   - Neo-Hookean
//   - Mooney-Rivlin
// default: linear elastic
//
// Sample runs:
// 1. Linear elastic material, prescribed displacement, full Jacobian update
//    mpirun -np 4 ./test_beam -o 2 -rs 1 -mt 0 -pt 0 -of ./Output/TestBeam
// 2. Neo-Hookean material, prescribed displacement, full Jacobian update
//    mpirun -np 4 ./test_beam -o 2 -rs 1 -mt 2 -pt 0 -of ./Output/TestBeam
// 3. Linear elastic material, prescribed displacement, reduced Jacobian update (every 10 iters)
//    mpirun -np 4 ./test_beam -o 2 -rs 1 -mt 0 -pt 0 -kgu 10 -of ./Output/TestBeam

#include "../materials.hpp"
#include "../solvers/quasi_static_elasticity_solver.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::elasticity_ecm2;


enum class ProblemType : int
{
  PrescribedDisplacement = 0,
  PrescribedTraction = 1,
  BodyForce = 2
};

void prescribed_displacement_fun(const Vector &x, real_t time, Vector &v);
void prescribed_load_fun(const Vector &x, real_t time, Vector &v);
void body_force_fun(const Vector &x, real_t time, Vector &v);

int main(int argc, char *argv[])
{

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mpi::Init(argc, argv);
   Hypre::Init();

   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   const char *mesh_file = "../../data/beam-hex.mesh";

   int order = 1;
   int material_type = 0; // 0: linear elastic, 1: Saint-Venant-Kirchoff, 2: Neo-Hookean, 3: Mooney-Rivlin
   ProblemType problem_type = ProblemType::PrescribedDisplacement;
   int k_grad_update = 1; // Gradient update frequency for FrozenNewtonSolver, k=1 corresponds to standard NewtonSolver

   //const char *device_config = "cpu";
   const char *outfolder = "./Output/TestBeam";
   int serial_refinement_levels = 0;
   bool paraview = false;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options.
   ///////////////////////////////////////////////////////////////////////////////////////////////

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&material_type, "-mt", "--material-type",
                  "Material type: 0 - linear elastic, 1 - Saint-Venant-Kirchoff, "
                  "2 - Neo-Hookean, 3 - Mooney-Rivlin.");
   args.AddOption((int *)&problem_type, "-pt", "--problem-type",
                  "Problem type: 0 - prescribed displacement, "
                  "1 - prescribed traction, 2 - body force.");
   args.AddOption(&k_grad_update, "-kgu", "--k-grad-update",
                  "Gradient update frequency for the FrozenNewtonSolver. "
                  "k=1 corresponds to the standard NewtonSolver.");
   //args.AddOption(&device_config, "-d", "--device",
   //               "Device configuration string, see Device::Configure().");
   args.AddOption(&serial_refinement_levels, "-rs", "--ref-serial",
                  "Number of uniform refinements on the serial mesh.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable ParaView DataCollection output.");
   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder.");
   args.ParseCheck();

   //Device device(device_config);
   //if (Mpi::Root())
   //{
   //   device.Print();
   //}

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create Mesh
   ///////////////////////////////////////////////////////////////////////////////////////////////


   auto mesh = Mesh(mesh_file, 1, 1);
   mesh.EnsureNodes();

   for (int l = 0; l < serial_refinement_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();


   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Create Solid Solver
   ///////////////////////////////////////////////////////////////////////////////////////////////

   bool verbose = true;

   ElasticitySolver3D solver(&pmesh, order, verbose);

   ParGridFunction &u_gf = *(solver.GetDisplacementGridFunction());

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Set BCs  and  setup problem
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Define all essential boundaries. In this specific example, this includes
   // all fixed and statically displaced degrees of freedom on mesh entities in
   // the defined attributes.
   // Boundaries: 0 = beam left side, 1 = beam right side, 2 = beam lateral
   Array<int> fixed_attr(pmesh.bdr_attributes.Max());
   fixed_attr = 0;
   fixed_attr[1] = 1;

   Array<int> displaced_attr(pmesh.bdr_attributes.Max());
   displaced_attr = 0;
   displaced_attr[0] = 1;

   Array<int> domain_attr(pmesh.attributes.Max());
   domain_attr = 1;

   switch (problem_type)
   {
      case ProblemType::PrescribedDisplacement:
         solver.AddPrescribedDisplacement(prescribed_displacement_fun, displaced_attr);
         break;
      case ProblemType::PrescribedTraction:
         solver.AddBoundaryLoad(prescribed_load_fun, displaced_attr);
         break;
      case ProblemType::BodyForce:
         solver.AddBodyForce(body_force_fun, domain_attr);
         break;
      default:
         mfem_error("Unknown problem type");
   }
      
   solver.AddFixedConstraint(fixed_attr);

   // Material selection
   // Parameters trying to match the hooke miniapp
   switch (material_type)
   {
   case 0: // Linear elastic
   {
      real_t E = 2e4/150;  // Young's modulus  ~ 133.33
      real_t nu = 1.0/3.0; // Poisson's ratio  ~ 0.333 
      auto material = make_linear_elastic<3>(E, nu);
      solver.SetMaterial(material);
      break;
   }
   case 1: // Saint-Venant-Kirchoff
   {
      real_t E = 2e4/150;  // Young's modulus  ~ 133.33
      real_t nu = 1.0/3.0; // Poisson's ratio  ~ 0.333
      auto material = make_saint_venant_kirchoff<3>(E, nu);
      solver.SetMaterial(material);
      break;
   }
   case 2: // Neo-Hookean
   {
      real_t kappa = 13.07; // Bulk modulus (matches linear elastic K)
      real_t c = 5.0;       // Shear parameter (matches linear elastic G)
      auto material = make_neo_hookean<3>(kappa, c);
      solver.SetMaterial(material);
      break;
   }
   case 3: // Mooney-Rivlin
   { // NOTE: there's something wrong with this material either parameter or implementation
      real_t kappa = 13.07; // Bulk modulus (matches Neo-Hookean)
      real_t c1 = 5.0;      // First Mooney-Rivlin parameter
      real_t c2 = 5.0;      // Second Mooney-Rivlin parameter (c1 + c2 = 10.0 = mu)
      auto material = make_mooney_rivlin<3>(kappa, c1, c2);
      solver.SetMaterial(material);
      break;
   }
   default:
      mfem_error("Unknown material type");
   }

   solver.Setup(k_grad_update);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Set up output
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ParaViewDataCollection pvdc_solid("elasticity_output", &pmesh);
   pvdc_solid.SetPrefixPath(outfolder);
   pvdc_solid.SetDataFormat(VTKFormat::BINARY32);
   pvdc_solid.SetHighOrderOutput(true);
   pvdc_solid.SetLevelsOfDetail(order);
   pvdc_solid.SetCycle(0);
   pvdc_solid.SetTime(0.0);
   pvdc_solid.RegisterField("displacement", &u_gf);
   pvdc_solid.Save();


   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7.  Solve problem
   ///////////////////////////////////////////////////////////////////////////////////////////////

   solver.Solve();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Cleanup and finalize
   ///////////////////////////////////////////////////////////////////////////////////////////////

   pvdc_solid.SetCycle(1);
   pvdc_solid.SetTime(0.1);
   pvdc_solid.RegisterField("displacement", &u_gf);
   pvdc_solid.Save();

   return 0;
}


void prescribed_displacement_fun(const Vector &x, real_t time, Vector &v)
{
   v = 0.0;
   v[1] = -0.8; // constant displacement in y-direction
}


void prescribed_load_fun(const Vector &x, real_t time, Vector &v)
{
   v = 0.0;
   v[1] = -1e-2; // constant load in y-direction
}

void body_force_fun(const Vector &x, real_t time, Vector &v)
{
   v = 0.0;
   v[1] = -3e-3; // constant body force in y-direction
}