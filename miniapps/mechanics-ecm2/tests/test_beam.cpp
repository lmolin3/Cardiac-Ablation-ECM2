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
// By default the NeoHookeanMaterial is used. 
//

#include "../materials.hpp"
#include "../solvers/quasi_static_elasticity_solver.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::elasticity_ecm2;

void prescribed_displacement_fun(const Vector &x, real_t time, Vector &v);

int main(int argc, char *argv[])
{

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mpi::Init(argc, argv);
   Hypre::Init();

   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   int order = 1;
   int material_type = 0; // 0: linear elastic, 1: Saint-Venant-Kirchoff, 2: Neo-Hookean, 3: Mooney-Rivlin
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
   args.AddOption(&material_type, "-mt", "--material-type",
                  "Material type: 0 - linear elastic, 1 - Saint-Venant-Kirchoff, "
                  "2 - Neo-Hookean, 3 - Mooney-Rivlin.");
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


   auto mesh =
       Mesh::MakeCartesian3D(8, 2, 2, Element::HEXAHEDRON, 8.0, 1.0, 1.0); 
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
   Array<int> fixed_attr(pmesh.bdr_attributes.Max());
   fixed_attr = 0;
   fixed_attr[4] = 1;

   Array<int> displaced_attr(pmesh.bdr_attributes.Max());
   displaced_attr = 0;
   displaced_attr[2] = 1;

   solver.AddFixedConstraint(fixed_attr);
   solver.AddPrescribedDisplacement(prescribed_displacement_fun, displaced_attr);

// Material selection
   switch (material_type)
   {
   case 0: // Linear elastic
   {
      real_t E = 1e6;  // Young's modulus
      real_t nu = 0.3; // Poisson's ratio
      auto material = make_linear_elastic<3>(E, nu);
      solver.SetMaterial(material);
      break;
   }
   case 1: // Saint-Venant-Kirchoff
   {
      real_t E = 1e6;  // Young's modulus
      real_t nu = 0.3; // Poisson's ratio
      auto material = make_saint_venant_kirchoff<3>(E, nu);
      solver.SetMaterial(material);
      break;
   }
   case 2: // Neo-Hookean
   {
      real_t kappa = 5e5; // Bulk modulus
      real_t c = 2e5;     // Shear parameter
      auto material = make_neo_hookean<3>(kappa, c);
      solver.SetMaterial(material);
      break;
   }
   case 3: // Mooney-Rivlin
   {
      real_t kappa = 5e5; // Bulk modulus
      real_t c1 = 1.6e5;  // First Mooney-Rivlin parameter
      real_t c2 = 4e4;    // Second Mooney-Rivlin parameter
      auto material = make_mooney_rivlin<3>(kappa, c1, c2);
      solver.SetMaterial(material);
      break;
   }
   default:
      mfem_error("Unknown material type");
   }

   solver.Setup();

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
   v[0] = 1e-1; // constant displacement in x-direction
}