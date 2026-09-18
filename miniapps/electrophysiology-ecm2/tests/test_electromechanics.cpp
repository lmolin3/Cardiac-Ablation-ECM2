//         One-way coupled cardiac electromechanics -- tissue level
//
// A propagating action potential on a ventricular tissue slab, driving active
// contraction through the Land 2017 cell model and a finite-strain Holzapfel
// material. This is the tissue-scale counterpart of bench_electromechanics,
// which verifies the same coupling on a single element.
//
// Physics, per time step (first order operator splitting, D -> R):
//
//   1. diffusion    monodomain PDE, chi Cm dV/dt = div(sigma grad V)
//   2. reaction     TP06 cell ODEs at every dof, caches dt
//   3. contraction  Land 2017 ODEs, on demand, driven by [Ca2+]i
//   4/5. mechanics  quasi-static equilibrium with Ta frozen
//
// Geometry follows the Niederer et al. benchmark slab: 20 x 7 x 3 mm, stimulated
// in a 1.5 mm corner cube, so the wave runs the long axis of the slab. Units are
// mm / ms / mV / kPa throughout, which is what makes the Land tension (kPa) and
// the Holzapfel parameters (kPa) directly comparable.
//
// Fibers rotate transmurally, as in real myocardium: the fiber angle sweeps
// linearly through the wall from -60 deg at z = 0 to +60 deg at z = sz. The same
// field is used twice -- it sets the conduction anisotropy (sigma is transversely
// isotropic about f0) AND the direction the active tension pulls along -- so the
// wave travels fastest along the local fiber and the slab contracts anisotropically.
//
// ---------------------------------------------------------------------------
// Sample runs
// ---------------------------------------------------------------------------
//
// Backend is -dev, and it is the only thing that changes between host and GPU.
//
//   Quick look, CPU:      mpirun -np 4 ./test_electromechanics -dev cpu -tf 100 -pv
//   GPU, active stress:   ./test_electromechanics -dev cuda -form stress -tf 400 -pv
//   GPU, active strain:   ./test_electromechanics -dev cuda -form strain -tf 400 -pv
//   Both, back to back:   ./test_electromechanics -dev cuda -form both -tf 400 -pv
//
//   Uniform fibers:       ./test_electromechanics -dev cuda -fa 0 -tf 400 -pv
//   Free-floating slab:   ./test_electromechanics -dev cuda -bc free -tf 400 -pv
//   Transient as CSV:     ./test_electromechanics -dev cuda -tf 400 -csv tissue.csv
//   Jacobi preconditioner:./test_electromechanics -dev cuda -pc jacobi -tf 400
//   Pure EP (no mechanics):./test_electromechanics -dev cuda -no-mech -o 2 \
//                            -nx 159 -ny 56 -nz 24 -tf 1 -dode 1
//   Large, EP o2 / mech o1:./test_electromechanics -dev cuda -o 2 -mo 1 \
//                            -nx 159 -ny 56 -nz 24 -tf 400 -ms 20 -pc jacobi -pv
//   Best cost/accuracy:   ./test_electromechanics -dev cuda -o 3 -mo 1 \
//                            -nx 67 -ny 23 -nz 10 -tf 400 -ms 20 -pc amg -pv
//
// ---------------------------------------------------------------------------
// Resolution and cost
// ---------------------------------------------------------------------------
//
// The default 0.5 mm mesh is the coarsest the Niederer benchmark accepts, and it
// is NOT a free choice: at 1 mm the transverse diffusivity (sigma_t/sigma_l ~ 1/8)
// is under-resolved and the wave never crosses the wall -- part of the slab stays
// at rest for the whole beat, and the mechanics then contract only the half that
// activated. 0.5 mm and 0.25 mm agree on Vm to 0.01 mV, so 0.5 mm is converged.
//
// Measured on one RTX-class GPU. "EP/step" is diffusion plus reaction; the
// mechanics column is one Newton solve with the default AMG preconditioner.
//
//   mesh          EP dofs   mech dofs   EP/step   CG iters   mechanics/solve
//   40x14x6         4,305      12,915    5.9 ms      12          0.40 s   (default)
//   80x28x12       30,537      91,611   17.5 ms      14          1.91 s
//   100x35x15      58,176     174,528   27.5 ms      14          3.74 s
//
// The EP side scales about linearly and is dominated by fixed per-step overhead
// below ~30k dofs. Because the two physics are staggered, it costs exactly what
// the pure-EP tests in this directory cost -- verified against BENCHMARK.md
// Case 2 (order 2, PA, CG rtol 1e-6, CUDA):
//
//   run                                   dofs      diffusion/step   per Mdof
//   test_monodomain -d 3 -rs 4 -o 2    1,684,865       258.5 ms      153.4 ms
//   this test, -no-mech                1,766,303       276.5 ms      156.6 ms
//
// 2% apart, and this test additionally carries a full anisotropic conductivity
// tensor with rotating fibers where the benchmark has an isotropic constant.
// Registering a contraction model adds ~0.4 ms/step.
//
// CAVEAT, and it is a sharp one on a small card. The mechanics operator allocates
// its dFEM quadrature caches at CONSTRUCTION, not at first solve. At 1.77M EP
// dofs (5.3M mechanics dofs) that is ~2.4 GB on top of the EP data, which takes
// an 8 GB device from 64% to 94% full -- and the diffusion step then degrades
// 5.9x, from 276 ms to 1619 ms, without the mechanics having solved even once:
//
//   -no-mech   diffusion 276 ms/step   peak 5268 MiB
//   -mech      diffusion 1619 ms/step  peak 7711 MiB
//
// So a slow EP step in a coupled run is a memory symptom, not an EP regression;
// use -no-mech to confirm. Note the caches are NOT quadrature driven -- cutting
// the rule from 27 to 8 points per element (-qo) changes peak memory by 0.2% --
// they are the per-field E-vector working set, which scales with dofs.
//
// Which is what -mo fixes. The mechanics solves for a DISPLACEMENT VECTOR, so at
// equal order and mesh it has exactly 3x the dofs of the scalar EP problem. It
// does not need equal order: the deformation field is far smoother than the
// activation front. Running EP at order 2 and mechanics at order 1 on the SAME
// mesh (no transfer operator -- dFEM interpolates each field to the shared
// quadrature rule independently) at 1.77M EP dofs:
//
//   -mo    mech dofs    peak mem    diffusion/step   mechanics/solve
//    2     5,298,909    7780 MiB      1722 ms           2.51 s
//    1       684,000    6616 MiB       250 ms           0.14 s
//
// 7.75x fewer mechanics dofs, and the diffusion step returns to the pure-EP
// baseline (BENCHMARK.md Case 2: 258.5 ms). The accuracy cost is under 1%:
// on a 50x18x8 order-2 mesh, peak |u| is 1.2020e-01 mm at -mo 2 against
// 1.1904e-01 mm at -mo 1, with peak Ta identical (it comes from the EP side).
//
// ---------------------------------------------------------------------------
// High order on a coarse mesh: the single biggest win
// ---------------------------------------------------------------------------
//
// What resolves the depolarisation wavefront is the DOF SPACING, not the element
// size. So at fixed dof count you can trade elements for polynomial order, and
// both physics get cheaper at once.
//
// PA rewards that directly -- its arithmetic intensity comes from tensor-product
// structure that a linear element barely has (see BENCHMARK.md, Case 1). And the
// mechanics benefits far more, because at order 1 on the SAME mesh its dof count
// falls with the element count, not with the EP dofs.
//
// Equal EP resolution (~59k dofs, 20x7x3 mm slab), mechanics at order 1:
//
//   EP order   mesh        EP dofs   mech dofs   diffusion/step   mechanics/solve
//      1     100x35x15      58,176     174,528       19.4 ms          6.73 s
//      2      50x18x8       63,529      26,163       13.6 ms          0.58 s
//      3      33x12x5       59,200       7,956        9.3 ms          0.21 s
//
// Order 3 is 2.1x faster on diffusion, needs 21.9x fewer mechanics dofs, and is
// 32x faster on the mechanics solve -- about 18x faster per simulated ms overall,
// at the same EP resolution.
//
// And it costs nothing in wavefront quality. At t = 60 ms on the ~59k meshes the
// plateau is [16.42, 23.54] mV at order 1 against [15.77, 23.52] at order 3:
// identical to 0.02 mV at the peak, with no Gibbs overshoot even at h = 0.8 mm.
// The coarse elements are fine because the dofs inside them are not.
//
// At 442k dofs the diffusion ordering is the same: 115.5 ms/step at order 1,
// 86.1 at order 2, 63.0 at order 3, 69.3 at order 4. Order 3 is the sweet spot;
// order 4 buys nothing back.
//
// So prefer order 2-3 on a coarser mesh over order 1 on a fine one:
//
//   ./test_electromechanics -dev cuda -o 3 -mo 1 -nx 67 -ny 23 -nz 10 \
//                           -tf 400 -dt 0.05 -ms 20 -pc amg -pv
//
// Preconditioner memory is the remaining ceiling. AMG's assembled matrix and
// hierarchy cost ~10 kB per mechanics dof: 1.7 GB at 174k dofs, and at 684k it
// peaks at 7834 MiB, i.e. 96% of an 8 GB card. Use -pc amg up to a few hundred
// thousand mechanics dofs and -pc jacobi above that.
//
// The mechanics preconditioner is what decides whether that added cost is
// bearable. Jacobi is matrix-free but its CG count grows with the mesh; AMG
// assembles the Hessian once per Newton iteration and its CG count does not:
//
//   mech dofs    Jacobi CG   AMG CG   Jacobi    AMG
//      12,915        65        12      0.29 s   0.40 s   <- AMG loses when tiny
//      91,611       131        14      3.55 s   1.91 s
//     174,528       165        14      8.64 s   3.74 s
//
// Hence -pc amg is the default, and the gap widens with size. Below ~30k mechanics
// dofs use -pc jacobi; on a single element AMG has nothing to coarsen and both
// slows down and stalls Newton, which is why bench_electromechanics asks for
// Jacobi explicitly. Both preconditioners produce the same displacement to all
// printed digits.
//
// -ms is the other knob. The mechanics are quasi-static, so it changes only how
// finely the deformation is sampled in time, not the EP solution:
//
//   ./test_electromechanics -dev cuda -nx 100 -ny 35 -nz 15 -tf 400 -ms 200 -pv
//
//./test_electromechanics -dev cuda -o 2 -mo 1 -nx 100 -ny 35 -nz 15 \
                        -tf 400 -dt 0.05 -ms 20 -pc amg \
                        -pv -of ./Output/em-tissue-442kEP -csv tissue.csv
//


#include "mfem.hpp"
#include "../lib/reaction_solver.hpp"
#include "../lib/monodomain_solver.hpp"
#include "../lib/hyperelasticity_solver.hpp"
#include "../bc/ep_bchandler.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>

using namespace std;
using namespace mfem;
using namespace mfem::electrophysiology;

// ---------------------------------------------------------------------------
// Problem context
// ---------------------------------------------------------------------------

struct s_MeshContext
{
   // Niederer benchmark slab [mm].
   // 0.5 mm, the coarsest resolution the Niederer benchmark accepts. At 1 mm
   // the transverse diffusivity (sigma_t/sigma_l ~ 1/8) is under-resolved and
   // the wave fails to propagate through the wall at all.
   real_t sx = 20.0; int nx = 40;
   real_t sy = 7.0;  int ny = 14;
   real_t sz = 3.0;  int nz = 6;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
} Mesh_ctx;

struct s_EPContext
{
   // Monodomain conductivities [S/mm], transversely isotropic about the fiber.
   real_t sigma_l = 1.32e-4;   // along the fiber
   real_t sigma_t = 1.70e-5;   // across it
   real_t chi = 1.4e2;         // surface-to-volume ratio [1/mm]
   real_t Cm = 1e-5;           // membrane capacitance [mF/mm^2]
} ep_ctx;

struct s_StimContext
{
   real_t t_start = 1.0;      // [ms]
   // Canonical TP06 protocol: 1 ms. The Niederer benchmark specifies 2 ms, but
   // that is a volumetric current density for a phenomenological model; for a
   // physiological model ReactionSolver passes the amplitude straight into the
   // cell ODE as A/F, and 2 ms of it drives the upstroke to ~72 mV against a
   // physiological peak near 38 mV.
   real_t t_duration = 1.0;   // [ms]
   real_t Iampl = 52.0;       // [A/F] -- the cell model's stimulus units
   real_t cube = 1.5;         // corner cube side [mm]
} stim_ctx;

struct s_FiberContext
{
   // Transmural fiber rotation, endo (z=0) to epi (z=sz), in degrees.
   bool rotating = true;
   real_t angle_endo = -60.0;
   real_t angle_epi = 60.0;
} fiber_ctx;

// ---------------------------------------------------------------------------
// Fibers: one definition, used by both the conduction tensor and the mechanics
// ---------------------------------------------------------------------------

// Unit fiber direction at x. Rotating linearly through the wall thickness is the
// standard rule-based approximation of ventricular architecture; with -fa 0 it
// degenerates to the slab axis, which is the easier case to reason about.
void FiberDirection(const Vector &x, Vector &f0)
{
   f0.SetSize(3);
   if (!fiber_ctx.rotating)
   {
      f0 = 0.0; f0(0) = 1.0;
      return;
   }
   const real_t s = (Mesh_ctx.sz > 0.0) ? (x(2) / Mesh_ctx.sz) : 0.0;
   const real_t deg = fiber_ctx.angle_endo +
                      s * (fiber_ctx.angle_epi - fiber_ctx.angle_endo);
   const real_t th = deg * M_PI / 180.0;
   f0(0) = std::cos(th);
   f0(1) = std::sin(th);
   f0(2) = 0.0;
}

// sigma = sigma_t I + (sigma_l - sigma_t) f0 (x) f0
void ConductivityFunction(const Vector &x, DenseSymmetricMatrix &Sigma)
{
   Vector f0;
   FiberDirection(x, f0);

   const real_t dl = ep_ctx.sigma_l - ep_ctx.sigma_t;
   Sigma = 0.0;
   for (int i = 0; i < 3; i++)
   {
      for (int j = 0; j <= i; j++)
      {
         Sigma(i, j) = (i == j ? ep_ctx.sigma_t : 0.0) + dl * f0(i) * f0(j);
      }
   }
}

// Niederer S1: a corner cube.
real_t StimulationMask(const Vector &x)
{
   const real_t L = stim_ctx.cube;
   const bool inside = (x(0) <= L) && (x(1) <= L) && (x(2) <= L);
   return inside ? stim_ctx.Iampl : 0.0;
}

real_t StimulationAmplitude(real_t t)
{
   return (t >= stim_ctx.t_start && t <= stim_ctx.t_start + stim_ctx.t_duration)
          ? 1.0 : 0.0;
}

namespace
{

/// Global min/max of a true-dof vector, reduced across ranks.
void GlobalRange(const Vector &v, MPI_Comm comm, real_t &lo, real_t &hi)
{
   const real_t l = v.Min(), h = v.Max();
   MPI_Allreduce(&l, &lo, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN, comm);
   MPI_Allreduce(&h, &hi, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
}

/// Largest nodal displacement magnitude, reduced across ranks.
real_t MaxDisplacement(ParGridFunction &d_gf)
{
   d_gf.HostRead();
   const ParFiniteElementSpace *fes = d_gf.ParFESpace();
   const int nd = fes->GetNDofs();          // scalar dofs, byNODES ordering
   real_t local = 0.0;
   for (int i = 0; i < nd; i++)
   {
      real_t m = 0.0;
      for (int c = 0; c < 3; c++)
      {
         const real_t v = d_gf[i + c * nd];
         m += v * v;
      }
      local = std::max(local, std::sqrt(m));
   }
   real_t global = local;
   MPI_Allreduce(&local, &global, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                 fes->GetParMesh()->GetComm());
   return global;
}

} // namespace


int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // ---- options ----------------------------------------------------------
   int order = 1;
   const char *device_config = "cpu";
   const char *form_name = "stress";
   const char *bc_name = "clamped";
   bool pa = true;
   real_t lin_rtol = 1e-6;
   real_t mech_ntol = 1e-6;
   real_t mech_ltol = 1e-4;
   int mech_lmax = 2000;
   real_t dt = 0.05;
   real_t t_final = 400.0;
   int dt_ode = 3;    // dt/dt_ode ~ 0.017 ms, the resolution the TP06 upstroke needs
   int ode_solver_type = 21;
   int mech_stride = 20;
   int fibers_rotate = 1;
   bool paraview = true;
   int save_freq = 10;
   int compression_level = 1;
   const char *outfolder = "./Output/electromechanics-tissue/";
   const char *csv_file = "";
   const char *prec_name = "amg";
   bool run_mech = true;
   int mech_quad_order = -1;   // -1 = 2*order+1
   int mech_order = 1;         // FE order for the displacement/fiber spaces
   HolzapfelParameters mat;    // passive material, see -c/-kappa/-k1/-k2
   int verbose = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&Mesh_ctx.nx, "-nx", "--nx", "Elements along the slab (x).");
   args.AddOption(&Mesh_ctx.ny, "-ny", "--ny", "Elements across the slab (y).");
   args.AddOption(&Mesh_ctx.nz, "-nz", "--nz", "Elements through the wall (z).");
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Uniform serial refinements.");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Uniform parallel refinements.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&device_config, "-dev", "--device",
                  "MFEM device backend: cpu, cuda, hip, ...");
   args.AddOption(&form_name, "-form", "--formulation",
                  "Active formulation: stress, strain or both.");
   args.AddOption(&bc_name, "-bc", "--boundary",
                  "Mechanics BC: 'clamped' (fix x=0) or 'free' (fix x=0 and x=sx).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-fa", "--full-assembly",
                  "Diffusion assembly level.");
   args.AddOption(&lin_rtol, "-rtol", "--linear-rtol",
                  "CG relative tolerance for the diffusion solve.");
   args.AddOption(&mech_ntol, "-ntol", "--newton-rtol",
                  "Newton relative tolerance for the mechanics.");
   args.AddOption(&mech_ltol, "-ltol", "--mech-linear-rtol",
                  "CG relative tolerance inside the mechanics Newton solve.");
   args.AddOption(&mech_lmax, "-lmax", "--mech-linear-maxit",
                  "CG iteration cap inside the mechanics Newton solve.");
   args.AddOption(&dt, "-dt", "--time-step", "Outer time step [ms].");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time [ms].");
   args.AddOption(&dt_ode, "-dode", "--ode-substeps",
                  "ODE substeps per diffusion step. The TP06 upstroke is stiff; "
                  "too few overshoots the peak potential.");
   args.AddOption(&mech_stride, "-ms", "--mechanics-stride",
                  "Solve the mechanics every N diffusion steps. The mechanics are "
                  "quasi-static, so this only sets how finely the deformation is "
                  "sampled in time.");
   args.AddOption(&fibers_rotate, "-fa", "--fiber-rotation",
                  "1 = transmural fiber rotation (default), 0 = uniform along x.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable ParaView output.");
   args.AddOption(&save_freq, "-sf", "--save-freq",
                  "Save every N mechanics solves.");
   args.AddOption(&mat.c, "-c", "--matrix-stiffness",
                  "Holzapfel isotropic matrix stiffness c [kPa].");
   args.AddOption(&mat.kappa, "-kappa", "--bulk-penalty",
                  "Holzapfel bulk penalty kappa [kPa]. kappa/c sets how close to "
                  "incompressible the tissue is, and therefore how badly a low-order "
                  "displacement space volumetrically locks.");
   args.AddOption(&mat.k1, "-k1", "--fiber-stiffness",
                  "Holzapfel fiber stiffness k1 [kPa].");
   args.AddOption(&mat.k2, "-k2", "--fiber-exponent",
                  "Holzapfel fiber exponential stiffening k2 [-].");
   args.AddOption(&mech_order, "-mo", "--mech-order",
                  "FE order for the mechanics (displacement and fibers). Coarser than "
                  "the EP order is the point: the deformation field is far smoother "
                  "than the activation front, and the dFEM derivative caches scale with "
                  "the mechanics dofs. The active field stays on the EP space, so there "
                  "is no transfer -- dFEM interpolates each field to the shared "
                  "quadrature rule independently.");
   args.AddOption(&mech_quad_order, "-qo", "--mech-quad-order",
                  "Integration rule order for the mechanics (-1 = 2*order+1). The dFEM "
                  "derivative caches scale with the quadrature point count, so this is "
                  "the direct knob on mechanics memory.");
   args.AddOption(&run_mech, "-mech", "--mechanics", "-no-mech", "--no-mechanics",
                  "Build and solve the mechanics. -no-mech leaves a pure-EP run, which "
                  "is the apples-to-apples comparison against test_monodomain: the "
                  "mechanics operator allocates its quadrature caches at construction, "
                  "so on a small card it can starve the EP solve even when never solved.");
   args.AddOption(&prec_name, "-pc", "--preconditioner",
                  "Mechanics preconditioner: amg (default), jacobi or none.");
   args.AddOption(&compression_level, "-cl", "--compression-level",
                  "zlib level for ParaView data (0 = off).");
   args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");
   args.AddOption(&csv_file, "-csv", "--csv-output",
                  "Write the transient to this CSV (one file per formulation).");
   args.AddOption(&verbose, "-v", "--verbose", "0 quiet, 1 progress, 2 + solvers.");
   args.ParseCheck();

   fiber_ctx.rotating = (fibers_rotate != 0);

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   std::vector<ActiveFormulation> forms;
   const std::string fsel(form_name);
   if (fsel == "stress" || fsel == "both")
   {
      forms.push_back(ActiveFormulation::ActiveStress);
   }
   if (fsel == "strain" || fsel == "both")
   {
      forms.push_back(ActiveFormulation::ActiveStrain);
   }
   MFEM_VERIFY(!forms.empty(), "-form must be stress, strain or both");

   // ---- mesh -------------------------------------------------------------
   Mesh serial_mesh = Mesh::MakeCartesian3D(
                         Mesh_ctx.nx, Mesh_ctx.ny, Mesh_ctx.nz, Element::HEXAHEDRON,
                         Mesh_ctx.sx, Mesh_ctx.sy, Mesh_ctx.sz, true);
   for (int l = 0; l < Mesh_ctx.serial_ref_levels; l++)
   {
      serial_mesh.UniformRefinement();
   }
   serial_mesh.EnsureNodes();

   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }
   mesh.EnsureNodes();

   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 3, "The tissue electromechanics test is 3D only.");

   for (auto formulation : forms)
   {
      const char *tag = (formulation == ActiveFormulation::ActiveStress)
                        ? "EM-active-stress" : "EM-active-strain";
      if (Mpi::Root())
      {
         mfem::out << "\n=== " << tag << " ===" << endl;
      }

      // ---- spaces --------------------------------------------------------
      H1_FECollection fec(order, dim);
      ParFiniteElementSpace fes(&mesh, &fec);          // scalar: V, Ta

      // The mechanics lives on the same mesh but may use a lower order. dFEM
      // interpolates every registered field to the shared quadrature rule
      // independently, so mixing orders needs no transfer operator: the active
      // field stays on the EP space and is read at the quadrature points.
      // Not clamped to the EP order: at FIXED dof count a higher-order
      // mechanics space has fewer elements, hence a smaller per-element
      // E-vector working set, which is what dominates dFEM memory.
      const int mo = std::max(1, mech_order);
      H1_FECollection mech_fec(mo, dim);
      ParFiniteElementSpace vfes(&mesh, &mech_fec, dim);   // vector: u, f0

      if (Mpi::Root())
      {
         mfem::out << "EP dofs:        " << fes.GlobalTrueVSize()
                   << "  (order " << order << ", scalar)\n"
                   << "Mechanics dofs: " << vfes.GlobalTrueVSize()
                   << "  (order " << mo << ", vdim " << dim << ")" << endl;
      }

      // ---- coefficients --------------------------------------------------
      ConstantCoefficient chi_coeff(ep_ctx.chi);
      ConstantCoefficient Cm_coeff(ep_ctx.Cm);
      SymmetricMatrixFunctionCoefficient sigma_coeff(dim, ConductivityFunction);

      // ---- diffusion (PDE) -----------------------------------------------
      auto *bc = new BCHandler(&mesh);   // solver takes ownership
      MonodomainDiffusionSolver diff_solver(&fes, bc, &sigma_coeff, &chi_coeff,
                                            &Cm_coeff, ode_solver_type,
                                            verbose > 1);
      diff_solver.EnablePA(pa);
      diff_solver.Setup(dt, 0, lin_rtol, true);

      // ---- reaction (ODEs) + contraction ---------------------------------
      ReactionSolver reaction_solver(&fes, &chi_coeff, &Cm_coeff,
                                     IonicModelType::TENTUSSCHER_PANFILOV_EPI,
                                     TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN,
                                     dt_ode);
      reaction_solver.RegisterModels(IonicModelType::TENTUSSCHER_PANFILOV_EPI,
                                     ContractionModelType::LAND_2017);

      std::vector<real_t> initial_states, parameters;
      reaction_solver.GetDefaultStates(initial_states);
      reaction_solver.GetDefaultParameters(parameters);
      reaction_solver.GetModel()->DisableInternalTimeManagement(parameters.data());
      reaction_solver.Setup(initial_states, parameters);

      // Separable stimulus: the corner cube is projected once and each step only
      // rescales it, instead of re-projecting a space-time coefficient.
      FunctionCoefficient stim_mask(StimulationMask);
      reaction_solver.SetSeparableStimulation(&stim_mask, StimulationAmplitude);
      reaction_solver.SetStimulationWindow(stim_ctx.t_start,
                                           stim_ctx.t_start + stim_ctx.t_duration);
      reaction_solver.EnableSubstepStimulusProjection(true);

      // ---- mechanics ------------------------------------------------------
      // Matched to the MECHANICS order, not the EP order. Deriving it from the EP
      // order is a trap once the two differ: EP order 3 with mechanics order 1
      // gives a 7th-order rule, 64 points per element instead of 8, and every one
      // of them costs an Enzyme-differentiated energy evaluation on every Hessian
      // apply. The active field is order 3 inside the element, so a richer rule
      // does integrate it more faithfully -- raise -qo if that matters more than
      // the 8x -- but the displacement is what the Newton solve is resolving.
      const int qorder = (mech_quad_order > 0) ? mech_quad_order : 2 * mo + 1;
      const IntegrationRule &ir =
         IntRules.Get(mesh.GetTypicalElementGeometry(), qorder);
      if (Mpi::Root())
      {
         mfem::out << "Mechanics quadrature: order " << qorder << ", "
                   << ir.GetNPoints() << " points/element" << endl;
      }

      VectorFunctionCoefficient f0_coeff(dim, FiberDirection);

      std::unique_ptr<HyperelasticitySolver> mech;
      if (run_mech)
      {
         mech = std::make_unique<HyperelasticitySolver>(vfes, fes, vfes, ir,
                                                        formulation, mat);
         const std::string pc(prec_name);
         if (pc == "amg")         { mech->SetPreconditioner(MechanicsPreconditioner::AMG); }
         else if (pc == "jacobi") { mech->SetPreconditioner(MechanicsPreconditioner::Jacobi); }
         else if (pc == "none")   { mech->SetPreconditioner(MechanicsPreconditioner::None); }
         else { MFEM_ABORT("-pc must be amg, jacobi or none"); }

         mech->SetNewtonOptions(25, mech_ntol, verbose > 1 ? 1 : -1);
         mech->SetLinearSolverOptions(mech_lmax, mech_ltol, verbose > 2 ? 1 : -1);
         mech->SetFiberField(f0_coeff);

         // Attribute 1 is x = 0 and attribute 2 is x = sx for MakeCartesian3D.
         Array<int> ess_bdr(mesh.bdr_attributes.Max());
         ess_bdr = 0;
         ess_bdr[0] = 1;                                    // always clamp x = 0
         if (std::string(bc_name) == "free") { ess_bdr[1] = 1; }  // also clamp x = sx
         mech->SetEssentialAttributes(ess_bdr);
      }

      // ---- output ---------------------------------------------------------
      auto *u_gf = diff_solver.GetPotentialGf();
      Vector u;
      reaction_solver.GetPotential(u);
      u_gf->SetFromTrueDofs(u);

      ParGridFunction Ta_gf(&fes);
      Ta_gf = 0.0;
      ParGridFunction fiber_gf(&vfes);
      fiber_gf.ProjectCoefficient(f0_coeff);

      const int ca_idx = reaction_solver.GetModel()->GetCalciumIndex();
      reaction_solver.SyncStateGridFunctions();
      ParGridFunction *ca_gf = reaction_solver.GetStateGridFunction(ca_idx);

      ParaViewDataCollection pvdc(tag, &mesh);
      pvdc.SetPrefixPath(outfolder);
      pvdc.SetDataFormat(VTKFormat::BINARY32);
      pvdc.SetCompression(compression_level != 0);
      pvdc.SetCompressionLevel(compression_level);
      if (order > 1)
      {
         pvdc.SetHighOrderOutput(true);
         pvdc.SetLevelsOfDetail(order);
      }
      pvdc.RegisterField("potential", u_gf);
      pvdc.RegisterField("calcium", ca_gf);
      pvdc.RegisterField("active_tension", &Ta_gf);
      if (mech) { pvdc.RegisterField("displacement", &mech->GetDisplacement()); }
      pvdc.RegisterField("fibers", &fiber_gf);
      if (paraview)
      {
         pvdc.SetCycle(0);
         pvdc.SetTime(0.0);
         pvdc.Save();
      }

      std::ofstream csv;
      if (csv_file && csv_file[0] && Mpi::Root())
      {
         std::string base(csv_file);
         const size_t dot = base.find_last_of('.');
         const std::string stem = (dot == std::string::npos) ? base : base.substr(0, dot);
         const std::string ext = (dot == std::string::npos) ? ".csv" : base.substr(dot);
         csv.open(stem + "-" + tag + ext);
         csv << "# tissue electromechanics -- " << tag << "\n"
             << "time_ms,Vm_min_mV,Vm_max_mV,peak_calcium_mM,peak_Ta_kPa,"
                "max_disp_mm,newton_iters\n";
         csv << std::scientific << std::setprecision(8);
      }

      // ---- staggered driver loop ------------------------------------------
      StopWatch chrono_total, chrono;
      chrono_total.Start();
      real_t t_diffusion = 0.0, t_reaction = 0.0, t_mech = 0.0, t_io = 0.0;

      real_t t = 0.0;
      int step = 0, mech_solves = 0;
      real_t peak_Ta = 0.0, peak_disp = 0.0, peak_ca = 0.0;
      bool all_converged = true;
      Vector gamma;

      while (t < t_final - 1e-12)
      {
         real_t dt_step = std::min(dt, t_final - t);

         // 1. EP diffusion (PDE)
         chrono.Clear(); chrono.Start();
         diff_solver.Step(u, t, dt_step, true);
         chrono.Stop(); t_diffusion += chrono.RealTime();

         // 2. EP reaction (ODEs); caches dt, invalidates the tension
         chrono.Clear(); chrono.Start();
         reaction_solver.Step(u, t, dt_step, true);
         chrono.Stop(); t_reaction += chrono.RealTime();

         diff_solver.UpdateTimeStepHistory(u);
         t += dt_step;
         step++;

         // Per-step EP line. Only the potential range: the tension would cost
         // a contraction solve, and the calcium a device sync of the state
         // grid functions, so both stay on the mechanics line below.
         real_t vlo_ep, vhi_ep;
         GlobalRange(u, fes.GetComm(), vlo_ep, vhi_ep);
         if (Mpi::Root() && verbose)
         {
            mfem::out << std::fixed << std::setprecision(2)
                      << "  step " << std::setw(6) << step
                      << "   t = " << std::setw(8) << t << " ms"
                      << "   Vm [" << std::setw(8) << vlo_ep << ", "
                      << std::setw(7) << vhi_ep << "] mV" << endl;
         }

         if (step % mech_stride != 0) { continue; }

         if (Mpi::Root() && verbose)
         {
            mfem::out << "    Solving mechanics ..." << endl;
         }

         chrono.Clear(); chrono.Start();

         // 3. Contraction (on demand)
         const Vector &Ta = reaction_solver.GetActiveTension();

         // 4. Update mechanics with the frozen activation
         if (mech)
         {
            if (formulation == ActiveFormulation::ActiveStress)
            {
               mech->UpdateActiveField(Ta);
            }
            else
            {
               HyperelasticitySolver::ShorteningFromTension(Ta, gamma);
               mech->UpdateActiveField(gamma);
            }
         }

         // 5. Solve equilibrium
         if (mech) { mech->SolveQuasiStaticStep(); }
         chrono.Stop(); t_mech += chrono.RealTime();

         if (mech && !mech->GetConverged())
         {
            all_converged = false;
            if (Mpi::Root() && verbose)
            {
               mfem::out << "  Newton did NOT converge at t = " << t << endl;
            }
         }
         mech_solves++;

         // ---- diagnostics ----
         reaction_solver.SyncStateGridFunctions();
         u_gf->SetFromTrueDofs(u);
         Ta_gf.SetFromTrueDofs(Ta);

         real_t vlo, vhi, ta_lo, ta_hi;
         GlobalRange(u, fes.GetComm(), vlo, vhi);
         GlobalRange(Ta, fes.GetComm(), ta_lo, ta_hi);
         const real_t disp = mech ? MaxDisplacement(mech->GetDisplacement()) : 0.0;
         const real_t ca = ca_gf->Max();

         peak_Ta = std::max(peak_Ta, ta_hi);
         peak_disp = std::max(peak_disp, disp);
         peak_ca = std::max(peak_ca, ca);

         if (csv.is_open())
         {
            csv << t << "," << vlo << "," << vhi << "," << ca << ","
                << ta_hi << "," << disp << "," << (mech ? mech->GetNumNewtonIterations() : 0)
                << "\n";
         }

         if (Mpi::Root() && verbose)
         {
            mfem::out << std::fixed << std::setprecision(2)
                      << "    -> [Ca]_max = " << std::scientific << std::setprecision(3)
                      << ca << " mM" << std::fixed << std::setprecision(2)
                      << "   Ta_max = " << std::setw(7) << ta_hi << " kPa"
                      << "   |u|_max = " << std::setw(7) << disp << " mm"
                      << "   newton " << (mech ? mech->GetNumNewtonIterations() : 0)
                      << "  cg " << (mech ? mech->GetNumLinearIterations() : 0) << endl;
         }

         chrono.Clear(); chrono.Start();
         if (paraview && (mech_solves % save_freq == 0))
         {
            pvdc.SetCycle(mech_solves);
            pvdc.SetTime(t);
            pvdc.Save();
         }
         chrono.Stop(); t_io += chrono.RealTime();
      }

      chrono_total.Stop();

      if (Mpi::Root())
      {
         mfem::out << std::scientific << std::setprecision(4)
                   << "\n  peak [Ca2+]i  = " << peak_ca << " mM\n"
                   << "  peak Ta       = " << peak_Ta << " kPa\n"
                   << "  peak |u|      = " << peak_disp << " mm\n"
                   << "  Newton        = " << (all_converged ? "converged everywhere"
                                               : "FAILED at least once") << "\n"
                   << std::fixed << std::setprecision(3)
                   << "\n  diffusion " << t_diffusion << " s"
                   << " | reaction " << t_reaction << " s"
                   << " | mechanics " << t_mech << " s"
                   << " | I/O " << t_io << " s"
                   << " | total " << chrono_total.RealTime() << " s\n"
                   << "  (" << step << " EP steps, " << mech_solves
                   << " mechanics solves)" << endl;
      }
   }

   return 0;
}
