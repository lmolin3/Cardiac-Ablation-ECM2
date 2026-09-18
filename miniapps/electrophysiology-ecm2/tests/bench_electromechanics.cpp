//        One-way coupled cardiac electromechanics -- single element benchmark
//
// Verification suite for the staggered EP -> contraction -> mechanics loop,
// on a single hexahedron (or a small cube of them). This is a correctness
// benchmark, not a tissue simulation -- see test_electromechanics for a
// realistic tissue-level slab.
//
//   Test A  Pure EP: register no contraction model and check that no
//           contraction/mechanics storage is allocated at all.
//   Test B  Fail fast: pair an EP model without [Ca2+]i dynamics
//           (Mitchell-Schaeffer) with the calcium-driven Land 2017 model and
//           check that registration throws instead of silently producing zero
//           tension.
//   Test C  Single element benchmark: a unit cube with a fixed base, stimulated
//           so the whole cube activates. Verifies that active tension is
//           generated and that the cube shortens along the fiber direction f0,
//           for both the active stress and the active strain formulation.
//
// ---------------------------------------------------------------------------
// Sample runs
// ---------------------------------------------------------------------------
//
// Backend selection is the -d/--device flag, and it is the ONLY thing that has
// to change between a host run and a GPU run: the solvers keep their state in
// mfem::Vector, so the memory manager moves it. Both backends must produce the
// same numbers -- that is part of what this suite checks.
//
//   CPU, 4 ranks:       mpirun -np 4 ./bench_electromechanics -t all -d cpu
//   GPU (CUDA):         mpirun -np 1 ./bench_electromechanics -t all -d cuda
//   GPU (HIP):          mpirun -np 1 ./bench_electromechanics -t all -d hip
//   GPU, 1 rank/GPU:    mpirun -np 2 ./bench_electromechanics -t all -d cuda
//
// Individual tests:
//   Pure EP allocation:      ./bench_electromechanics -t a
//   Fail-fast pairing:       ./bench_electromechanics -t b
//   Benchmark, both forms:   ./bench_electromechanics -t c
//   Active stress only:      ./bench_electromechanics -t c -form stress
//   Active strain only:      ./bench_electromechanics -t c -form strain
//
// Watching the contraction (see "Visualizing" below):
//   Refined cube + output:   ./bench_electromechanics -t c -form stress -n 8 -pv
//   Full twitch, GPU:        ./bench_electromechanics -t c -d cuda -n 8 -tf 500 -pv
//   Transient as CSV:        ./bench_electromechanics -t c -csv twitch.csv
//   Both forms, compare:     ./bench_electromechanics -t c -n 8 -pv -csv twitch.csv
//
// Convergence / debugging:
//   Newton residuals:        ./bench_electromechanics -t c -v 1
//   Newton + CG residuals:   ./bench_electromechanics -t c -v 2
//   Cheaper mechanics:       ./bench_electromechanics -t c -ms 10   (solve every 10 steps)
//   Higher order:            ./bench_electromechanics -t c -o 2 -n 4
//
// ---------------------------------------------------------------------------
// Visualizing the contraction
// ---------------------------------------------------------------------------
//
// -pv writes a ParaView collection per formulation into
// <output-folder>/em-active-stress/ (resp. em-active-strain/), carrying four
// fields: potential, calcium, active_tension and displacement.
//
// The mesh written is the REFERENCE configuration -- displacement is stored as
// a field on it rather than baked into the node positions, so the EP fields
// stay comparable across time steps. To actually watch the cube contract:
//
//   1. Open <output-folder>/em-active-stress/em-active-stress.pvd
//   2. Filters -> Alphabetical -> "Warp By Vector"
//   3. Vectors = displacement, Scale Factor = 1
//   4. Colour by active_tension, then press Play
//
// The cube shortens along x (the fiber direction f0) and bulges transversely.
// Raise the scale factor to exaggerate the motion on a stiff parameter set.
//
// Use -n 8 for visualization: the default -n 1 is a single element, which is
// the right size for a verification benchmark but has too few nodes to look
// like anything.
//
// -csv writes one row per mechanics solve -- time, peak calcium, peak tension,
// free-face shortening and Newton iterations -- which is the quickest way to
// see the twitch itself: tension rises well behind the calcium transient, and
// shortening tracks tension. Plot it with e.g.
//
//   python3 -c "import csv,sys,matplotlib.pyplot as plt; \
//     r=[l for l in csv.DictReader(open('twitch-em-active-stress.csv'))]; \
//     t=[float(x['time_ms']) for x in r]; \
//     plt.plot(t,[float(x['peak_Ta_kPa']) for x in r],label='Ta [kPa]'); \
//     plt.plot(t,[-100*float(x['free_face_ux_mm']) for x in r],label='shortening [%]'); \
//     plt.xlabel('t [ms]'); plt.legend(); plt.show()"
//

#include "mfem.hpp"
#include "../lib/monodomain_solver.hpp"
#include "../lib/reaction_solver.hpp"
#include "../lib/hyperelasticity_solver.hpp"
#include "../bc/ep_bchandler.hpp"

#include <iostream>
#include <iomanip>
#include <memory>
#include <fstream>
#include <string>

using namespace std;
using namespace mfem;
using namespace mfem::electrophysiology;

namespace
{

int failures = 0;

void Check(bool ok, const std::string &what)
{
   if (Mpi::Root())
   {
      mfem::out << (ok ? "  [ PASS ] " : "  [ FAIL ] ") << what << endl;
   }
   if (!ok) { failures++; }
}

// The cube is fixed on x = 0 and the fibers run along x, so a contracting cell
// pulls the free face at x = 1 back towards the base: u_x < 0 there.
void FiberDirection(const Vector &, Vector &f0)
{
   f0 = 0.0;
   f0(0) = 1.0;
}

/**
 * @brief Most-negative fiber-direction displacement on the free face x = 1.
 *
 * This is the scalar that says "the cube contracted": with the base clamped and
 * f0 along x, shortening shows up as u_x < 0 on the opposite face. Sampled at
 * the element nodes lying on that face and reduced across ranks.
 */
/// Global min/max of a true-dof vector, reduced across ranks.
void GlobalRange(const Vector &v, MPI_Comm comm, real_t &lo, real_t &hi)
{
   const real_t l = v.Min(), h = v.Max();
   MPI_Allreduce(&l, &lo, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN, comm);
   MPI_Allreduce(&h, &hi, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
}

real_t FreeFaceFiberDisplacement(ParGridFunction &d_gf)
{
   d_gf.HostRead();
   const ParFiniteElementSpace *dfes = d_gf.ParFESpace();
   ParMesh *pm = dfes->GetParMesh();

   real_t local_min = 0.0;
   bool found = false;
   Array<int> vdofs;
   for (int e = 0; e < pm->GetNE(); e++)
   {
      dfes->GetElementVDofs(e, vdofs);
      const FiniteElement *fe = dfes->GetFE(e);
      ElementTransformation *tr = pm->GetElementTransformation(e);
      const IntegrationRule &nodes = fe->GetNodes();
      const int ndof = fe->GetDof();
      for (int i = 0; i < ndof; i++)
      {
         Vector phys(3);
         tr->Transform(nodes.IntPoint(i), phys);
         if (std::abs(phys(0) - 1.0) < 1e-10)
         {
            const int vd = vdofs[i];   // x component, byNODES ordering
            const real_t val = d_gf[vd >= 0 ? vd : -1 - vd];
            local_min = found ? std::min(local_min, val) : val;
            found = true;
         }
      }
   }
   real_t global_min = local_min;
   MPI_Allreduce(&local_min, &global_min, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, pm->GetComm());
   return global_min;
}

} // namespace


// ---------------------------------------------------------------------------
// Test A: a pure-EP run must not allocate any contraction state.
// ---------------------------------------------------------------------------
void TestA_PureEP(ParFiniteElementSpace &fes, Coefficient &chi, Coefficient &Cm)
{
   if (Mpi::Root()) { mfem::out << "\nTest A -- pure EP, no mechanics allocation\n"; }

   ReactionSolver rs(&fes, &chi, &Cm, IonicModelType::TENTUSSCHER_PANFILOV_EPI,
                     TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN, 1);

   // Explicitly register with no contraction model.
   rs.RegisterModels(IonicModelType::TENTUSSCHER_PANFILOV_EPI,
                     ContractionModelType::NONE);
   rs.Setup();

   Check(!rs.HasContractionModel(), "no contraction model registered");
   Check(rs.GetContractionAllocatedEntries() == 0,
         "zero contraction entries allocated (got " +
         std::to_string(rs.GetContractionAllocatedEntries()) + ")");

   // The EP side must still work normally.
   Vector u(fes.GetTrueVSize());
   rs.GetPotential(u);
   real_t t = 0.0, dt = 0.1;
   rs.Step(u, t, dt, true);
   Check(u.Size() == fes.GetTrueVSize(), "EP step runs without a contraction model");
}


// ---------------------------------------------------------------------------
// Test B: an EP model without calcium cannot drive a calcium-based contraction
// model, and that must be caught at registration.
// ---------------------------------------------------------------------------
void TestB_IncompatiblePairing(ParFiniteElementSpace &fes,
                               Coefficient &chi, Coefficient &Cm)
{
   if (Mpi::Root()) { mfem::out << "\nTest B -- incompatible pairing throws\n"; }

   ReactionSolver rs(&fes, &chi, &Cm, IonicModelType::MITCHELL_SCHAEFFER,
                     TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN, 1);

   // Sanity: Mitchell-Schaeffer is phenomenological and has no [Ca2+]i.
   MitchellSchaeffer ms;
   Check(!ms.HasCalcium(), "Mitchell-Schaeffer reports no calcium");
   Land17Model land;
   Check(land.RequiresCalcium(), "Land 2017 reports it requires calcium");

   bool threw = false;
   std::string message;
   try
   {
      rs.RegisterModels(IonicModelType::MITCHELL_SCHAEFFER,
                        ContractionModelType::LAND_2017);
   }
   catch (const std::runtime_error &e)
   {
      threw = true;
      message = e.what();
   }

   Check(threw, "RegisterModels() threw std::runtime_error");
   Check(message.find("requires [Ca2+]i") != std::string::npos,
         "exception explains the missing calcium: \"" + message + "\"");

   // The compatible pairing must NOT throw.
   bool ok = true;
   try
   {
      ReactionSolver rs2(&fes, &chi, &Cm, IonicModelType::TENTUSSCHER_PANFILOV_EPI,
                         TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN, 1);
      rs2.RegisterModels(IonicModelType::TENTUSSCHER_PANFILOV_EPI,
                         ContractionModelType::LAND_2017);
      rs2.Setup();
      ok = rs2.HasContractionModel() && rs2.GetContractionAllocatedEntries() > 0;
   }
   catch (const std::exception &)
   {
      ok = false;
   }
   Check(ok, "TP06 + Land 2017 registers and allocates contraction state");
}


// ---------------------------------------------------------------------------
// Test C: single element benchmark, driven by the full staggered loop.
// ---------------------------------------------------------------------------
// All three scalars are extrema over the whole run, sampled at every mechanics
// solve, so they describe the twitch rather than whatever the state happened to
// be at t_final (calcium and tension are both well into their decay by then).
struct BenchmarkResult
{
   real_t Vm_min = 1e300;           // resting potential reached [mV]
   real_t Vm_max = -1e300;          // upstroke peak reached [mV]
   real_t peak_Ta = 0.0;            // max active tension [kPa]
   real_t peak_calcium = 0.0;       // max [Ca2+]i [mM]
   real_t min_free_face_ux = 0.0;   // most negative fiber-direction displacement [mm]
   bool newton_converged = true;
   real_t clamp_lo = 0.0, clamp_hi = 0.0;   // the range Vm was clamped to
};

BenchmarkResult TestC_SingleElement(ParMesh &mesh, int order,
                                    ActiveFormulation formulation,
                                    real_t t_final, real_t dt, int mech_stride,
                                    bool paraview, const char *outfolder,
                                    int verbose, const char *csv_file,
                                    int ode_substeps)
{
   BenchmarkResult result;

   const int dim = mesh.Dimension();

   // ---- EP spaces and coefficients -------------------------------------
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&mesh, &fec);              // scalar: potential, Ta
   ParFiniteElementSpace vfes(&mesh, &fec, dim);        // vector: displacement, fibers

   ConstantCoefficient chi_coeff(140.0);   // [1/mm]
   ConstantCoefficient Cm_coeff(0.01);     // [uF/mm^2]

   // Isotropic conductivity; on a single element the diffusion term is nearly
   // inactive, but the loop is exercised as specified.
   DenseMatrix sigma(dim);
   sigma = 0.0;
   for (int d = 0; d < dim; d++) { sigma(d, d) = 0.1; }
   MatrixConstantCoefficient sigma_coeff(sigma);

   // ---- Diffusion (PDE) solver -----------------------------------------
   auto *bc = new BCHandler(&mesh);   // solver takes ownership
   MonodomainDiffusionSolver diff_solver(&fes, bc, &sigma_coeff, &chi_coeff,
                                         &Cm_coeff, 21, false);
   diff_solver.Setup(dt, 0, 1e-8, true);

   // ---- Reaction (ODE) solver, EP + contraction -------------------------
   ReactionSolver reaction_solver(&fes, &chi_coeff, &Cm_coeff,
                                  IonicModelType::TENTUSSCHER_PANFILOV_EPI,
                                  TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN,
                                  ode_substeps);
   reaction_solver.RegisterModels(IonicModelType::TENTUSSCHER_PANFILOV_EPI,
                                  ContractionModelType::LAND_2017);

   std::vector<real_t> initial_states, parameters;
   reaction_solver.GetDefaultStates(initial_states);
   reaction_solver.GetDefaultParameters(parameters);
   // The stimulation is driven externally through the coefficient below, so the
   // model's own pulse train is switched off.
   reaction_solver.GetModel()->DisableInternalTimeManagement(parameters.data());
   reaction_solver.Setup(initial_states, parameters);
   // Recorded after Setup(), which is where the model's own range is adopted.
   reaction_solver.GetVRange(result.clamp_lo, result.clamp_hi);

   // Whole-domain stimulus, one pulse of 2 ms at 52 A/F (the TP06 default).
   // Canonical TP06 protocol: 52 A/F for 1 ms. The duration is not a free
   // knob -- the stimulus current enters dV/dt directly, so doubling it
   // injects ~52 mV of extra depolarisation and drives the peak well past
   // the physiological +40 mV.
   const real_t stim_amp = 52.0, stim_start = 1.0, stim_duration = 1.0;
   auto stim_fn = [=](const Vector &, real_t t) -> real_t
   {
      return (t >= stim_start && t <= stim_start + stim_duration) ? stim_amp : 0.0;
   };
   FunctionCoefficient Istim_coeff(stim_fn);
   reaction_solver.SetStimulation(&Istim_coeff, false);
   reaction_solver.SetStimulationWindow(stim_start, stim_start + stim_duration);
   // Sample the pulse at every ODE substep. By default the stimulus is
   // projected once per outer step, which rounds a 1 ms window up to the
   // 0.5 ms grid and over-delivers ~50% of the charge -- enough to push the
   // upstroke tens of mV past its physiological peak. The declared window
   // above keeps the extra projections confined to the pulse itself.
   reaction_solver.EnableSubstepStimulusProjection(true);

   // ---- Mechanics solver -------------------------------------------------
   const IntegrationRule &ir =
      IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * order + 1);

   HyperelasticitySolver mech(vfes, fes, vfes, ir, formulation);
   // Jacobi, not the solver's AMG default. This benchmark is a single element
   // (81 mechanics dofs); AMG has nothing to coarsen there, so it is both slower
   // than Jacobi and less robust -- it stalls Newton at peak tension, where the
   // problem is most nonlinear. AMG earns its keep at tissue scale instead (see
   // test_electromechanics), where the CG iteration count stops growing.
   mech.SetPreconditioner(MechanicsPreconditioner::Jacobi);
   mech.SetNewtonOptions(25, 1e-8, verbose ? 1 : -1);
   mech.SetLinearSolverOptions(2000, 1e-8, verbose > 1 ? 1 : -1);

   VectorFunctionCoefficient f0_coeff(dim, FiberDirection);
   mech.SetFiberField(f0_coeff);

   // Fix the base x = 0 (boundary attribute 1 of MakeCartesian3D).
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   mech.SetEssentialAttributes(ess_bdr);

   // ---- Output -----------------------------------------------------------
   Vector u(fes.GetTrueVSize());
   reaction_solver.GetPotential(u);
   ParGridFunction u_gf(&fes);
   u_gf.Distribute(u);
   ParGridFunction Ta_gf(&fes);
   Ta_gf = 0.0;

   // Calcium is the quantity that actually couples the two physics, so it is
   // worth seeing next to the tension it produces.
   const int ca_idx = reaction_solver.GetModel()->GetCalciumIndex();
   reaction_solver.SyncStateGridFunctions();
   ParGridFunction *ca_gf = reaction_solver.GetStateGridFunction(ca_idx);

   const char *tag = (formulation == ActiveFormulation::ActiveStress)
                     ? "em-active-stress" : "em-active-strain";

   std::unique_ptr<ParaViewDataCollection> pvdc;
   if (paraview)
   {
      pvdc = std::make_unique<ParaViewDataCollection>(tag, &mesh);
      pvdc->SetPrefixPath(outfolder);
      pvdc->SetDataFormat(VTKFormat::BINARY);
      pvdc->SetHighOrderOutput(order > 1);
      pvdc->SetLevelsOfDetail(order);
      pvdc->RegisterField("potential", &u_gf);
      pvdc->RegisterField("calcium", ca_gf);
      pvdc->RegisterField("active_tension", &Ta_gf);
      // Stored as a field on the reference mesh; apply ParaView's "Warp By
      // Vector" filter on it to see the cube actually deform.
      pvdc->RegisterField("displacement", &mech.GetDisplacement());
      pvdc->SetCycle(0);
      pvdc->SetTime(0.0);
      pvdc->Save();
   }

   std::ofstream csv;
   if (csv_file && csv_file[0] && Mpi::Root())
   {
      // One file per formulation, so running "-form both -csv twitch.csv"
      // leaves twitch-active-stress.csv and twitch-active-strain.csv.
      std::string base(csv_file);
      const size_t dot = base.find_last_of('.');
      const std::string stem = (dot == std::string::npos) ? base : base.substr(0, dot);
      const std::string ext = (dot == std::string::npos) ? ".csv" : base.substr(dot);
      csv.open(stem + "-" + tag + ext);
      csv << "# one-way coupled cardiac electromechanics -- " << tag << "\n"
          << "time_ms,Vm_min_mV,Vm_max_mV,peak_calcium_mM,peak_Ta_kPa,free_face_ux_mm,newton_iters\n";
      csv << std::scientific << std::setprecision(8);
   }

   // -----------------------------------------------------------------------
   //  Task 5.1: the staggered multi-physics driver loop
   // -----------------------------------------------------------------------
   real_t t = 0.0;
   int step = 0;
   Vector gamma;   // only used by the active strain formulation

   while (t < t_final - 1e-12)
   {
      real_t dt_step = std::min(dt, t_final - t);

      // 1. EP diffusion (PDE)
      diff_solver.Step(u, t, dt_step, true);

      // 2. EP reaction (ODEs); caches dt and invalidates the tension
      reaction_solver.Step(u, t, dt_step, true);

      step++;
      t += dt_step;

      if (step % mech_stride != 0) { continue; }

      // 3. Contraction (on demand): integrates the Land ODEs once per step
      const Vector &Ta = reaction_solver.GetActiveTension();

      // 4. Update mechanics with the frozen activation
      if (formulation == ActiveFormulation::ActiveStress)
      {
         mech.UpdateActiveField(Ta);
      }
      else
      {
         HyperelasticitySolver::ShorteningFromTension(Ta, gamma);
         mech.UpdateActiveField(gamma);
      }

      // 5. Solve equilibrium
      mech.SolveQuasiStaticStep();
      if (!mech.GetConverged())
      {
         result.newton_converged = false;
         if (verbose && Mpi::Root())
         {
            mfem::out << "    Newton did NOT converge at t = " << t
                      << " after " << mech.GetNumNewtonIterations()
                      << " iterations\n";
         }
      }

      // ---- diagnostics ----
      // The state grid functions mirror the device state, so they have to be
      // refreshed before anything reads calcium back.
      reaction_solver.SyncStateGridFunctions();

      const real_t ux = FreeFaceFiberDisplacement(mech.GetDisplacement());
      real_t vlo, vhi;
      GlobalRange(u, fes.GetComm(), vlo, vhi);
      result.Vm_min = std::min(result.Vm_min, vlo);
      result.Vm_max = std::max(result.Vm_max, vhi);
      result.peak_Ta = std::max(result.peak_Ta, Ta.Max());
      result.peak_calcium = std::max(result.peak_calcium, ca_gf->Max());
      result.min_free_face_ux = std::min(result.min_free_face_ux, ux);

      if (csv.is_open())
      {
         csv << t << "," << vlo << "," << vhi << ","
             << ca_gf->Max() << "," << Ta.Max() << ","
             << ux << "," << mech.GetNumNewtonIterations() << "\n";
      }

      if (paraview)
      {
         u_gf.Distribute(u);
         Ta_gf.Distribute(Ta);
         pvdc->SetCycle(step);
         pvdc->SetTime(t);
         pvdc->Save();
      }
   }

   return result;
}


int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *which = "all";
   const char *form_name = "both";
   const char *device_config = "cpu";
   int order = 1;
   int nx = 1;
   real_t t_final = 350.0;
   real_t dt = 0.5;
   int mech_stride = 1;
   bool paraview = false;
   int verbose = 0;
   const char *csv_file = "";
   int ode_substeps = 25;
   const char *outfolder = "./Output/electromechanics";

   OptionsParser args(argc, argv);
   args.AddOption(&which, "-t", "--test", "Test to run: a, b, c or all.");
   args.AddOption(&form_name, "-form", "--formulation",
                  "Active formulation for test C: stress, strain or both.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&nx, "-n", "--num-elements",
                  "Elements per direction of the unit cube.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time [ms].");
   args.AddOption(&dt, "-dt", "--time-step", "Outer time step [ms].");
   args.AddOption(&mech_stride, "-ms", "--mechanics-stride",
                  "Solve the mechanics every N steps (1 = every step).");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable ParaView output.");
   args.AddOption(&ode_substeps, "-sub", "--ode-substeps",
                  "Inner ODE substeps per outer step. The TP06 upstroke is stiff; "
                  "too few overshoots the peak potential.");
   args.AddOption(&csv_file, "-csv", "--csv-output",
                  "Write the twitch transient to this CSV file "
                  "(one file per formulation). Empty disables.");
   args.AddOption(&verbose, "-v", "--verbose",
                  "Verbosity: 0 silent, 1 Newton, 2 Newton + CG.");
   args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   const std::string sel(which);
   const bool run_a = (sel == "all" || sel == "a");
   const bool run_b = (sel == "all" || sel == "b");
   const bool run_c = (sel == "all" || sel == "c");

   // Unit cube, fixed on x = 0.
   Mesh serial_mesh =
      Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   serial_mesh.EnsureNodes();
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   mesh.EnsureNodes();

   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec);
   ConstantCoefficient chi_coeff(140.0), Cm_coeff(0.01);

   if (run_a) { TestA_PureEP(fes, chi_coeff, Cm_coeff); }
   if (run_b) { TestB_IncompatiblePairing(fes, chi_coeff, Cm_coeff); }

   if (run_c)
   {
      const std::string fsel(form_name);
      std::vector<ActiveFormulation> forms;
      if (fsel == "stress" || fsel == "both")
      {
         forms.push_back(ActiveFormulation::ActiveStress);
      }
      if (fsel == "strain" || fsel == "both")
      {
         forms.push_back(ActiveFormulation::ActiveStrain);
      }

      for (auto f : forms)
      {
         const char *label = (f == ActiveFormulation::ActiveStress)
                             ? "active stress" : "active strain";
         if (Mpi::Root())
         {
            mfem::out << "\nTest C -- single element benchmark (" << label << ")\n";
         }

         BenchmarkResult r = TestC_SingleElement(mesh, order, f, t_final, dt,
                                                 mech_stride, paraview, outfolder, verbose,
                                                 csv_file, ode_substeps);

         if (Mpi::Root())
         {
            mfem::out << std::scientific << std::setprecision(4)
                      << "    Vm range        = [" << r.Vm_min << ", "
                      << r.Vm_max << "] mV  (clamp ["
                      << r.clamp_lo << ", " << r.clamp_hi << "])\n"
                      << "    peak [Ca2+]i    = " << r.peak_calcium << " mM\n"
                      << "    peak Ta         = " << r.peak_Ta << " kPa\n"
                      << "    free-face u_f0  = " << r.min_free_face_ux << " mm\n";
         }

         // A physiological model must be free to reach its own resting and peak
         // potential. ReactionSolver clamps to [Vmin, Vmax]; if those are left
         // at the dimensionless defaults the action potential is silently
         // flattened against the rails -- which still depolarises enough to open
         // ICaL, so every other check below stays green. Testing that the
         // potential never TOUCHES a rail is the invariant that catches it, and
         // unlike a fixed threshold it stays valid if the clamp is retuned.
         Check(r.Vm_min > r.clamp_lo && r.Vm_max < r.clamp_hi,
               std::string(label) + ": action potential is unclamped");
         Check(r.Vm_min < -80.0 && r.Vm_max > 20.0,
               std::string(label) + ": action potential spans a physiological range");
         Check(r.peak_calcium > 1e-4,
               std::string(label) + ": calcium transient develops");
         Check(r.peak_Ta > 1.0,
               std::string(label) + ": active tension is generated");
         Check(r.min_free_face_ux < 0.0,
               std::string(label) + ": cube shortens along f0");
         Check(r.newton_converged,
               std::string(label) + ": Newton converged at every step");
      }
   }

   if (Mpi::Root())
   {
      mfem::out << "\n" << (failures == 0 ? "All checks passed." :
                            std::to_string(failures) + " check(s) FAILED.") << endl;
   }
   return failures == 0 ? 0 : 1;
}
