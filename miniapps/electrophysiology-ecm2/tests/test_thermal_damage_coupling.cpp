//
// Extends test_thermal_coupling.cpp by:
//   - Prescribing an analytical, time-increasing temperature profile (ablation-like)
//   - Solving the three-state cell-death model (N/U/D) with CellDeathSolver
//   - Computing the combined damage field G = U + D
//   - Providing both temperature (T) and damage (G) fields to the EP solver
//
// The temperature rises from body temperature (310.15 K) toward a peak ablation
// temperature in a spatially localised hotspot centred at the domain origin.
//
// Solve the monodomain equation on a 5x5 cm^2 domain using first-order operator
// splitting (D-->R).  By default: 5x5 quad mesh with 2 serial refinements, linear
// elements, Mitchell-Schaeffer ionic model.
//
// Sample runs:
//
//   Sequential run (Corner stimulation, 500 ms, with thermal+damage coupling):
//     ./test_thermal_damage_coupling -hd -ht
//
//   With ParaView output:
//     ./test_thermal_damage_coupling -hd -ht -pv -of ./Output/
//
// mpirun -np 8 ./test_thermal_damage_coupling -d 2 -tf 50 -fa -o 2 -rs 4 -dt 0.1 -sf 10 -of /home/shared/Output/Electrophysiology/TestThermalDamageCoupling/
//

#include "mfem.hpp"
#include "../lib/reaction_solver.hpp"
#include "../lib/monodomain_solver.hpp"
#include "../../celldeath-ecm2/lib/celldeath_solver.hpp"
#include "../../common-ecm2/custom_coefficients.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace electrophysiology;
using namespace mfem::common_ecm2;

// -----------------------------------------------------------------------
// Forward declarations
// -----------------------------------------------------------------------
real_t stimulation_plane_wave(const Vector &x);
real_t temperature_function(const Vector &x, real_t t);

// -----------------------------------------------------------------------
// Global contexts
// -----------------------------------------------------------------------
struct s_MeshContext
{
    int dim = 2;
    bool hex = true;
    int n = 5;
    int serial_ref_levels = 2;
    int parallel_ref_levels = 0;
} Mesh_ctx;

struct ep_Context
{
    bool has_damage_dependency = true;
    bool has_thermal_dependency = true;
    real_t sigma = 2.4e-3;      // conductivity [S/cm]
    real_t sigma_min = 0.24e-3; // minimum conductivity (fully damaged) [S/cm]
    real_t chi = 1.4e3;         // surface-to-volume ratio [cm^-1]
    real_t Cm = 1e-3;           // membrane capacitance [mF/cm^2]
    IonicModelType model_type = IonicModelType::MITCHELL_SCHAEFFER_TD_DEPENDENT;
} ep_ctx;

struct stim_Context
{
    real_t t_start = 0.0;
    real_t t_duration = 2.0; // [ms]
    real_t t_period = 200.0; // [ms]
    real_t Iampl = 50;       // [mA/cm^3]
} stim_ctx;

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char *argv[])
{
    /////////////////////////////////////////////////////////////////////////////
    //------     1. Initialize MPI and HYPRE.
    /////////////////////////////////////////////////////////////////////////////
    Mpi::Init(argc, argv);
    Hypre::Init();

    /////////////////////////////////////////////////////////////////////////////
    //------     2. Parse command-line options.
    /////////////////////////////////////////////////////////////////////////////

    int order = 1;
    bool pa = false;
    bool last_step = false;
    real_t dt = 0.05;       // [ms]
    real_t t = 0.0;         // [ms]
    real_t t_final = 60000.0; // [ms] (60 s, clinical ablation duration)
    int dt_ode = 1;
    int ode_solver_type = 21; // Backward Euler
    bool paraview = true;
    const char *outfolder = "./Output/";
    int save_freq = 200;       // save every 200 steps (every 10 ms)
    bool verbose = true;

    OptionsParser args(argc, argv);
    args.AddOption(&Mesh_ctx.dim, "-d", "--dim", "Mesh dimension (2 or 3)");
    args.AddOption(&Mesh_ctx.hex, "-hex", "--hex", "-tri", "--tri",
                   "Use hex/quad elements (default) or tri/tet");
    args.AddOption(&Mesh_ctx.n, "-n", "--mesh-size",
                   "Number of elements in one direction");
    args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--serial-ref-levels",
                   "Number of uniform serial refinement levels");
    args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--parallel-ref-levels",
                   "Number of uniform parallel refinement levels");
    args.AddOption(&order, "-o", "--order",
                   "Finite element polynomial degree");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-fa", "--full-assembly",
                   "Enable or disable partial assembly");
    args.AddOption(&dt, "-dt", "--time-step", "Time step [ms]");
    args.AddOption(&t_final, "-tf", "--time-final", "Final time [ms]");
    args.AddOption(&dt_ode, "-dode", "--ode-substeps",
                   "Number of ODE substeps for the reaction solver");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                   ODESolver::Types.c_str());
    args.AddOption(&stim_ctx.Iampl, "-i", "--i-stim",
                   "Stimulation current amplitude");
    args.AddOption(&stim_ctx.t_start, "-ts", "--t-stim-start",
                   "Stimulation start time [ms]");
    args.AddOption(&stim_ctx.t_duration, "-td", "--t-stim-duration",
                   "Stimulation duration [ms]");
    args.AddOption(&stim_ctx.t_period, "-tp", "--t-stim-period",
                   "Stimulation period [ms]");
    args.AddOption(&ep_ctx.has_damage_dependency, "-hd", "--has-damage",
                   "-nhd", "--no-damage",
                   "Enable damage dependency in the EP reaction model");
    args.AddOption(&ep_ctx.has_thermal_dependency, "-ht", "--has-thermal",
                   "-nht", "--no-thermal",
                   "Enable thermal dependency in the EP reaction model");
    args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                   "Enable or disable ParaView output");
    args.AddOption(&outfolder, "-of", "--output-folder", "Output folder");
    args.AddOption(&save_freq, "-sf", "--save-freq",
                   "Save frequency (in time steps)");
    args.AddOption(&verbose, "-v", "--verbose", "-q", "--quiet",
                   "Enable or disable console output");
    args.ParseCheck();

    /////////////////////////////////////////////////////////////////////////////
    //------     3. Create serial and parallel mesh (shared_ptr for CellDeath)
    /////////////////////////////////////////////////////////////////////////////

    auto elem_type = Mesh_ctx.hex ? Element::QUADRILATERAL : Element::TRIANGLE;
    Mesh *serial_mesh_2d = new Mesh(
        Mesh::MakeCartesian2D(Mesh_ctx.n, Mesh_ctx.n, elem_type, true));
    serial_mesh_2d->EnsureNodes();
    GridFunction *nodes = serial_mesh_2d->GetNodes();
    *nodes *= 5.0;
    *nodes -= 2.5; // domain [-2.5, 2.5]^2

    Mesh *serial_mesh = nullptr;
    if (Mesh_ctx.dim == 3)
        serial_mesh = Extrude2D(serial_mesh_2d, 2, 0.5);
    else
        serial_mesh = serial_mesh_2d;

    for (int l = 0; l < Mesh_ctx.serial_ref_levels; l++)
        serial_mesh->UniformRefinement();

    // Use shared_ptr so CellDeathSolver can share ownership.
    auto pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, *serial_mesh);

    for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
        pmesh->UniformRefinement();

    delete serial_mesh;

    // Reference point for solution monitoring (domain centre)
    Vector point(Mesh_ctx.dim);
    point = 0.0;
    if (Mesh_ctx.dim == 3) point(2) = 0.25;
    DenseMatrix points(Mesh_ctx.dim, 1);
    points.SetCol(0, point);
    Array<int> elem_ids;
    Array<IntegrationPoint> ips;
    pmesh->FindPoints(points, elem_ids, ips);

    /////////////////////////////////////////////////////////////////////////////
    //------     4. Define FE space (shared by EP and CellDeath)
    /////////////////////////////////////////////////////////////////////////////

    H1_FECollection fec(order, pmesh->Dimension());
    ParFiniteElementSpace fespace(pmesh.get(), &fec);
    auto tdofs = fespace.GlobalTrueVSize();
    if (Mpi::Root())
        cout << "Number of unknowns: " << tdofs << endl;

    /////////////////////////////////////////////////////////////////////////////
    //------     5. Temperature GridFunction (analytical, time-increasing)
    /////////////////////////////////////////////////////////////////////////////

    ParGridFunction temperature_gf(&fespace);
    FunctionCoefficient temperature_coeff(temperature_function);
    temperature_coeff.SetTime(t);
    temperature_gf.ProjectDiscCoefficient(temperature_coeff, GridFunction::ARITHMETIC);

    /////////////////////////////////////////////////////////////////////////////
    //------     6. CellDeath solver  (N/U/D on same mesh and order as EP)
    /////////////////////////////////////////////////////////////////////////////

    // Cell-death Arrhenius parameters (Petras et al., cardiac-calibrated)
    // Literature values (A in s^-1) converted to ms^-1 (* 1e-3).
    // N -> U: piecewise A1/dE1, switching at 55 °C (328.15 K)
    const real_t T_switch  = 328.15;             // K (55 °C)
    const real_t A1_lo     = 8.87e70,  dE1_lo = 467.6e3;   // T <= 55 °C
    const real_t A1_hi     = 3.56e19,  dE1_hi = 144.7e3;   // T >  55 °C
    const real_t A2_const  = 5.35e8,   dE2    = 85.9e3;     // U -> N
    const real_t A3_const  = 1.6e12,   dE3    = 105.1e3;    // U -> D

    auto *celldeath_solver = new celldeath::CellDeathSolverEigen(
        order, &temperature_gf, verbose);

    // Override A1 and dE1 with Petras piecewise T-dependent functions.
    // A2, A3, dE2, dE3 stay constant (passed as real_t).
    celldeath_solver->SetArrheniusFunctions(
        /*A1 */ [=](real_t T) { return T > T_switch ? A1_hi : A1_lo; },
        /*A2 */ A2_const,
        /*A3 */ A3_const,
        /*dE1*/ [=](real_t T) { return T > T_switch ? dE1_hi : dE1_lo; },
        /*dE2*/ dE2,
        /*dE3*/ dE3);

    // Combined damage field G = U + D, living on the EP FE space.
    // Initialised to zero (no damage at t = 0).
    auto &G_gf = celldeath_solver->GetDamageVariableGf();

    /////////////////////////////////////////////////////////////////////////////
    //------     7. EP coefficients and solvers
    /////////////////////////////////////////////////////////////////////////////

    ConstantCoefficient chi_coeff(ep_ctx.chi);
    ConstantCoefficient Cm_coeff(ep_ctx.Cm);

    // Conductivity modulated by damage: sigma = sigma_min + (sigma - sigma_min)*(1 - G)
    GridFunctionDependentMatrixFunctionCoefficient sigma_coeff(
        Mesh_ctx.dim, &G_gf,
        [](real_t G, DenseMatrix &K) {
            G = std::min(1.0, std::max(0.0, G));
            real_t s = ep_ctx.sigma_min +
                       (ep_ctx.sigma - ep_ctx.sigma_min) * (1.0 - G);
            K = 0.0;
            K(0, 0) = s;
            K(1, 1) = s;
            if (K.NumRows() > 2)
                K(2, 2) = s;
        });

    auto *bc = new BCHandler(pmesh.get());

    bool solver_verbose = true;
    MonodomainDiffusionSolver *diff_solver =
        new MonodomainDiffusionSolver(&fespace, bc, &sigma_coeff,
                                      &chi_coeff, &Cm_coeff,
                                      ode_solver_type, solver_verbose);
    diff_solver->EnablePA(pa);

    TimeIntegrationScheme scheme = TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN;
    ReactionSolver *reaction_solver =
        new ReactionSolver(&fespace, &chi_coeff, &Cm_coeff,
                           ep_ctx.model_type, scheme, dt_ode);

    // Pass damage field G to the reaction model
    if (ep_ctx.has_damage_dependency)
    {
        auto f_damage = [](real_t g) { return g; }; // Identity function (damage enters linearly in the model)
        std::vector<real_t> td_delta_tau = {0.16, 0.16, 0.16, 0.16};
        reaction_solver->SetDamageParameters(&G_gf, f_damage, td_delta_tau);
    }

    // Pass temperature field to the reaction model
    if (ep_ctx.has_thermal_dependency)
    {
        real_t T_ref = 310.15; // K (37 °C)
        real_t A = 1.0;        // Moore term: eta = A*(1 + B*(T - Tref))
        real_t B = 0.07;       // ~7 %/°C peak-conductance sensitivity
        real_t Q10 = 3;        // Q10 for gating kinetics
        reaction_solver->SetThermalParameters(&temperature_gf, A, B, T_ref, Q10);
    }

    /////////////////////////////////////////////////////////////////////////////
    //------     8. Setup diffusion and reaction solvers
    /////////////////////////////////////////////////////////////////////////////

    diff_solver->Setup(dt);

    std::vector<double> initial_states, parameters;
    reaction_solver->GetDefaultStates(initial_states);
    reaction_solver->GetDefaultParameters(parameters);

    parameters[reaction_solver->GetModel()->parameter_index("IstimEnd")] =
        t_final; // Allow stimulation pulses throughout the entire simulation
    parameters[reaction_solver->GetModel()->parameter_index("IstimStart")] =
        stim_ctx.t_start;
    parameters[reaction_solver->GetModel()->parameter_index("IstimPulseDuration")] =
        stim_ctx.t_duration;
    parameters[reaction_solver->GetModel()->parameter_index("IstimPeriod")] =
        stim_ctx.t_period; // Periodic stimulation 

    // Reduce APD for faster tests (as in reference)
    if (ep_ctx.model_type == IonicModelType::MITCHELL_SCHAEFFER_TD_DEPENDENT)
    {
        parameters[reaction_solver->GetModel()->parameter_index("tau_close")] /= 4;
        parameters[reaction_solver->GetModel()->parameter_index("tau_open")] /= 4;
    }

    reaction_solver->Setup(initial_states, parameters);

    // Stimulation: plane wave in the x-direction at the left edge of the domain.
    FunctionCoefficient Istim_coeff(stimulation_plane_wave);
    reaction_solver->SetStimulation(&Istim_coeff);

    /////////////////////////////////////////////////////////////////////////////
    //------     9. ParaView output
    /////////////////////////////////////////////////////////////////////////////

    auto *u_gf = diff_solver->GetPotentialGf();
    Vector u;
    reaction_solver->GetPotential(u);
    u_gf->SetFromTrueDofs(u);

    auto *Istim_gf = reaction_solver->GetStimulationGF();

    ParaViewDataCollection pvdc("EP_ThermalDamage", pmesh.get());
    pvdc.SetPrefixPath(outfolder);
    pvdc.SetDataFormat(VTKFormat::BINARY32);
    pvdc.SetCompression(true);
    pvdc.SetCompressionLevel(9);
    if (order > 1)
    {
        pvdc.SetHighOrderOutput(true);
        pvdc.SetLevelsOfDetail(order);
    }
    pvdc.RegisterField("potential", u_gf);
    pvdc.RegisterField("Istim", Istim_gf);
    reaction_solver->RegisterFields(pvdc);
    pvdc.RegisterField("temperature", &temperature_gf);
    pvdc.RegisterField("damage_G", &G_gf);
    pvdc.RegisterField("N_alive", &celldeath_solver->GetAliveCellsGf());
    pvdc.RegisterField("U_vulnerable", &celldeath_solver->GetVulnerableCellsGf());
    pvdc.RegisterField("D_dead", &celldeath_solver->GetDeadCellsGf());

    if (paraview)
    {
        pvdc.SetCycle(0);
        pvdc.SetTime(t);
        pvdc.Save();
    }

    /////////////////////////////////////////////////////////////////////////////
    //------     10. Time-stepping loop
    /////////////////////////////////////////////////////////////////////////////

    if (Mpi::Root())
    {
        out << "----------------------------------------------------------------------\n";
        out << std::left
            << std::setw(8)  << "Step"
            << std::setw(14) << "Time [ms]"
            << std::setw(12) << "dt [ms]"
            << std::setw(14) << "Potential"
            << std::setw(14) << "max(G)"
            << "\n";
        out << "----------------------------------------------------------------------\n";
    }

    real_t potential = 0.0;
    int count = 0;

    for (int step = 0; !last_step; ++step)
    {
        if (t + dt >= t_final - dt / 2)
            last_step = true;

        real_t t_new = t + dt;

        // ----  Update temperature field to t_new  ----
        temperature_coeff.SetTime(t_new);
        temperature_gf.ProjectDiscCoefficient(temperature_coeff, GridFunction::ARITHMETIC);

        // ----  Solve cell-death ODE  ----
        celldeath_solver->Solve(t, dt);

        // ----  Update conductivity (damage-dependent sigma)  ----
        diff_solver->Update();

        // ----  EP diffusion step  ----
        diff_solver->Step(u, t, dt, true);

        // ----  EP reaction step  ----
        reaction_solver->Step(u, t, dt, true);

        // ----  Accept step  ----
        u_gf->SetFromTrueDofs(u);
        diff_solver->UpdateTimeStepHistory(u);
        t += dt;

        // ----  Monitor potential at domain centre  ----
        real_t pot_loc = (elem_ids[0] >= 0)
                             ? u_gf->GetValue(elem_ids[0], ips[0])
                             : 0.0;
        MPI_Allreduce(&pot_loc, &potential, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        // ----  Monitor max damage  ----
        real_t G_max_loc = G_gf.Max();
        real_t G_max = 0.0;
        MPI_Allreduce(&G_max_loc, &G_max, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);

        // ----  Save output  ----
        if (step % save_freq == 0 && paraview)
        {
            pvdc.SetCycle(step + 1);
            pvdc.SetTime(t);
            pvdc.Save();
        }

        if (Mpi::Root() && verbose)
        {
            out << std::left
                << std::setw(8)  << step
                << std::setw(14) << std::scientific << std::setprecision(4) << t
                << std::setw(12) << std::scientific << std::setprecision(4) << dt
                << std::setw(14) << std::scientific << std::setprecision(4) << potential
                << std::setw(14) << std::scientific << std::setprecision(4) << G_max
                << "\n";
        }

    }

    /////////////////////////////////////////////////////////////////////////////
    //------     11. Cleanup
    /////////////////////////////////////////////////////////////////////////////

    delete celldeath_solver;
    delete reaction_solver;
    delete diff_solver;

    return 0;
}

// -----------------------------------------------------------------------
// Function definitions
// -----------------------------------------------------------------------

// Corner stimulation: circular region at the bottom-left corner of the domain.
real_t stimulation_plane_wave(const Vector &x)
{
    real_t xs = -2.5;
    real_t xr = -2.3;
    // real_t sharpness = 5e2; //  transition
    //  return stim_ctx.Iampl * 0.5 * (1.0 - std::tanh(sharpness * (x(0) - xs)));
    return (x(0) >= xs && x(0) <= xr) ? stim_ctx.Iampl : 0.0;
}

// Ablation-like temperature profile:
//   - Gaussian spatial decay from peak temperature at the origin
//   - T(r) = T_body + (T_max - T_body) * exp(-r^2 / (2 * sigma_th^2))
//   - sigma_th controls the heated region width (~characteristic ablation radius)
//
real_t temperature_function(const Vector &x, real_t t)
{
    const real_t T_body    = 310.15; // K  (37 °C, body temperature)
    const real_t T_max     = 343.15; // K  (70 °C, peak ablation temperature)
    const real_t sigma_th  = 0.5;    // cm (spatial spread of the heated region)

    real_t r2 = x(0) * x(0) + x(1) * x(1);
    if (x.Size() > 2)
        r2 += x(2) * x(2);

    real_t spatial = std::exp(-r2 / (2.0 * sigma_th * sigma_th));

    return T_body + (T_max - T_body) * spatial;
}
