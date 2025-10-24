//
// Solves the monodomain equation using first order operator splitting (D-->R).
// The diffusion is solved with MFEM time integration, the reaction solver
// uses explicit auto-generated code from GotranX depending on the ionic model.
//
// Solve for 500ms on a unit square, one pacing at the bottom left corner.
// By default we use a 5x5 quad mesh with 2 serial refinements, and linear elements.
// The Mitchell-Schaeffer ionic model is used.
//
// Sample runs:
//
// 2D example:
//   mpirun -np 4 ./test_monodomain -o 1 -sf 10 -of ./Output/
//
// 3D example:
//   mpirun -np 8 ./test_monodomain -d 3 -o 2 -rs 1 -sf 1 -of ./Output/
//
// Different stimulation types can be selected with -st option:
//   0 - Corner stimulation (default)
//   1 - Plane wave stimulation
//
// Test assembly type: (~230k dofs, 2x speedup with implicit time integrator, 3x speedup with explicit time integrator)
// - Full assembly (default): -fa
//       mpirun -np 4 ./test_monodomain -tf 1 -fa -o 6 -rs 5 --no-paraview -of ./Output/Electrophysiology/TestAssembly/FA
// - Partial assembly: -pa
//       mpirun -np 4 ./test_monodomain -tf 1 -pa -o 6 -rs 5 --no-paraview -of ./Output/Electrophysiology/TestAssembly/PA
//
//
// Spiral examples (2D, but can be run in 3D as well) using S1-S2 cross-field protocol:
//
// 1) Single spiral wave initiation, S2 from a line in the bottom half-domain:
//    mpirun -np 4 ./test_monodomain -st 2 -d 2 -tf 500 -mt 1 -fa -o 8 -rs 1 -dt 0.1 -sf 10 -of ./Output 
//
// 2) Multiple spiral wave initiations, S2 from rectangular area in the center:
//    mpirun -np 4 ./test_monodomain -st 3 -d 2 -tf 500 -mt 1 -fa -o 8 -rs 1 -dt 0.1 -sf 10 -of ./Output  -ems
// 
#include "mfem.hpp"
#include "../lib/reaction_solver.hpp"
#include "../lib/monodomain_solver.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace electrophysiology;

real_t stimulation_corner(const Vector &x);
real_t stimulation_plane_wave(const Vector &x);
real_t stimulation_spiral_wave(const Vector &x, real_t t);
real_t stimulation_spiral_wave_v2(const Vector &x, real_t t);
real_t stimulation_spiral_wave_v3(const Vector &x, real_t t);

inline bool CheckS1ReachedCenter(real_t potential_left, real_t potential_right, real_t recovery_left, real_t recovery_right);

void conductivity_function(const Vector &x, DenseMatrix &Sigma);

enum class StimulationType : int
{
    CORNER = 0,
    PLANE_WAVE = 1,
    SPIRAL_WAVE = 2,
    SPIRAL_WAVE_v2 = 3,
    SPIRAL_WAVE_v3 = 4
};

struct s_MeshContext // mesh
{
    int dim = 2;
    bool hex = true; // use quad/hex elements, otherwise tri/tet mesh
    int n = 5;
    int serial_ref_levels = 2;
    int parallel_ref_levels = 0;
} Mesh_ctx;

struct ep_Context
{
    // Electrophysiology parameters
    // From Niederer et al. Benchmark
    real_t sigma = 2.4e-3; // conductivity   [S/cm] = 2.4e-1 [S/m]
    real_t chi = 1.4e3;    // surface-to-volume ratio [cm^-1]
    real_t Cm = 1e-3;      // membrane capacitance [mF/cm^2] = 1 [uF/cm^2]
    real_t matrix_factor = 1.0;
    IonicModelType model_type = IonicModelType::MITCHELL_SCHAEFFER; // FENTON_KARMA
} ep_ctx;

struct stim_Context
{
    real_t t_start = 0.0;    // stimulation start time [ms]
    real_t t_duration = 2.0; // stimulation duration [ms]
    real_t Iampl = 50;       // stimulation current amplitude  [mA/cm^3] = 5e4 [uA/cm^3]
    StimulationType stim_type = StimulationType::CORNER;
} stim_ctx;


struct spiral_Context
{
    // Spiral wave specific
    real_t S1_threshold_potential = -67.0; // mV
    real_t S1_threshold_recovery = 0.5;    // dimensionless
    bool S1_depolarized_center = false;
    bool S2_started = false;
    Vector S2_center;
    real_t S2_width = 0.4;
    real_t t_start_S2 = 0.0;       // time to start S2 stimulus (determined dynamically)
    real_t Iampl_S2 = 50.0;        // S2 stimulation current amplitude
    real_t t_duration_S2 = 1.0;    // S2 stimulation duration [ms]
    bool enable_multiple_stimulations = false;
} spiral_ctx;

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

    // Finite element
    int order = 1;
    bool pa = false; // partial assembly
    // Timestepping
    bool last_step = false;
    real_t dt = 0.05;         // Time step (ms)
    real_t t = 0.0;           // Current time (ms)
    real_t t_final = 500.0;   // Final time (ms)
    int dt_ode = 1;           // number of ODE substeps for the reaction solver
    int ode_solver_type = 21; // Backward Euler. See ODESolver::Select for other options.
    // Output
    bool paraview = true;
    const char *outfolder = "./Output/";
    int save_freq = 1; // save solution every save_freq time steps
    bool verbose = true;
    // Timing
    real_t t_setup_reaction = 0.0;
    real_t t_assembly = 0.0;
    real_t t_diffusion = 0.0;
    real_t t_reaction = 0.0;
    real_t t_total_solution = 0.0;
    real_t t_total = 0.0;
    real_t t_mesh = 0.0;
    real_t t_misc = 0.0;
    real_t t_io = 0.0;
    StopWatch chrono, chrono_total;

    OptionsParser args(argc, argv);
    // Mesh related options
    args.AddOption(&Mesh_ctx.dim, "-d", "--dim", "Mesh dimension (2 or 3)");
    args.AddOption(&Mesh_ctx.hex, "-hex", "--hex", "-tri", "--tri",
                   "Use hex/quad elements (default) or tri/tet elements");
    args.AddOption(&Mesh_ctx.n, "-n", "--mesh-size", "Number of elements in one direction");
    args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--serial-ref-levels",
                   "Number of uniform refinement levels for the serial mesh");
    args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--parallel-ref-levels",
                   "Number of uniform refinement levels for the parallel mesh");
    // Finite element space related options
    args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-fa", "--full-assembly",
                   "Enable or disable partial assembly (default disabled)");
    // Time stepping related options
    args.AddOption(&dt, "-dt", "--time-step", "Time step size");
    args.AddOption(&t_final, "-tf", "--time-final", "Final time");
    args.AddOption(&dt_ode, "-dode", "--ode-substeps", "Number of ODE substeps for the reaction solver");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                   ODESolver::Types.c_str());
    // Electrophysiology related options
    args.AddOption(&stim_ctx.Iampl, "-i", "--i-stim", "Stimulation current amplitude");
    args.AddOption(&stim_ctx.t_start, "-ts", "--t-stim-start", "Stimulation start time");
    args.AddOption(&stim_ctx.t_duration, "-td", "--t-stim-duration", "Stimulation duration");
    args.AddOption((int *)&stim_ctx.stim_type, "-st", "--stim-type",
                   "Stimulation type: 0-CORNER, 1-PLANE_WAVE");
    args.AddOption((int *)&ep_ctx.model_type, "-mt", "--model-type",
                   "Ionic model type: 0-MITCHELL_SCHAEFFER, 1-FENTON_KARMA");
    args.AddOption(&spiral_ctx.enable_multiple_stimulations, "-ems", "--enable-multiple-stims",
                     "-dms", "--disable-multiple-stims",
                     "Enable or disable multiple S2 stimulations (default disabled)");
    // Output
    args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                   "Enable or disable Paraview output (default enabled)");
    args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");
    args.AddOption(&save_freq, "-sf", "--save-freq", "Save frequency (in time steps)");
    args.AddOption(&verbose, "-v", "--verbose", "-q", "--quiet",
                   "Enable or disable console output (default enabled)");
    args.ParseCheck();

    /////////////////////////////////////////////////////////////////////////////
    //------     3. Create serial and parallel mesh
    /////////////////////////////////////////////////////////////////////////////

    chrono_total.Start();

    chrono.Clear();
    chrono.Start();

    auto type = Mesh_ctx.hex ? Element::QUADRILATERAL : Element::TRIANGLE;
    Mesh *serial_mesh_2d = new Mesh(Mesh::MakeCartesian2D(Mesh_ctx.n, Mesh_ctx.n, type, true));
    serial_mesh_2d->EnsureNodes();
    GridFunction *nodes = serial_mesh_2d->GetNodes();
    *nodes *= 5.0;
    *nodes -= 2.5;

    Mesh *serial_mesh = nullptr;
    if (Mesh_ctx.dim == 3)
    {
        serial_mesh = Extrude2D(serial_mesh_2d, 2, 0.5);
    }
    else
    {
        serial_mesh = serial_mesh_2d;
    }

    for (int l = 0; l < Mesh_ctx.serial_ref_levels; l++)
    {
        serial_mesh->UniformRefinement();
    }

    // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh once in parallel to increase the resolution.
    ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);

    for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    delete serial_mesh; // the serial mesh is no longer needed


    // Find the reference point for evaluating the solution

    DenseMatrix points;
    if (stim_ctx.stim_type == StimulationType::CORNER || stim_ctx.stim_type == StimulationType::PLANE_WAVE)
    {
        Vector point(Mesh_ctx.dim);
        point(0) = 0.0;
        point(1) = 0.0;
        if (Mesh_ctx.dim == 3)
        {
            point(2) = 0.25;
        }
        points.SetSize(Mesh_ctx.dim, 1);
        points.SetCol(0, point);
    }
    else 
    {
        Vector point_left(Mesh_ctx.dim), point_right(Mesh_ctx.dim);
        point_left(0) = 0.0 - spiral_ctx.S2_width;
        point_left(1) = 0.0;
        point_right(0) = 0.0 + spiral_ctx.S2_width;
        point_right(1) = 0.0;
        if (Mesh_ctx.dim == 3)
        {
            point_left(2) = 0.25;
            point_right(2) = 0.25;
        }
        points.SetSize(Mesh_ctx.dim, 2);
        points.SetCol(0, point_left);
        points.SetCol(1, point_right);
    }

    Array<int> elem_ids;
    Array<IntegrationPoint> ips;

    mesh.FindPoints(points, elem_ids, ips);

    chrono.Stop();
    t_mesh = chrono.RealTime();

    /////////////////////////////////////////////////////////////////////////////
    //------     4. Define finite element space
    /////////////////////////////////////////////////////////////////////////////

    chrono.Clear();
    chrono.Start();

    // 4.1 Define H1 continuous high-order Lagrange finite elements of the given order.
    H1_FECollection fec(order, mesh.Dimension());
    ParFiniteElementSpace fespace(&mesh, &fec);
    auto tdofs = fespace.GlobalTrueVSize();
    if (Mpi::Root())
    {
        cout << "Number of unknowns: " << tdofs << endl;
    }

    /////////////////////////////////////////////////////////////////////////////
    //------     5. Define parameters
    /////////////////////////////////////////////////////////////////////////////

    //<--- 5.1 Define the chi and Cm coefficients
    ConstantCoefficient chi_coeff(ep_ctx.chi); // surface-to-volume ratio
    ConstantCoefficient Cm_coeff(ep_ctx.Cm);   // membrane capacitance

    //<--- 5.2 Define the conductivity coefficient
    MatrixFunctionCoefficient sigma_coeff(Mesh_ctx.dim, conductivity_function);

    chrono.Stop();
    t_misc += chrono.RealTime();

    /////////////////////////////////////////////////////////////////////////////
    //------     6. Define Diffusion and Reaction solver
    /////////////////////////////////////////////////////////////////////////////

    //<--- 5.1 Define the BCHandler (not populated)
    auto bc = new BCHandler(&mesh); // DiffusionSolver takes ownership of bc

    //<--- 5.2 Define the MonodomainDiffusionSolver
    bool solver_verbose = true;
    MonodomainDiffusionSolver *diff_solver = new MonodomainDiffusionSolver(&fespace, bc, &sigma_coeff, &chi_coeff, &Cm_coeff, ode_solver_type, solver_verbose);
    diff_solver->EnablePA(pa);

    //<--- 5.3 Define the ReactionSolver
    TimeIntegrationScheme solver_type = TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN; // TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN;
    ReactionSolver *reaction_solver = new ReactionSolver(&fespace, &chi_coeff, &Cm_coeff, ep_ctx.model_type, solver_type, dt_ode);

    /////////////////////////////////////////////////////////////////////////////
    //------     7. Add BCs and Setup the solvers
    /////////////////////////////////////////////////////////////////////////////

    //<--- 7.1 Add BCs to the BCHandler (none for now)

    //<--- 7.2 Setup the Diffusion and Reaction solvers

    // This setup the diffusion solver (assembles operators and setup ODESolver)           chi Cm dudt = div(sigma grad u) + bcs
    chrono.Clear();
    chrono.Start();
    diff_solver->Setup(dt);
    chrono.Stop();
    t_assembly = chrono.RealTime();

    // This setup the reaction solver (initializes states and parameters with defaults)    dudt = -Iion + Iapp; dwdt = f(u,w)
    // If needed, initial states and parameters can be passed as std::vector<double>
    // You can use empty vectors and call GetDefaultStates/GetDefaultParameters to get the default values
    // and modify them before passing to Setup()
    // Once can retrieve also the indices using names, but note that these might be model-dependent
    // since the code is autogenerated from GotranX
    chrono.Clear();
    chrono.Start();
    std::vector<double> initial_states;
    std::vector<double> parameters;
    reaction_solver->GetDefaultStates(initial_states);
    reaction_solver->GetDefaultParameters(parameters);
    // Modify parameters if needed --> check inside the specific ionic model what parameters are available
    parameters[reaction_solver->GetModel()->parameter_index("IstimEnd")] = stim_ctx.t_start + stim_ctx.t_duration; //[ms]
    parameters[reaction_solver->GetModel()->parameter_index("IstimStart")] = stim_ctx.t_start;                     //[ms]
    parameters[reaction_solver->GetModel()->parameter_index("IstimPulseDuration")] = stim_ctx.t_duration;          //[ms]}

    // Unphysical modifications to reduce the APD for faster tests
    int state_idx = -1;
    if (ep_ctx.model_type == IonicModelType::MITCHELL_SCHAEFFER)
    {
        state_idx = reaction_solver->GetModel()->state_index("h");
        parameters[reaction_solver->GetModel()->parameter_index("tau_close")] /= 4; //[ms]
        parameters[reaction_solver->GetModel()->parameter_index("tau_open")] /= 4;  //[ms]
    }
    else if (ep_ctx.model_type == IonicModelType::FENTON_KARMA)
    {
        state_idx = reaction_solver->GetModel()->state_index("v");
        parameters[reaction_solver->GetModel()->parameter_index("tau_si")] *= 2.0;
        parameters[reaction_solver->GetModel()->parameter_index("tau_w_plus")] /= 2.0;
        parameters[reaction_solver->GetModel()->parameter_index("tau_v_plus")] *= 2.0;
    }

    // Spiral wave specific
    if (stim_ctx.stim_type > StimulationType::PLANE_WAVE)
    {
        reaction_solver->GetModel()->DisableInternalTimeManagement(parameters.data());
    }

    // Modify initial states if needed
    // initial_states[reaction_solver->GetModel()->state_index("h")] = 1.0; // initial h []
    reaction_solver->Setup(initial_states, parameters);

    //<--- 7.3 Define and set the stimulation current
    // switch case stim_ctx.stim_type, pick one of the defined stimulation functions

    Coefficient *Istim_coeff = nullptr;
    switch (stim_ctx.stim_type)
    {
    case StimulationType::CORNER:
        Istim_coeff = new FunctionCoefficient(stimulation_corner);
        break;
    case StimulationType::PLANE_WAVE:
        Istim_coeff = new FunctionCoefficient(stimulation_plane_wave);
        break;
    case StimulationType::SPIRAL_WAVE:
        Istim_coeff = new FunctionCoefficient(stimulation_spiral_wave);
        break;
    case StimulationType::SPIRAL_WAVE_v2:
        Istim_coeff = new FunctionCoefficient(stimulation_spiral_wave_v2);
        break;
    case StimulationType::SPIRAL_WAVE_v3:
        Istim_coeff = new FunctionCoefficient(stimulation_spiral_wave_v3);
        break;
    default:
        mfem_error("Unknown stimulation type!");
    }

    reaction_solver->SetStimulation(Istim_coeff);

    chrono.Stop();
    t_setup_reaction = chrono.RealTime();

    /////////////////////////////////////////////////////////////////////////////
    //------     8. Setup output
    /////////////////////////////////////////////////////////////////////////////

    chrono.Clear();
    chrono.Start();

    // @note: maybe we can create a function inside ReactionSolver to register the fields
    // This way for each ionic model we can define what fields to output
    auto u_gf = diff_solver->GetPotentialGf();
    Vector u;
    reaction_solver->GetPotential(u);
    u_gf->SetFromTrueDofs(u);

    auto Istim_gf = reaction_solver->GetStimulationGF();

    ParaViewDataCollection pvdc("EP", &mesh);
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
    reaction_solver->RegisterFields(pvdc); // register ionic model states

    ParGridFunction *state_gf = nullptr;
    bool own_gf = reaction_solver->GetStateGridFunction(state_idx, state_gf);

    if (paraview)
    {
        // Save initial condition
        pvdc.SetCycle(0);
        pvdc.SetTime(t);
        pvdc.Save();
    }

    chrono.Stop();
    t_misc += chrono.RealTime();

    /////////////////////////////////////////////////////////////////////////////
    //------     9. Solve the problem
    /////////////////////////////////////////////////////////////////////////////

    if (Mpi::Root())
    {
        out << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        out << std::left
            << std::setw(8) << "Step"
            << std::setw(16) << "Time"
            << std::setw(16) << "dt"
            << std::setw(16) << "Potential"
            << std::endl;
        out << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
    }

    real_t potential = 0.0;
    real_t potential_right = 0.0;
    real_t recovery = 0.0;
    real_t recovery_right = 0.0;

    int count = 0;

    for (int step = 0; !last_step; ++step)
    {
        if (t + dt >= t_final - dt / 2)
        {
            last_step = true;
        }

        //<--- Solve Diffusion step
        chrono.Clear();
        chrono.Start();
        diff_solver->Step(u, t, dt, true);
        chrono.Stop();
        t_diffusion += chrono.RealTime();

        //<--- Solve Reaction step
        // @note: the internal state vector is updated inside the ReactionSolver::Step(...)
        chrono.Clear();
        chrono.Start();
        reaction_solver->Step(u, t, dt, true);
        chrono.Stop();
        t_reaction += chrono.RealTime();

        //<--- Update the solution
        u_gf->SetFromTrueDofs(u);
        diff_solver->UpdateTimeStepHistory(u);
        t += dt;

        //<--- Compute potential at the reference point (center of the domain)
        // Only evaluate grid functions if the point was found on this rank
        real_t potential_loc = (elem_ids[0] >= 0) ? u_gf->GetValue(elem_ids[0], ips[0]) : 0.0;
        real_t recovery_loc = (elem_ids[0] >= 0) ? state_gf->GetValue(elem_ids[0], ips[0]) : 0.0;

        // Reduce values across all ranks to get the result from whichever rank found the point
        MPI_Allreduce(&potential_loc, &potential, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&recovery_loc, &recovery, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        //<--- Check if back front reached center (for spiral wave only)
        if (stim_ctx.stim_type > StimulationType::PLANE_WAVE && (!spiral_ctx.S2_started || spiral_ctx.enable_multiple_stimulations))
        {
            real_t potential_loc_right = (elem_ids[1] >= 0) ? u_gf->GetValue(elem_ids[1], ips[1]) : 0.0;
            real_t recovery_loc_right = (elem_ids[1] >= 0) ? state_gf->GetValue(elem_ids[1], ips[1]) : 0.0;
            MPI_Allreduce(&potential_loc_right, &potential_right, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&recovery_loc_right, &recovery_right, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            if (CheckS1ReachedCenter(potential, potential_right, recovery, recovery_right))
            {
                spiral_ctx.t_start_S2 = t;
            }
        }

        //<--- Save results
        chrono.Clear();
        chrono.Start();
        if (step % save_freq == 0 && paraview)
        {
            pvdc.SetCycle(step + 1);
            pvdc.SetTime(t);
            pvdc.Save();
        }
        chrono.Stop();
        t_io += chrono.RealTime();

        chrono.Clear();
        chrono.Start();

        if (Mpi::Root() && verbose)
        {
            out << std::left
                << std::setw(8) << step
                << std::setw(16) << std::scientific << std::setprecision(8) << t
                << std::setw(16) << std::scientific << std::setprecision(8) << dt
                << std::setw(16) << std::scientific << std::setprecision(8) << potential
                << std::endl;
        }

        count++;
    }

    chrono_total.Stop();
    t_total = chrono_total.RealTime();

    t_total_solution = t_diffusion + t_reaction;
    t_diffusion /= count;
    t_reaction /= count;

    // Compute global times
    real_t t_setup_reaction_g, t_diffusion_g, t_reaction_g, t_total_solution_g, t_misc_g, t_mesh_g, t_total_g, t_io_g;
    MPI_Allreduce(&t_io, &t_io_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_misc, &t_misc_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_mesh, &t_mesh_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_setup_reaction, &t_setup_reaction_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_diffusion, &t_diffusion_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_reaction, &t_reaction_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_total_solution, &t_total_solution_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_total, &t_total_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);

    real_t t_solution_g = t_diffusion_g + t_reaction_g;

    if (Mpi::Root())
    {
        // Print again the dofs
        out << std::endl;
        out << "Number of unknowns: " << tdofs << std::endl;
        out << "Number of time steps: " << count << std::endl;

        // Print timing summary
        out << std::endl;
        out << "-----------------------------------------------" << std::endl;
        out << "Timing Summary" << std::endl;
        out << "-----------------------------------------------" << std::endl;

        // Helper lambda to format time with appropriate units
        auto format_time = [](real_t time_s) -> std::string
        {
            if (time_s < 0.1)
            {
                return std::to_string(time_s * 1000.0) + " ms";
            }
            else
            {
                return std::to_string(time_s) + " s ";
            }
        };

        out << std::fixed << std::setprecision(3);
        out << std::setw(30) << std::left << "Mesh time:"
            << std::setw(15) << std::right << format_time(t_mesh_g) << std::endl;
        out << std::setw(30) << std::left << "Assembly time:"
            << std::setw(15) << std::right << format_time(t_assembly) << std::endl;
        out << std::setw(30) << std::left << "Setup reaction time:"
            << std::setw(15) << std::right << format_time(t_setup_reaction_g) << std::endl;
        out << std::setw(30) << std::left << "Miscellaneous time:"
            << std::setw(15) << std::right << format_time(t_misc_g) << std::endl;
        out << std::setw(30) << std::left << "I/O time:"
            << std::setw(15) << std::right << format_time(t_io_g) << std::endl;
        out << std::setw(30) << std::left << "Solution time:"
            << std::setw(15) << std::right << format_time(t_total_solution_g) << std::endl;
        out << std::endl;
        out << std::setw(30) << std::left << "Diffusion time (per step):"
            << std::setw(15) << std::right << format_time(t_diffusion_g) << " (" << (t_diffusion_g / t_solution_g) * 100.0 << " %)" << std::endl;
        out << std::setw(30) << std::left << "Reaction time (per step):"
            << std::setw(15) << std::right << format_time(t_reaction_g) << " (" << (t_reaction_g / t_solution_g) * 100.0 << " %)" << std::endl;
        out << std::endl;
        out << std::setw(30) << std::left << "Total time:"
            << std::setw(15) << std::right << format_time(t_total_g) << std::endl;
        out << "-----------------------------------------------" << std::endl;
    }

    /////////////////////////////////////////////////////////////////////////////
    //------     9. Cleanup
    /////////////////////////////////////////////////////////////////////////////

    delete Istim_coeff;
    delete reaction_solver;
    delete diff_solver;

    if (own_gf)
    {
        delete state_gf;
    }

    return 0;
}

void conductivity_function(const Vector &x, DenseMatrix &Sigma)
{
    Sigma = 0.0;
    Sigma(0, 0) = ep_ctx.matrix_factor * ep_ctx.sigma;
    Sigma(1, 1) = ep_ctx.matrix_factor * ep_ctx.sigma;
    if (x.Size() > 2)
    {
        Sigma(2, 2) = ep_ctx.matrix_factor * ep_ctx.sigma;
    }
}

// Define the stimulation current as a function
// Here we stimulate a circular region (r=0.1)on the bottom left corner of the domain
// Domain is [-1,1]x[-1,1]
real_t stimulation_corner(const Vector &x)
{
    real_t xc = -2.5;
    real_t yc = -2.5;
    real_t zc = 0.0;
    real_t R = 5e-1;        // Radius of the stimulated region
    real_t sharpness = 5e2; //  transition
    real_t r2 = (x(0) - xc) * (x(0) - xc) + (x(1) - yc) * (x(1) - yc);
    if (x.Size() > 2)
    {
        r2 += (x(2) - zc) * (x(2) - zc); // Add z-component for 3D
    }
    return stim_ctx.Iampl * 0.5 * (1.0 - std::tanh(sharpness * (r2 - R * R)));
}

// Define the stimulation current as a function
// Here we stimulate a plane wave at the left side of the domain, traveling rightwards
// Domain is [-2.5,2.5]x[-2.5,2.5]
real_t stimulation_plane_wave(const Vector &x)
{
    real_t xs = -2.5;
    real_t xr = -2.3;
    // real_t sharpness = 5e2; //  transition
    //  return stim_ctx.Iampl * 0.5 * (1.0 - std::tanh(sharpness * (x(0) - xs)));
    return (x(0) >= xs && x(0) <= xr) ? stim_ctx.Iampl : 0.0;
}

// Define the stimulation current as a function
// Here we stimulate a spiral wave using an S1-S2 protocol
// Domain is [-2.5,2.5]x[-2.5,2
// The S1 is a plane wave at the left side of the domain, traveling rightwards
// The S2 stimulates a narrow rectangular region at the center of the domain,
// with height equal to half the domain height (2.5), and width 0.2
// The S2 is triggered after the S1 has depolarized and repolarized the center
// of the domain, given by spiral_ctx.S1_repolarized_center
real_t stimulation_spiral_wave(const Vector &x, real_t t)
{
    // S1 plane wave parameters
    real_t xs = -2.5;
    real_t xr = -2.3;

    // S2 rectangular region parameters
    real_t x_center = 0.0;
    real_t y_center = 0.0;
    real_t half_width = spiral_ctx.S2_width / 2.0;     

    // Check if we are in the S1 stimulation phase
    if (t >= stim_ctx.t_start && t <= stim_ctx.t_start + stim_ctx.t_duration)
    {
        // S1: plane wave at the left side of the domain
        if (x(0) >= xs && x(0) <= xr)
        {
            return stim_ctx.Iampl;
        }
    }
    else if (spiral_ctx.S1_depolarized_center && spiral_ctx.S2_started)
    {
        // Check if we are in the S2 stimulation phase
        if (t >= spiral_ctx.t_start_S2 &&
            t <= spiral_ctx.t_start_S2 + spiral_ctx.t_duration_S2)
        {
            // S2: rectangular region at the center of the domain
            if (x(0) >= (x_center - half_width) && x(0) <= (x_center + half_width) && x(1) <= y_center)
            {
                return spiral_ctx.Iampl_S2;
            }
        }
    }

    return 0.0;

}

// Same as above but S2 is now a stimulation on a rectangular area in the center
real_t stimulation_spiral_wave_v2(const Vector &x, real_t t)
{
    // S1 plane wave parameters
    real_t xs = -2.5;
    real_t xr = -2.3;

    // S2 rectangular region parameters
    real_t x_center = 0.0;
    real_t y_center = 0.0;
    real_t half_width = spiral_ctx.S2_width / 2.0;     

    // Check if we are in the S1 stimulation phase
    if (t >= stim_ctx.t_start && t <= stim_ctx.t_start + stim_ctx.t_duration)
    {
        // S1: plane wave at the left side of the domain
        if (x(0) >= xs && x(0) <= xr)
        {
            return stim_ctx.Iampl;
        }
    }
    else if (spiral_ctx.S1_depolarized_center && spiral_ctx.S2_started)
    {
        // Check if we are in the S2 stimulation phase
        if (t >= spiral_ctx.t_start_S2 &&
            t <= spiral_ctx.t_start_S2 + spiral_ctx.t_duration_S2)
        {
            // S2: rectangular region in the center of the domain
            // x \in [x_center-half_width, x_center+half_width], y \in [y_center- 1, y_center+1]
            if (x(0) >= (x_center - half_width) && x(0) <= (x_center + half_width) &&
                x(1) >= (y_center - 1) && x(1) <= (y_center + 1))
            {
                return spiral_ctx.Iampl_S2;
            }
        }
    }

    return 0.0;

}

// Same as above but S2 is now a stimulation on two rectangular areas.
// One is in the mid top half of the domain, the other is centered in the bottom edge
real_t stimulation_spiral_wave_v3(const Vector &x, real_t t)
{
    // S1 plane wave parameters
    real_t xs = -2.5;
    real_t xr = -2.3;

    // S2 rectangular region parameters
    real_t x_center = 0.0;
    real_t y_center_1 = 1.25;
    real_t y_center_2 = -2.25;
    real_t quarter_width = spiral_ctx.S2_width / 4.0;

    // Check if we are in the S1 stimulation phase
    if (t >= stim_ctx.t_start && t <= stim_ctx.t_start + stim_ctx.t_duration)
    {
        // S1: plane wave at the left side of the domain
        if (x(0) >= xs && x(0) <= xr)
        {
            return stim_ctx.Iampl;
        }
    }
    else if (spiral_ctx.S1_depolarized_center && spiral_ctx.S2_started)
    {
        // Check if we are in the S2 stimulation phase
        if (t >= spiral_ctx.t_start_S2 &&
            t <= spiral_ctx.t_start_S2 + spiral_ctx.t_duration_S2)
        {
            // S2: rectangular regions at the center top and bottom of the domain
            // Top: x \in [x_center-quarter_width, x_center+quarter_width], y \in [y_center_1-0.5, y_center_1+0.5]
            // Bottom: x \in [x_center-quarter_width, x_center+quarter_width], y \in [y_center_2-0.5, y_center_2+0.5]
            if ( (x(0) >= (x_center - quarter_width) && x(0) <= (x_center + quarter_width) &&
                  x(1) >= (y_center_1 - 0.5) && x(1) <= (y_center_1 + 0.5)) ||
                 (x(0) >= (x_center - quarter_width) && x(0) <= (x_center + quarter_width) &&
                  x(1) >= (y_center_2 - 0.5) && x(1) <= (y_center_2 + 0.5)) )
            {
                return spiral_ctx.Iampl_S2;
            }
        }
    }

    return 0.0;
}

// Check if the S1 wave has reached the center (depolarized)
// and subsequently repolarized, so we can trigger the S2 stimulus
// Uses parameters from spiral_ctx
inline bool CheckS1ReachedCenter(real_t potential_left, real_t potential_right, real_t recovery_left, real_t recovery_right)
{
    if (!spiral_ctx.S1_depolarized_center)
    {
        if (potential_left >= spiral_ctx.S1_threshold_potential)
        {
            spiral_ctx.S1_depolarized_center = true;
            if (Mpi::Root())
            {
                cout << "S1 depolarized center detected!" << endl;
            }
        }
    }
    else
    {
        // Check if at:
        // center-width (left): recovery > threshold (repolarized), potential < threshold (resting)
        // center+width (right): recovery < threshold (still refractory), potential > threshold (depolarized/repolarizing)
        bool recovery_flag = (recovery_left > spiral_ctx.S1_threshold_recovery && recovery_right < spiral_ctx.S1_threshold_recovery);
        bool potential_flag = (potential_left < spiral_ctx.S1_threshold_potential && potential_right > spiral_ctx.S1_threshold_potential);
        
        // Only trigger if not currently stimulating and conditions are met
        if (recovery_flag && !spiral_ctx.S2_started)
        {
            spiral_ctx.S2_started = true;
            if (Mpi::Root())
            {
                out << "S1 repolarized center detected!" << std::endl;
                out << "  Left  (x=" << -spiral_ctx.S2_width << "): V=" << potential_left << " mV, h=" << recovery_left << endl;
                out << "  Right (x=" << spiral_ctx.S2_width << "): V=" << potential_right << " mV, h=" << recovery_right << endl;
            }
        }
        // Reset S2_started flag when conditions are no longer met (allows re-triggering)
        // Only reset if multiple stimulations are enabled
        else if (!recovery_flag && spiral_ctx.S2_started && spiral_ctx.enable_multiple_stimulations)
        {
            spiral_ctx.S2_started = false;
        }
    }

    return spiral_ctx.S2_started && spiral_ctx.S1_depolarized_center;
}