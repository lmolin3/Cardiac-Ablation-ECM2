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
//   ./test_niederer_benchmark -tf 100 -of ./Output/Niederer
//
// GPU:
//   ./test_niederer_benchmark -tf 100 -dev cuda -of ./Output/Niederer
//
//
// Backend: -dev selects the MFEM device ("cpu" default, "cuda" for GPU). Assembly is
// partial (-pa) by default -- matrix-free, so it scales to meshes where the assembled
// matrix does not fit; -fa is faster per step on meshes that do.
//
// Solver: -rtol <t> CG relative tolerance (default 1e-6; 1e-4 is defensible inside the
// operator splitting and ~1.7x faster). -ws/-no-ws warm-starts the implicit solve from
// the previous step (default on).
//
// Output: -cl <0-9> zlib level (default 1), -lod <n> ParaView levels of detail.
//

#include "mfem.hpp"
#include "../lib/reaction_solver.hpp"
#include "../lib/monodomain_solver.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace electrophysiology;

real_t stimulation_mask(const Vector &x);
real_t stimulation_amplitude(real_t t);
void conductivity_function(const Vector &x, DenseSymmetricMatrix &Sigma);

struct s_MeshContext // mesh
{
    bool hex = true; // use quad/hex elements, otherwise tri/tet mesh
    Vector dx; 
    int serial_ref_levels = 0;   // rs=0  ~60k dofs,  rs=1  ~450k dofs for linear elements
    int parallel_ref_levels = 0;
} Mesh_ctx;

struct ep_Context
{
    // Electrophysiology parameters
    // From Niederer et al. Benchmark
    real_t sigma = 2.4e-4; // conductivity   [S/mm] = 2.4e-1 [S/m]
    real_t chi = 1.4e2;    // surface-to-volume ratio [mm^-1]
    real_t Cm = 1e-5;      // membrane capacitance [mF/mm^2] = 1 [uF/cm^2]
    real_t matrix_factor = 1.0;
    IonicModelType model_type = IonicModelType::MITCHELL_SCHAEFFER;  // IonicModelType::TEN_TUSCHER_PANFILOV; // FENTON_KARMA
} ep_ctx;

struct stim_Context
{
    real_t t_start = 0.0;    // stimulation start time [ms]
    real_t t_duration = 2.0; // stimulation duration [ms]
    real_t Iampl = 50;       // stimulation current amplitude  [mA/cm^3] = 5e4 [uA/cm^3]
} stim_ctx;


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
    bool pa = true;  // partial assembly: matrix-free, the only path that scales to
                     // large meshes (the assembled matrix is ~106M nnz at 1.7M dofs)
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
    const char *device_config = "cpu"; // MFEM device backend ("cpu", "cuda", ...)
    int prec_type = 0;                 // 0: Jacobi, 1: LOR+AMG (PA implicit solver only)
    real_t lin_rtol = 1e-6;            // CG relative tolerance for the implicit diffusion solve
    bool warm_start = true;            // warm-start the implicit CG solve

    args.AddOption(&Mesh_ctx.dx, "-dx", "--mesh-size", "Mesh spacing in x, y, z directions. Default: [0.2, 0.2, 0.2]");
    args.AddOption(&Mesh_ctx.hex, "-hex", "--hex", "-tri", "--tri",
                   "Use hex/quad elements (default) or tri/tet elements");
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
    // Output
    args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                   "Enable or disable Paraview output (default enabled)");
    args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");
    args.AddOption(&save_freq, "-sf", "--save-freq", "Save frequency (in time steps)");
    args.AddOption(&verbose, "-v", "--verbose", "-q", "--quiet",
                   "Enable or disable console output (default enabled)");
    args.AddOption(&device_config, "-dev", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&prec_type, "-pt", "--prec-type",
                   "Preconditioner for the PA implicit solver: 0-Jacobi, 1-LOR+AMG.");
    args.AddOption(&lin_rtol, "-rtol", "--linear-rel-tol",
                   "Relative tolerance of the CG solve in the implicit diffusion step.");
    args.AddOption(&warm_start, "-ws", "--warm-start", "-no-ws", "--no-warm-start",
                   "Warm-start the implicit CG solve from the previous step's du/dt.");
    args.ParseCheck();

    //<--- Configure the MFEM device backend. Must happen before any Vector/mesh
    // allocation so that memory is placed in the right space.
    Device device(device_config);
    if (Mpi::Root()) { device.Print(); }

    /////////////////////////////////////////////////////////////////////////////
    //------     3. Create serial and parallel mesh
    /////////////////////////////////////////////////////////////////////////////

    chrono_total.Start();

    chrono.Clear();
    chrono.Start();

    // Create mesh from benchmark specifications
    real_t sx = 20.0; int nx = 100;
    real_t sy = 7.0;  int ny = 35;
    real_t sz = 3.0;  int nz = 15;

    // If dx is provided, override nx, ny, nz
    // Check if dx has been provided, and id dx>0
    if (Mesh_ctx.dx.Size() == 3 && Mesh_ctx.dx[0] > 0 && Mesh_ctx.dx[1] > 0 && Mesh_ctx.dx[2] > 0)
    {
        nx = static_cast<int>(sx / Mesh_ctx.dx[0]);
        ny = static_cast<int>(sy / Mesh_ctx.dx[1]);
        nz = static_cast<int>(sz / Mesh_ctx.dx[2]);

        if (Mpi::Root())
        {
            out << "\nMesh spacing provided:" << endl;
            out << "dx = [" << Mesh_ctx.dx[0] << ", " << Mesh_ctx.dx[1] << ", " << Mesh_ctx.dx[2] << "]" << endl;
            out << "n = [" << nx << ", " << ny << ", " << nz << "]" << endl;
        }
    }
    else
    {
        if (Mpi::Root())
        {
            real_t dx0 = sx / nx;
            real_t dy0 = sy / ny;
            real_t dz0 = sz / nz;
            out << "\nUsing default mesh size:" << endl;
            out << "dx = [" << dx0 << ", " << dy0 << ", " << dz0 << "]" << endl;
            out << "n = [" << nx << ", " << ny << ", " << nz << "]" << endl;
        }
    }
    
    auto type = Mesh_ctx.hex ? Element::HEXAHEDRON : Element::TETRAHEDRON;
    Mesh *serial_mesh = new Mesh(Mesh::MakeCartesian3D(nx, ny, nz, type, sx, sy, sz, true));

    // Refine in serial
    for (int l = 0; l < Mesh_ctx.serial_ref_levels; l++)
    {
        serial_mesh->UniformRefinement();
    }

    if(Mpi::Root())
    {
        out << "Number of elements: " << serial_mesh->GetNE() << "\n" << endl;
    }

    // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh once in parallel to increase the resolution.
    ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);

    for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    delete serial_mesh; // the serial mesh is no longer needed


    // 5. Define reference points P1-P8 for evaluation of the solution
    Vector P1(3), P2(3), P3(3), P4(3), P5(3), P6(3), P7(3), P8(3);  


    /////////////////////////////////////////////////////////////////////////////
    //------     4. Define finite element space
    /////////////////////////////////////////////////////////////////////////////

    chrono.Clear();
    chrono.Start();

    // 4.1 Define H1 continuous high-order Lagrange finite elements of the given order.
    H1_FECollection fec(order, 3);
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
    SymmetricMatrixFunctionCoefficient sigma_coeff(3, conductivity_function);

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
    diff_solver->Setup(dt, prec_type, lin_rtol, warm_start);
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

    real_t Vimn = -85.23; // minimum potential [mV] from Niederer benchmark specifications
    real_t Vimx = 15.0;   // maximum potential [mV] assuming a 100 mV action potential
    reaction_solver->SetVRange(Vimn, Vimx); 

    // Unphysical modifications to reduce the APD for faster tests
    int state_idx = -1;
    if (ep_ctx.model_type == IonicModelType::MITCHELL_SCHAEFFER || ep_ctx.model_type == IonicModelType::MITCHELL_SCHAEFFER_TD_DEPENDENT)
    {
        state_idx = reaction_solver->GetModel()->state_index("h");
        //parameters[reaction_solver->GetModel()->parameter_index("tau_close")] /= 4; //[ms]
        //parameters[reaction_solver->GetModel()->parameter_index("tau_open")] /= 4;  //[ms]
    }
    else if (ep_ctx.model_type == IonicModelType::FENTON_KARMA)
    {
        state_idx = reaction_solver->GetModel()->state_index("v");
        //parameters[reaction_solver->GetModel()->parameter_index("tau_si")] *= 2.0;
        //parameters[reaction_solver->GetModel()->parameter_index("tau_w_plus")] /= 2.0;
        //parameters[reaction_solver->GetModel()->parameter_index("tau_v_plus")] *= 2.0;
    }

    // Modify initial states if needed
    // initial_states[reaction_solver->GetModel()->state_index("h")] = 1.0; // initial h []
    reaction_solver->Setup(initial_states, parameters);

    //<--- 7.3 Define and set the stimulation current
    // switch case stim_ctx.stim_type, pick one of the defined stimulation functions

    // The S1 stimulus is a fixed spatial region switched on for a fixed window, i.e.
    // separable: the mask is projected once and each step only rescales it on device.
    Coefficient *Istim_coeff = new FunctionCoefficient(stimulation_mask);
    reaction_solver->SetSeparableStimulation(Istim_coeff, stimulation_amplitude);

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
    Istim_gf->ProjectCoefficient(*Istim_coeff);

    ParaViewDataCollection pvdc("EP", &mesh);
    pvdc.SetPrefixPath(outfolder);
    pvdc.SetDataFormat(VTKFormat::BINARY32);
    pvdc.SetCompression(true);
    // zlib level 1: same output size as level 9 to within ~2%, ~2x faster to
    // write. See electrophysiology-ecm2/PERFORMANCE.md, "Output (ParaView) cost".
    pvdc.SetCompressionLevel(1);
    if (order > 1)
    {
        pvdc.SetHighOrderOutput(true);
        pvdc.SetLevelsOfDetail(order);
    }
    pvdc.RegisterField("potential", u_gf);
    pvdc.RegisterField("Istim", Istim_gf);
    reaction_solver->RegisterFields(pvdc); // register ionic model states

    ParGridFunction *state_gf = reaction_solver->GetStateGridFunction(state_idx);

    if (paraview)
    {
        // Save initial condition
        pvdc.SetCycle(0);
        pvdc.SetTime(t);
        reaction_solver->SyncStateGridFunctions();
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

        //<--- Save results
        chrono.Clear();
        chrono.Start();
        if (step % save_freq == 0 && paraview)
        {
            pvdc.SetCycle(step + 1);
            pvdc.SetTime(t);
            reaction_solver->SyncStateGridFunctions();
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
    //<--- Final whole-field checksums, for comparing runs across backends/settings.
    {
        real_t l2 = 0.0, l1 = 0.0, vmin = 0.0, vmax = 0.0;
        {
            Vector uh(u); uh.HostRead();
            const real_t *h = uh.HostRead();
            real_t s2 = 0.0, s1 = 0.0, mn = h[0], mx = h[0];
            for (int i = 0; i < uh.Size(); i++)
            { s2 += h[i]*h[i]; s1 += std::abs(h[i]); mn = std::min(mn,h[i]); mx = std::max(mx,h[i]); }
            MPI_Allreduce(&s2, &l2, 1, MFEM_MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&s1, &l1, 1, MFEM_MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&mn, &vmin, 1, MFEM_MPI_REAL_T, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&mx, &vmax, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
        }
        if (Mpi::Root())
        {
            out << "\nFinal solution checksums (all dofs):\n"
                << "  ||u||_2  = " << std::scientific << std::setprecision(14) << std::sqrt(l2) << "\n"
                << "  ||u||_1  = " << std::scientific << std::setprecision(14) << l1 << "\n"
                << "  min(u)   = " << std::scientific << std::setprecision(14) << vmin << "\n"
                << "  max(u)   = " << std::scientific << std::setprecision(14) << vmax << "\n" << std::endl;
        }
    }

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

    return 0; 
}

void conductivity_function(const Vector &x, DenseSymmetricMatrix &Sigma)
{
    Sigma = 0.0;
    // Monodomain conductivities [S/mm]
    double sigma_long = 0.000132;   // fiber/longitudinal (x-direction)
    double sigma_trans = 0.000017;  // transverse (y, z directions)

    Sigma(0, 0) = sigma_long;    // x-direction (fiber)
    Sigma(1, 1) = sigma_trans;   // y-direction
    if (x.Size() > 2)
    {
        Sigma(2, 2) = sigma_trans; // z-direction
    }
}

// Define the stimulation current as a function
// Stimulate a 1.5 x 1.5 x 1.5 mm cube at the origin (corner)
// Spatial mask only: the 1.5 mm corner cube of the benchmark. Kept separate from the
// time gate so the solver can project it once -- see SetSeparableStimulation().
real_t stimulation_mask(const Vector &x)
{
    real_t x0 = 0.0, y0 = 0.0, z0 = 0.0;
    real_t L = 1.5; // mm, cube side length

    bool inside = (x(0) >= x0 && x(0) <= x0 + L) &&
                  (x(1) >= y0 && x(1) <= y0 + L) &&
                  (x.Size() < 3 || (x(2) >= z0 && x(2) <= z0 + L));

    return inside ? stim_ctx.Iampl : 0.0;
}

// Scalar time gate: the S1 pulse.
real_t stimulation_amplitude(real_t t)
{
    const bool active = (stim_ctx.t_start <= t) &&
                        (t <= stim_ctx.t_start + stim_ctx.t_duration);
    return active ? 1.0 : 0.0;
}