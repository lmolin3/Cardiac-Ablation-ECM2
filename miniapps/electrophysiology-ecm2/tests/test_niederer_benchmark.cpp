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
//

#include "mfem.hpp"
#include "../lib/reaction_solver.hpp"
#include "../lib/monodomain_solver.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace electrophysiology;

real_t stimulation_function(const Vector &x, real_t t);
void conductivity_function(const Vector &x, DenseMatrix &Sigma);

struct s_MeshContext // mesh
{
    bool hex = true; // use quad/hex elements, otherwise tri/tet mesh
    int n = 5;
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

    // Create mesh from benchmark specifications
    real_t sx = 20.0; int nx = 100;
    real_t sy = 7.0;  int ny = 35;
    real_t sz = 3.0;  int nz = 15;
    auto type = Mesh_ctx.hex ? Element::HEXAHEDRON : Element::TETRAHEDRON;
    Mesh *serial_mesh = new Mesh(Mesh::MakeCartesian3D(nx, ny, nz, type, sx, sy, sz, true));

    // Refine in serial
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
    MatrixFunctionCoefficient sigma_coeff(3, conductivity_function);

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

    real_t Vimn = -85.23; // minimum potential [mV] from Niederer benchmark specifications
    real_t Vimx = 15.0;   // maximum potential [mV] assuming a 100 mV action potential
    reaction_solver->SetVRange(Vimn, Vimx); 

    // Unphysical modifications to reduce the APD for faster tests
    int state_idx = -1;
    if (ep_ctx.model_type == IonicModelType::MITCHELL_SCHAEFFER || ep_ctx.model_type == IonicModelType::MITCHELL_SCHAEFFER_TD_DEPENDENT)
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

    // Modify initial states if needed
    // initial_states[reaction_solver->GetModel()->state_index("h")] = 1.0; // initial h []
    reaction_solver->Setup(initial_states, parameters);

    //<--- 7.3 Define and set the stimulation current
    // switch case stim_ctx.stim_type, pick one of the defined stimulation functions

    Coefficient *Istim_coeff = new FunctionCoefficient(stimulation_function);
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
    Istim_gf->ProjectCoefficient(*Istim_coeff);

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

    ParGridFunction *state_gf = reaction_solver->GetStateGridFunction(state_idx);

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

void conductivity_function(const Vector &x, DenseMatrix &Sigma)
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
real_t stimulation_function(const Vector &x, real_t t)
{
    real_t x0 = 0.0, y0 = 0.0, z0 = 0.0;
    real_t L = 1.5; // mm, cube side length

    // Check if point is inside the cube
    bool inside = (x(0) >= x0 && x(0) <= x0 + L) &&
                  (x(1) >= y0 && x(1) <= y0 + L) &&
                  (x.Size() < 3 || (x(2) >= z0 && x(2) <= z0 + L));

    bool active = (stim_ctx.t_start <= t) && (t <= stim_ctx.t_start + stim_ctx.t_duration);

    return (inside && active) ? stim_ctx.Iampl : 0.0;
}