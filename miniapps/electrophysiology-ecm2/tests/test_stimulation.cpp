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
// Use internal time management of gotranx:
//   mpirun -np 4 ./test_stimulation -no-dgm -of ./Output/GotranxTime
//
// Disable internal time management of gotranx, and manage time from provided function:
//   mpirun -np 8 ./test_stimulation -dgm -of ./Output/MFEMTime
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
real_t stimulation_corner_td(const Vector &x, real_t t);

void conductivity_function(const Vector &x, DenseMatrix &Sigma);


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
    real_t sigma = 2; // conductivity   [mS/cm]
    real_t chi = 2e3; // surface-to-volume ratio [cm^-1]
    real_t Cm = 1e-3; // membrane capacitance [uF/cm^2]
    real_t matrix_factor = 1.0;
} ep_ctx;

struct stim_Context
{
    real_t t_start = 0.0;    // stimulation start time [ms]
    real_t t_duration = 1.0; // stimulation duration [ms]
    real_t Iampl = 1000;     // stimulation current amplitude  [mA/cm^3] = 1 [mA/mm^3]
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
    int order = 2;
    // Timestepping
    bool last_step = false;
    real_t dt = 0.05;       // Time step (ms)
    real_t t = 0.0;         // Current time (ms)
    real_t t_final = 50.0; // Final time (ms)
    int dt_ode = 1;         // number of ODE substeps for the reaction solver
    // Output
    bool paraview = true;
    const char *outfolder = "./Output/";
    int save_freq = 2; // save solution every save_freq time steps
    bool verbose = true;
    // Timing
    real_t t_setup = 0.0;
    real_t t_diffusion = 0.0;
    real_t t_reaction = 0.0;
    real_t t_total_solution = 0.0;
    real_t t_total = 0.0;
    real_t t_mesh = 0.0;
    real_t t_misc = 0.0;
    real_t t_io = 0.0;
    StopWatch chrono, chrono_total;

    bool disable_gotranx_time_management = false;
    
    OptionsParser args(argc, argv);
    // Time stepping related options
    args.AddOption(&disable_gotranx_time_management, "-dgm", "--disable-gotranx-time-management", "-no-dgm", "--enable-gotranx-time-management",
                   "Disable internal time management of gotranx ionic models.");
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
    Mesh *serial_mesh = new Mesh(Mesh::MakeCartesian2D(Mesh_ctx.n, Mesh_ctx.n, type, true));
    serial_mesh->EnsureNodes();
    GridFunction *nodes = serial_mesh->GetNodes();
    *nodes *= 5.0;
    *nodes -= 2.5;

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
    Vector point(Mesh_ctx.dim);
    point(0) = 0.0;
    point(1) = 0.0;
    if (Mesh_ctx.dim == 3)
    {
        point(2) = 0.25;
    }
    DenseMatrix points(Mesh_ctx.dim, 1);
    points.SetCol(0, point);

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
    auto tdofs = fespace.GetTrueVSize();
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

    chrono.Clear();
    chrono.Start();

    //<--- 5.1 Define the BCHandler (not populated)
    auto bc = new BCHandler(&mesh); // DiffusionSolver takes ownership of bc

    //<--- 5.2 Define the MonodomainDiffusionSolver
    int ode_solver_type = 21; // Backward Euler
    bool solver_verbose = true;
    MonodomainDiffusionSolver *diff_solver = new MonodomainDiffusionSolver(&fespace, bc, &sigma_coeff, &chi_coeff, &Cm_coeff, ode_solver_type, solver_verbose);

    //<--- 5.3 Define the ReactionSolver
    IonicModelType model_type = IonicModelType::MITCHELL_SCHAEFFER;
    TimeIntegrationScheme solver_type = TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN; // TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN;
    ReactionSolver *reaction_solver = new ReactionSolver(&fespace, &chi_coeff, &Cm_coeff, model_type, solver_type, dt_ode);


    /////////////////////////////////////////////////////////////////////////////
    //------     7. Add BCs and Setup the solvers
    /////////////////////////////////////////////////////////////////////////////

    //<--- 7.1 Add BCs to the BCHandler (none for now)

    //<--- 7.2 Setup the Diffusion and Reaction solvers

    // This setup the diffusion solver (assembles operators and setup ODESolver)           chi Cm dudt = div(sigma grad u) + bcs
    diff_solver->Setup(dt);

    // This setup the reaction solver (initializes states and parameters with defaults)    dudt = -Iion + Iapp; dwdt = f(u,w)
    // If needed, initial states and parameters can be passed as std::vector<double>
    // You can use empty vectors and call GetDefaultStates/GetDefaultParameters to get the default values
    // and modify them before passing to Setup()
    // Once can retrieve also the indices using names, but note that these might be model-dependent
    // since the code is autogenerated from GotranX
    std::vector<double> initial_states;
    std::vector<double> parameters;
    reaction_solver->GetDefaultStates(initial_states);
    reaction_solver->GetDefaultParameters(parameters);
    // Modify parameters if needed --> check inside the specific ionic model what parameters are available

    if (!disable_gotranx_time_management)
    {
        parameters[reaction_solver->GetModel()->parameter_index("IstimEnd")] = stim_ctx.t_start + stim_ctx.t_duration; //[ms]
        parameters[reaction_solver->GetModel()->parameter_index("IstimStart")] = stim_ctx.t_start;                     //[ms]
        parameters[reaction_solver->GetModel()->parameter_index("IstimPulseDuration")] = stim_ctx.t_duration;          //[ms]
    }
    else
    {
        // Disable internal time management of gotranx ionic model
        reaction_solver->GetModel()->DisableInternalTimeManagement(parameters.data());
    }

    // initialize the states
    initial_states[reaction_solver->GetModel()->state_index("h")] = 1.0; // initial h []

    reaction_solver->Setup(initial_states, parameters);

    //<--- 7.3 Define and set the stimulation current
    // switch case stim_ctx.stim_type, pick one of the defined stimulation functions

    Coefficient *Istim_coeff = nullptr;
    if (disable_gotranx_time_management)
    {
        Istim_coeff = new FunctionCoefficient(stimulation_corner_td);
    }
    else
    {
        Istim_coeff = new FunctionCoefficient(stimulation_corner);
    }

    reaction_solver->SetStimulation(Istim_coeff);

    chrono.Stop();
    t_setup = chrono.RealTime();

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

    real_t potential = 0.0; // just placeholder -->  we can check what we want to output
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

        //<--- Evaluate potential at the reference point
        if (verbose)
        {
            // Only evaluate grid functions if the point was found on this rank
            real_t potential_loc = (elem_ids[0] >= 0) ? u_gf->GetValue(elem_ids[0], ips[0]) : 0.0;

            // Reduce values across all ranks to get the result from whichever rank found the point
            MPI_Allreduce(&potential_loc, &potential, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

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
    real_t t_setup_g, t_diffusion_g, t_reaction_g, t_total_solution_g, t_misc_g, t_mesh_g, t_total_g, t_io_g;
    MPI_Allreduce(&t_io, &t_io_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_misc, &t_misc_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_mesh, &t_mesh_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_setup, &t_setup_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_diffusion, &t_diffusion_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_reaction, &t_reaction_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_total_solution, &t_total_solution_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t_total, &t_total_g, 1, MFEM_MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);

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
        out << std::setw(30) << std::left << "Setup time:"
            << std::setw(15) << std::right << format_time(t_setup_g) << std::endl;
        out << std::setw(30) << std::left << "Miscellaneous time:"
            << std::setw(15) << std::right << format_time(t_misc_g) << std::endl;
        out << std::setw(30) << std::left << "I/O time:"
            << std::setw(15) << std::right << format_time(t_io_g) << std::endl;
        out << std::setw(30) << std::left << "Solution time:"
            << std::setw(15) << std::right << format_time(t_total_solution_g) << std::endl;
        out << std::endl;
        out << std::setw(30) << std::left << "Diffusion time (per step):"
            << std::setw(15) << std::right << format_time(t_diffusion_g) << std::endl;
        out << std::setw(30) << std::left << "Reaction time (per step):"
            << std::setw(15) << std::right << format_time(t_reaction_g) << std::endl;
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


real_t stimulation_corner_td(const Vector &x, real_t t)
{
    real_t xc = -2.5;
    real_t yc = -2.5;
    real_t R = 5e-1;        // Radius of the stimulated region
    real_t sharpness = 5e2; //  transition
    real_t r2 = (x(0) - xc) * (x(0) - xc) + (x(1) - yc) * (x(1) - yc);

    const real_t tol = 1e-12;
    bool active = (t >= stim_ctx.t_start && (t - stim_ctx.t_start) <= (stim_ctx.t_duration + tol));

    return active ? stim_ctx.Iampl * 0.5 * (1.0 - std::tanh(sharpness * (r2 - R * R))) : 0.0;
}

