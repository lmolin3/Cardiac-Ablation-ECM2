
#include "mfem.hpp"
#include "../lib/reaction_solver.hpp"
#include "../lib/monodomain_solver.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace electrophysiology;

real_t Istim_function(const Vector &x);
void conductivity_function(const Vector &x, DenseMatrix &Sigma);

struct s_MeshContext // mesh
{
    int dim = 2;
    bool hex = true; // use quad/hex elements, otherwise tri/tet mesh
    int n = 5;
    int serial_ref_levels = 0;
    int parallel_ref_levels = 0;
} Mesh_ctx;

struct ep_Context
{
    real_t sigma = 1; // conductivity
    real_t chi = 1e-2;  // surface-to-volume ratio
    real_t Cm = 0.01;    // membrane capacitance
    real_t matrix_factor = 1.0;
    real_t Iampl = 0.2; // stimulation current amplitude
} ep_ctx;



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
    // Timestepping
    bool last_step = false;
    real_t dt = 0.01;        // Time step (ms) 
    real_t t = 0.0;          // Current time (ms)
    real_t t_final = 50.0;   // Final time (ms)
    // Output
    const char *outfolder = "./Output/";
    int save_freq = 1; // save solution every save_freq time steps

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
    // Time stepping related options
    args.AddOption(&dt, "-dt", "--time-step", "Time step size");
    args.AddOption(&t_final, "-tf", "--time-final", "Final time");
    // Electrophysiology related options
    args.AddOption(&ep_ctx.Iampl, "-i", "--i-stim", "Stimulation current amplitude");
    // Output
    args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");
    args.AddOption(&save_freq, "-sf", "--save-freq", "Save frequency (in time steps)");
    args.ParseCheck();

    /////////////////////////////////////////////////////////////////////////////
    //------     3. Create serial and parallel mesh
    /////////////////////////////////////////////////////////////////////////////

    auto type = Mesh_ctx.hex ? Element::QUADRILATERAL : Element::TRIANGLE;

    Mesh serial_mesh = Mesh::MakeCartesian2D(Mesh_ctx.n, Mesh_ctx.n, type, true);
    serial_mesh.EnsureNodes();
    GridFunction *nodes = serial_mesh.GetNodes();
    *nodes *= 2.0;
    *nodes -= 1.0;

    for (int l = 0; l < Mesh_ctx.serial_ref_levels; l++)
    {
        serial_mesh.UniformRefinement();
    }

    // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh once in parallel to increase the resolution.
    ParMesh mesh(MPI_COMM_WORLD, serial_mesh);

    for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    serial_mesh.Clear(); // the serial mesh is no longer needed

    /////////////////////////////////////////////////////////////////////////////
    //------     4. Define finite element space
    /////////////////////////////////////////////////////////////////////////////

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
    ConstantCoefficient Cm_coeff(ep_ctx.Cm);    // membrane capacitance

    //<--- 5.2 Define the conductivity coefficient
    MatrixFunctionCoefficient sigma_coeff(Mesh_ctx.dim, conductivity_function);

    /////////////////////////////////////////////////////////////////////////////
    //------     6. Define Diffusion and Reaction solver
    /////////////////////////////////////////////////////////////////////////////

    //<--- 5.1 Define the BCHandler (not populated)
    auto bc = new BCHandler(&mesh); // DiffusionSolver takes ownership of bc

    //<--- 5.2 Define the MonodomainDiffusionSolver
    int ode_solver_type = 21; // Backward Euler
    bool verbose = true;
    MonodomainDiffusionSolver *diff_solver = new MonodomainDiffusionSolver(&fespace, bc, &sigma_coeff, &chi_coeff, &Cm_coeff, ode_solver_type, verbose);

    //<--- 5.3 Define the ReactionSolver
    IonicModelType model_type = IonicModelType::MITCHELL_SCHAEFFER;
    TimeIntegrationScheme solver_type = TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN;
    int dt_ode = 10; // number of ODE substeps for the reaction solver
    ReactionSolver *reaction_solver = new ReactionSolver(fespace, model_type, solver_type, dt_ode);


    /////////////////////////////////////////////////////////////////////////////
    //------     7. Add BCs and Setup the solvers
    /////////////////////////////////////////////////////////////////////////////

    //<--- 7.1 Add BCs to the BCHandler (none for now)


    //<--- 7.2 Setup the Diffusion and Reaction solvers

    // This setup the diffusion solver (assembles operators and setup ODESolver)           chi Cm dudt = div(sigma grad u) + bcs
    diff_solver->Setup(dt);

    // This setup the reaction solver (initializes states and parameters with defaults)    dudt = -Iion + Iapp; dwdt = f(u,w)
    // If needed, initial states and parameters can be passed as std::vector<double>
    reaction_solver->Setup();
    auto *Istim_coeff = new FunctionCoefficient(Istim_function);
    reaction_solver->SetStimulation(Istim_coeff);


    /////////////////////////////////////////////////////////////////////////////
    //------     8. Setup output
    /////////////////////////////////////////////////////////////////////////////

    // @note: maybe we can create a function inside ReactionSolver to register the fields
    // This way for each ionic model we can define what fields to output
    auto u_gf = diff_solver->GetPotentialGf();
    auto Istim_gf = new ParGridFunction(&fespace);
    Istim_gf->ProjectCoefficient(*Istim_coeff);

    ParaViewDataCollection pvdc("EP", &mesh);
    pvdc.SetPrefixPath(outfolder);
    pvdc.SetDataFormat(VTKFormat::BINARY);
    pvdc.SetCompression(true);
    pvdc.SetCompressionLevel(9);
    if (order > 1)
    {
        pvdc.SetHighOrderOutput(true);
        pvdc.SetLevelsOfDetail(order);
    }
    pvdc.RegisterField("potential", u_gf);
    pvdc.RegisterField("Istim", Istim_gf);  

    pvdc.SetCycle(0);
    pvdc.SetTime(t);
    pvdc.Save();

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

    Vector u;
    reaction_solver->GetPotential(u);
    u_gf->SetFromTrueDofs(u);

    real_t potential = 0.0; // just placeholder -->  we can check what we want to output

    for (int step = 0; !last_step; ++step)
    {
        if (t + dt >= t_final - dt / 2)
        {
            last_step = true;
        }

        //<--- Solve Diffusion step
        diff_solver->Step(u, t, dt, true);

        //<--- Solve Reaction step
        // @note: the internal state vector is updated inside the ReactionSolver::Step(...)
        reaction_solver->Step(u, t, dt, true);

        //<--- Update the solution
        u_gf->SetFromTrueDofs(u);
        diff_solver->UpdateTimeStepHistory(u);
        t += dt;

        //<--- Save results
        if (step % save_freq == 0)
        {
            pvdc.SetCycle(step + 1);
            pvdc.SetTime(t);
            pvdc.Save();
        }

        if (Mpi::Root())
        {
            out << std::left
                << std::setw(8) << step
                << std::setw(16) << std::scientific << std::setprecision(8) << t
                << std::setw(16) << std::scientific << std::setprecision(8) << dt
                << std::setw(16) << std::scientific << std::setprecision(8) << potential
                << std::endl;
        }
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
    Sigma(0, 1) = 0;
    Sigma(1, 0) = 0;
    Sigma(0, 0) = ep_ctx.matrix_factor * ep_ctx.sigma;
    Sigma(1, 1) = ep_ctx.matrix_factor * ep_ctx.sigma;
}


// Define the stimulation current as a function
// Here we stimulate a circular region (r=0.1)on the bottom left corner of the domain
// Domain is [-1,1]x[-1,1]
real_t Istim_function(const Vector &x)
{
    real_t xc = -1.0;
    real_t yc = -1.0;
    real_t R = 1e-1;          // Radius of the stimulated region
    real_t sharpness = 5e2;   //  transition
    real_t r2 = (x(0) - xc)*(x(0) - xc) + (x(1) - yc)*(x(1) - yc);

    return ep_ctx.Iampl * 0.5 * (1.0 - std::tanh(sharpness * (r2 - R*R)));
}