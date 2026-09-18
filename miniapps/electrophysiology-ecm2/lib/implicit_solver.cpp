#include "implicit_solver.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::electrophysiology;

void ImplicitSolverBase::SetOperator(const Operator &op)
{
    linear_solver->SetOperator(op);
}

// Class for solver used in implicit time integration
ImplicitSolverFA::ImplicitSolverFA(Array<int> &ess_tdof_list_, int dim, real_t dt_,
                                   HypreParMatrix *M_, HypreParMatrix *K_,
                                   int prec_type_, real_t rel_tol_)
    : ImplicitSolverBase(ess_tdof_list_), M(M_), K(K_), T(nullptr), Te(nullptr),
      prec_type(prec_type_), rel_tol(rel_tol_)
{
    this->cached_dt = dt_;

    // Build the operator T = M + dt*K
    BuildOperator();

    // Create preconditioner and linear solver for the operator T.
    //
    // Jacobi is the default. T = chi*Cm*M + dt*sigma*K is strongly mass dominated for
    // cardiac parameters (chi*Cm ~ 1.4 against dt*sigma ~ 1e-4), so it is close to a
    // well-conditioned mass matrix: Jacobi converges in fewer iterations than AMG and
    // each iteration is far cheaper. Measured on 216k dofs, diffusion per step:
    // 47 ms with Jacobi against 887 ms with BoomerAMG on GPU (GPU utilisation is 99%
    // in both cases, so this is not a host-fallback effect -- AMG is simply doing a
    // lot of work that does not help on this operator).
    //
    // AMG stays available behind prec_type == 1 for regimes where the diffusion term
    // actually dominates (much larger dt or sigma).
    if (prec_type == 1)
    {
        prec = std::make_unique<HypreBoomerAMG>();
        HypreBoomerAMG *amg_prec = static_cast<HypreBoomerAMG *>(prec.get());
        amg_prec->SetPrintLevel(0);
        amg_prec->SetRelaxType(6);            // Chebyshev smoother
        amg_prec->SetCycleNumSweeps(1, 1);    // 1 pre/post sweep
        amg_prec->SetStrengthThresh(0.5);
        amg_prec->SetAggressiveCoarsening(2);
        amg_prec->SetCoarsening(8);           // HMIS
    }
    else
    {
        // Jacobi, built explicitly rather than through hypre.
        //
        // Neither HypreSmoother(Jacobi) nor HypreDiagScale works on a device-resident
        // HypreParMatrix in this build: CG reports
        //     "The preconditioner is not positive definite. (Br, r) = -1.44e-21"
        // and bails after 1 iteration instead of ~28, returning a solution wrong by
        // ~1e-4 relative with no error raised. It reproduces with both LEGACY and
        // FULL assembly, so the fault is hypre's device diagonal handling, not the
        // assembly.
        //
        // Build the diagonal with a matvec against a vector of ones, d = T * 1, and
        // hand it to MFEM's OperatorJacobiSmoother (a plain forall kernel).
        Vector ones(T->Height()), d(T->Height());
        ones.UseDevice(true);
        d.UseDevice(true);
        ones = 1.0;
        T->Mult(ones, d);
        prec = std::make_unique<OperatorJacobiSmoother>(d, ess_tdof_list);
    }
    prec->iterative_mode = false;

    // Default 1e-6, not 1e-8: this solve sits inside a first-order operator splitting
    // whose error is O(dt), so a tighter linear tolerance buys nothing physical.
    // Measured over 400 steps at 216k dofs vs a 1e-8 reference:
    //     1e-6 -> 9.9e-11 rel, 19.1 ms/step;  1e-4 -> 1.2e-8 rel, 11.3 ms/step
    // 1e-4 is defensible for production; left opt-in via -rtol.
    comm = M->GetComm();
    linear_solver = std::make_unique<CGSolver>(comm);
    linear_solver->iterative_mode = false;
    rel_tol_base = rel_tol;
    linear_solver->SetRelTol(rel_tol);
    linear_solver->SetAbsTol(0.0);
    linear_solver->SetMaxIter(1000);
    linear_solver->SetPrintLevel(0);
    linear_solver->SetPreconditioner(*prec);
    linear_solver->SetOperator(*T);
};

void ImplicitSolverFA::Mult(const Vector &x, Vector &y) const
{
    linear_solver->Mult(x, y);
}

void ImplicitSolverFA::BuildOperator()
{
    // Create the operator T = M + dt*K
    MFEM_VERIFY((M != nullptr) && (K != nullptr), "Operator M and K not set");

    // NOTE: use the free function mfem::Add() rather than HypreParMatrix::Add().
    // The latter is implemented by mfem::internal::hypre_CSRMatrixSum(), a plain
    // host loop over the CSR i/j/data arrays (linalg/hypre_parcsr.cpp). When hypre
    // is built with HYPRE_USING_DEVICE_MEMORY those arrays live in device memory,
    // so dereferencing them on host segfaults. mfem::Add() forwards to hypre's own
    // hypre_ParCSRMatrixAdd(), which is device aware.
    T = mfem::Add(1.0, *M, cached_dt, *K);

    Te = T->EliminateRowsCols(ess_tdof_list);
}

void ImplicitSolverFA::EliminateBC(const Vector &x, Vector &b) const
{
    MFEM_VERIFY((Te != nullptr) && (T != nullptr), "Operator T and Te not set");

    T->EliminateBC(*Te, ess_tdof_list, x, b);
}

ImplicitSolverFA::~ImplicitSolverFA()
{
    delete T;
    T = nullptr;
    delete Te;
    Te = nullptr;
}

// Class for solver used in implicit time integration (PA version)
ImplicitSolverPA::ImplicitSolverPA(ParFiniteElementSpace *fes_, real_t dt_,
                                   BCHandler *bcs_, Array<int> &ess_tdof_list_,
                                   MatrixCoefficient *diff_coeff_, Coefficient *mass_coeff_,
                                   int prec_type_, real_t rel_tol_, const IntegrationRule *ir,
                                   const IntegrationRule *mass_ir)
    : ImplicitSolverBase(ess_tdof_list_), fes(fes_), T_form(nullptr),
      lor(nullptr), dt_diff_coeff(nullptr),
      mass_coeff(mass_coeff_), diff_coeff(diff_coeff_),
      bcs(bcs_), prec_type(prec_type_), rel_tol(rel_tol_), integration_rule(ir),
      mass_integration_rule(mass_ir ? mass_ir : ir)
{
    cached_dt = dt_;

    comm = fes->GetComm();

    // Create product coefficients (dependent on timestep)
    dt_diff_coeff = std::make_unique<ScalarMatrixProductCoefficient>(cached_dt, *diff_coeff);

    //<--- Build the operator T and solvers
    BuildOperator();
};

void ImplicitSolverPA::BuildOperator()
{
    // Reassemble the operator
    T_form = std::make_unique<ParBilinearForm>(fes);
    auto *mi = new MassIntegrator(*mass_coeff);
    auto *ki = new DiffusionIntegrator(*dt_diff_coeff);
    mi->SetIntRule(mass_integration_rule); ki->SetIntRule(integration_rule);
    T_form->AddDomainIntegrator(mi);
    T_form->AddDomainIntegrator(ki);

    T_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    T_form->Assemble();
    T_form->FormSystemMatrix(ess_tdof_list, opT);

    // Recreate the preconditioner
    switch (prec_type)
    {
    case 0: // Jacobi Smoother
        prec = std::make_unique<OperatorJacobiSmoother>(*T_form, ess_tdof_list);
        break;
    case 1: // LOR
    {
        // Build the LOR preconditioner from a *separate* form that uses a scalar
        // diffusion coefficient. Batched LOR reads the diffusion coefficient via
        // DiffusionIntegrator::GetCoefficient(), which returns nullptr for an
        // integrator built from a MatrixCoefficient, and then silently substitutes
        // 1.0 -- roughly 4 orders of magnitude too large for cardiac dt*sigma,
        // which makes the preconditioner useless and CG stagnate.
        diff_trace_coeff = std::make_unique<MatrixTraceCoefficient>(diff_coeff);
        dt_diff_scalar_coeff =
            std::make_unique<ProductCoefficient>(cached_dt, *diff_trace_coeff);

        lor_form = std::make_unique<ParBilinearForm>(fes);
        auto *lor_mi = new MassIntegrator(*mass_coeff);
        auto *lor_ki = new DiffusionIntegrator(*dt_diff_scalar_coeff);
        // The batched LOR assembly uses its own collocated rule on the refined
        // low-order mesh, so this rule never reaches the preconditioner matrix.
        // It is still required: lor_form is assembled on the *high-order* space,
        // and at p >= 8 the default rule makes PADiffusionSetup3D request a
        // Q1D^3 thread block the device cannot launch.
        lor_mi->SetIntRule(mass_integration_rule); lor_ki->SetIntRule(integration_rule);
        lor_form->AddDomainIntegrator(lor_mi);
        lor_form->AddDomainIntegrator(lor_ki);
        lor_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        lor_form->Assemble();

        lor = std::make_unique<ParLORDiscretization>(*lor_form, ess_tdof_list);
        prec = std::make_unique<HypreBoomerAMG>(lor->GetAssembledMatrix());

        HypreBoomerAMG *amg_prec = static_cast<HypreBoomerAMG *>(prec.get());
        amg_prec->SetPrintLevel(0);
        amg_prec->SetRelaxType(6);           // Chebyshev Smoother for fast, parallel smoothing
        amg_prec->SetCycleNumSweeps(1, 1);   // 1 pre/post sweep (start here)
        amg_prec->SetStrengthThresh(0.5);    // Increased threshold for robustness
        amg_prec->SetAggressiveCoarsening(2); // Two levels of aggressive coarsening (Major speed boost)
        amg_prec->SetCoarsening(8);          // Ensure HMIS or equivalent (Coarsening 8 is typical default)
    }
    break;
    default:
        MFEM_ABORT("Unknown preconditioner type.");
    }

    // Reset the solver
    comm = fes->GetComm();
    linear_solver = std::make_unique<CGSolver>(comm);
    linear_solver->iterative_mode = false;
    rel_tol_base = rel_tol;
    linear_solver->SetRelTol(rel_tol);
    linear_solver->SetAbsTol(0.0);
    linear_solver->SetMaxIter(500);
    linear_solver->SetPrintLevel(0);
    linear_solver->SetOperator(*opT);
    linear_solver->SetPreconditioner(*prec);
}

void ImplicitSolverPA::EliminateBC(const Vector &x, Vector &b) const
{
    auto *constrainedT = opT.As<ConstrainedOperator>();
    constrainedT->EliminateRHS(x, b);
}

void ImplicitSolverPA::Mult(const Vector &x, Vector &y) const
{
    linear_solver->Mult(x, y);
}

ImplicitSolverPA::~ImplicitSolverPA()
{
    opT.Clear();
}
