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
                                   HypreParMatrix *M_, HypreParMatrix *K_, HypreParMatrix *RobinMass_)
    : ImplicitSolverBase(ess_tdof_list_), M(M_), K(K_), RobinMass(RobinMass_), T(nullptr), Te(nullptr)
{
    this->cached_dt = dt_;

    // Build the operator T = M + dt*K + dt*RobinMass
    BuildOperator();

    // Create preconditioner and linear solver for the operator T
    // prec = new HypreSmoother();
    // prec->SetType(HypreSmoother::Jacobi); // See hypre.hpp for more options --> use default l1-scaled block Gauss-Seidel/SSOR
    prec = std::make_unique<HypreBoomerAMG>();
    prec->iterative_mode = false;
    static_cast<HypreBoomerAMG *>(prec.get())->SetPrintLevel(0);

    linear_solver = std::make_unique<CGSolver>(M->GetComm());
    linear_solver->iterative_mode = false;
    linear_solver->SetRelTol(1e-8);
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
    // Create the operator T = M + dt*K + dt RobinMass = M + dt*(D + A - R) + dt RobinMass
    MFEM_VERIFY((M != nullptr) && (K != nullptr), "Operator M and K not set");

    T = new HypreParMatrix(*M);
    if (RobinMass)
    {
        auto tmp = ParAdd(K, RobinMass);
        T->Add(cached_dt, *tmp);
        delete tmp;
        tmp = nullptr;
    }
    else
    {
        T->Add(cached_dt, *K);
    }

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
                                   int prec_type_)
    : ImplicitSolverBase(ess_tdof_list_), fes(fes_), T_form(nullptr),
      lor(nullptr), dt_diff_coeff(nullptr),
      mass_coeff(mass_coeff_), diff_coeff(diff_coeff_),
      bcs(bcs_), prec_type(prec_type_)
{
    cached_dt = dt_;

    comm = fes->GetComm();

    // Create product coefficients (dependent on timestep)
    dt_diff_coeff = std::make_unique<ScalarMatrixProductCoefficient>(cached_dt, *diff_coeff);

    // Store Robin BC coefficients
    robin_coeffs.SetSize(0);
    robin_markers.SetSize(0);
    for (auto &robin_bc : bcs->GetRobinBcs())
    {
        // Add a Mass integrator on the Robin boundary with coefficient dt*H
        auto dtH = new ProductCoefficient(cached_dt, *robin_bc.h_coeff);
        robin_coeffs.Append(dtH);
        robin_markers.Append(&robin_bc.attr);
    }

    //<--- Build the operator T and solvers
    BuildOperator();
};

void ImplicitSolverPA::BuildOperator()
{
    // Reassemble the operator
    T_form = std::make_unique<ParBilinearForm>(fes);
    T_form->AddDomainIntegrator(new MassIntegrator(*mass_coeff));
    T_form->AddDomainIntegrator(new DiffusionIntegrator(*dt_diff_coeff));

    for (int i = 0; i < robin_coeffs.Size(); i++)
    {
        // Add a Mass integrator on the Robin boundary
        T_form->AddBoundaryIntegrator(new MassIntegrator(*robin_coeffs[i]), *robin_markers[i]);
    }

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
        lor = std::make_unique<ParLORDiscretization>(*T_form, ess_tdof_list);
        prec = std::make_unique<HypreBoomerAMG>(lor->GetAssembledMatrix());
        static_cast<HypreBoomerAMG *>(prec.get())->SetPrintLevel(0);
        break;
    default:
        MFEM_ABORT("Unknown preconditioner type.");
    }

    // Reset the solver
    linear_solver = std::make_unique<CGSolver>(fes->GetComm());
    linear_solver->iterative_mode = false;
    linear_solver->SetRelTol(1e-8);
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

    for (int i = 0; i < robin_coeffs.Size(); i++)
    {
        delete robin_coeffs[i];
    }
}
