#pragma once

#include <mfem.hpp>
#include "../bc/ep_bchandler.hpp"

namespace mfem
{

namespace electrophysiology
{
    // Abstract base class for implicit solvers
    class ImplicitSolverBase : public Solver
    {
    protected:
        std::unique_ptr<IterativeSolver> linear_solver;
        std::unique_ptr<Solver> prec;
        Array<int> ess_tdof_list;
        real_t cached_dt;

        virtual void BuildOperator() = 0;

    public:
        ImplicitSolverBase(Array<int> &ess_tdof_list_)
            : ess_tdof_list(ess_tdof_list_), cached_dt(0.0) {}

        virtual void SetOperator(const Operator &op);
        virtual void EliminateBC(const Vector &x, Vector &b) const = 0;
        virtual void Mult(const Vector &x, Vector &y) const = 0;

        virtual ~ImplicitSolverBase() = default;
    };

    // Solver for implicit time integration T du/dt = -K(T) + f
    // where Top = M + dt*K + dt RobinMass 
    class ImplicitSolverFA : public ImplicitSolverBase
    {
    private:
        HypreParMatrix *M, *K, *RobinMass;
        HypreParMatrix *T, *Te;

        // Assembles the operator T = M + dt*K + dt*RobinMass
        void BuildOperator() override;

    public:
        // Constructor: assemble the operator T and setup linear solver
        ImplicitSolverFA(Array<int> &ess_tdof_list_, int dim, real_t dt_,
                         HypreParMatrix *M_, HypreParMatrix *K_, HypreParMatrix *RobinMass_ = nullptr);

        void EliminateBC(const Vector &x, Vector &b) const override;

        void Mult(const Vector &x, Vector &y) const override;

        ~ImplicitSolverFA();
    };

    // Solver for implicit time integration Top du/dt = -K(T) + f
    // where Top = M + dt*K + dt RobinMass = M + dt*(D + A - R) + dt RobinMass
    class ImplicitSolverPA : public ImplicitSolverBase
    {
    private:
        MPI_Comm comm;
        OperatorHandle opT;
        ParFiniteElementSpace *fes; //< NOT OWNED
        std::unique_ptr<ParBilinearForm> T_form;
        std::unique_ptr<ParLORDiscretization> lor;
        std::unique_ptr<ScalarMatrixProductCoefficient> dt_diff_coeff;
        Coefficient *mass_coeff;       //< NOT OWNED
        MatrixCoefficient *diff_coeff; //< NOT OWNED
        BCHandler *bcs; //< NOT OWNED
        Array<ProductCoefficient *> robin_coeffs; // Required so that ProductCoefficients don't do out of scope and Coefficient::Project fails
        Array<Array<int> *> robin_markers;
        int prec_type;

        // Assembles PA operator opT, recreate linear solver and preconditioner
        void BuildOperator() override;

    public:
        // Constructor: assemble the (partially assembled) operator T and setup linear solver 
        ImplicitSolverPA(ParFiniteElementSpace *fes_, real_t dt_,
                         BCHandler *bcs_, Array<int> &ess_tdof_list_,
                         MatrixCoefficient *diff_coeff_, Coefficient *mass_coeff_,
                         int prec_type = 1);

        void EliminateBC(const Vector &x, Vector &b) const override;

        void Mult(const Vector &x, Vector &y) const override;

        ~ImplicitSolverPA();
    };

} // namespace electrophysiology

} // namespace mfem

