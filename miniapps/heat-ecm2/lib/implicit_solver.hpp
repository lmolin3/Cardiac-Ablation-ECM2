#ifndef IMPLICIT_SOLVER_HPP
#define IMPLICIT_SOLVER_HPP

#include <mfem.hpp>
#include "bc/heat_bchandler.hpp"

    // TODO:
    // - Add functions to handle re-assembly in case of changes in the mesh and/or coefficients

namespace mfem
{

namespace heat
{
    // Abstract base class for implicit solvers
    class ImplicitSolverBase : public Solver
    {
    protected:
        IterativeSolver *linear_solver;
        Solver *prec;
        Array<int> ess_tdof_list;
        real_t current_dt;

        virtual void BuildOperator() = 0;

    public:
        ImplicitSolverBase(Array<int> &ess_tdof_list_)
            : ess_tdof_list(ess_tdof_list_), current_dt(0.0) {}

        virtual void SetOperator(const Operator &op);
        virtual void SetTimeStep(real_t dt_, bool rebuild = false) = 0;
        virtual void EliminateBC(const Vector &x, Vector &b) const = 0;
        virtual void Mult(const Vector &x, Vector &y) const = 0;
        virtual void Rebuild() = 0;
        virtual void Reset() = 0;

        virtual ~ImplicitSolverBase();
    };

    // Solver for implicit time integration Top du/dt = -K(T) + f
    // where Top = M + dt*K + dt RobinMass = M + dt*(D + A - R) + dt RobinMass
    class ImplicitSolverFA : public ImplicitSolverBase
    {
    private:
        HypreParMatrix *M, *K, *RobinMass;
        HypreParMatrix *T, *Te;

        // Assembles the operator T = M + dt*K + dt*RobinMass
        void BuildOperator() override;

    public:
        // Constructor: assemble the operator T and setup linear solver
        ImplicitSolverFA(Array<int> &ess_tdof_list_, int dim, bool use_advection, real_t dt_,
                         HypreParMatrix *M_, HypreParMatrix *K_, HypreParMatrix *RobinMass_ = nullptr);

        // Set operators and delete the operator T
        //
        void SetOperators(HypreParMatrix *M_, HypreParMatrix *K_, HypreParMatrix *RobinMass_, bool rebuild = false);

        // Set the time step and delete the operator T (if dt changed from the cached one)
        void SetTimeStep(real_t dt_, bool rebuild = false) override;

        void EliminateBC(const Vector &x, Vector &b) const override;

        void Mult(const Vector &x, Vector &y) const override;

        // Rebuild the operator T.
        // This should be **explicitly** called and skips re-assembling if T has not been deleted.
        // Both SetOperators and SetTimeStep (in case of dt change) trigger deleting T, but rebuild only if rebuild flag is set to true.
        void Rebuild() override;

        // Reset the solver
        void Reset() override;

        ~ImplicitSolverFA();
    };

    // Solver for implicit time integration Top du/dt = -K(T) + f
    // where Top = M + dt*K + dt RobinMass = M + dt*(D + A - R) + dt RobinMass
    class ImplicitSolverPA : public ImplicitSolverBase
    {
    private:
        MPI_Comm comm;
        OperatorHandle opT;
        ParFiniteElementSpace *fes;
        ParBilinearForm *T;
        ParLORDiscretization *lor;
        ScalarMatrixProductCoefficient *dtKappa;
        ProductCoefficient *dtBeta;
        real_t dtConv;
        Coefficient *rhoC, *Beta;
        VectorCoefficient *u;
        MatrixCoefficient *Kappa;
        real_t alpha;
        BCHandler *bcs;
        Array<ProductCoefficient *> robin_coeffs; // Required so that ProductCoefficients don't do out of scope and Coefficient::Project fails
        Array<Array<int> *> robin_markers;
        Array<ProductCoefficient *> general_robin_coeffs;
        Array<Array<int> *> general_robin_markers;
        bool has_diffusion, has_advection, has_reaction;
        int prec_type;

        // Assembles PA operator opT, recreate linear solver and preconditioner
        void BuildOperator() override;

    public:
        // Constructor: assemble the (partially assembled) operator T and setup linear solver 
        ImplicitSolverPA(ParFiniteElementSpace *fes_, real_t dt_,
                         BCHandler *bcs_, Array<int> &ess_tdof_list_,
                         MatrixCoefficient *Kappa_ = nullptr, Coefficient *rhoC_ = nullptr,
                         real_t alpha_ = 0.0, VectorCoefficient *u_ = nullptr,
                         Coefficient *beta_ = nullptr, int prec_type = 0);

        // Set the time step and delete the operator T (if dt changed from the cached one)
        void SetTimeStep(real_t dt_, bool rebuild = false) override;

        void EliminateBC(const Vector &x, Vector &b) const override;

        void Mult(const Vector &x, Vector &y) const override;

        // Rebuild the operator T.
        // This should be **explicitly** called and skips re-assembling if T has not been deleted.
        // SetTimeStep (in case of dt change) trigger deleting T, but rebuild only if rebuild flag is set to true.
        void Rebuild() override;

        // Reset the solver
        void Reset() override;

        ~ImplicitSolverPA();
    };

} // namespace heat

} // namespace mfem

#endif // IMPLICIT_SOLVER_HPP