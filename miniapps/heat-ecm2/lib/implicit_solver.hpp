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
        double current_dt;

    public:
        ImplicitSolverBase(Array<int> &ess_tdof_list_)
            : ess_tdof_list(ess_tdof_list_), current_dt(0.0) {}

        virtual void SetOperator(const Operator &op);
        virtual void SetTimeStep(double dt_) = 0;
        virtual void EliminateBC(const Vector &x, Vector &b) const = 0;
        virtual void Mult(const Vector &x, Vector &y) const = 0;

        virtual ~ImplicitSolverBase();
    };

    // Solver for implicit time integration Top du/dt = -K(T) + f
    // where Top = M + dt*K + dt RobinMass = M + dt*(D + A - R) + dt RobinMass
    class ImplicitSolverFA : public ImplicitSolverBase
    {
    private:
        HypreParMatrix *M, *K, *RobinMass;
        HypreParMatrix *T, *Te;

        void BuildOperator();

    public:
        ImplicitSolverFA(HypreParMatrix *M_, HypreParMatrix *K_,
                         Array<int> &ess_tdof_list_, int dim, bool use_advection);

        void SetOperators(HypreParMatrix *M_, HypreParMatrix *K_, HypreParMatrix *RobinMass_);

        void SetTimeStep(real_t dt_) override;

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
        bool has_diffusion, has_advection, has_reaction;
        int prec_type;

    public:
        ImplicitSolverPA(ParFiniteElementSpace *fes_, real_t dt_,
                         BCHandler *bcs_, Array<int> &ess_tdof_list_,
                         MatrixCoefficient *Kappa_ = nullptr, Coefficient *rhoC_ = nullptr,
                         real_t alpha_ = 0.0, VectorCoefficient *u_ = nullptr,
                         Coefficient *beta_ = nullptr, int prec_type = 0);

        void SetTimeStep(real_t dt_) override;

        void EliminateBC(const Vector &x, Vector &b) const override;

        void Mult(const Vector &x, Vector &y) const override;

        ~ImplicitSolverPA();
    };

} // namespace heat

} // namespace mfem

#endif // IMPLICIT_SOLVER_HPP