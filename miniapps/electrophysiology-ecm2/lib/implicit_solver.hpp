#pragma once

#include <mfem.hpp>
#include "../bc/ep_bchandler.hpp"

namespace mfem
{

namespace electrophysiology
{
    /** @brief Scalar surrogate for a matrix-valued coefficient: trace(M)/dim.
     *
     * Needed because MFEM's batched LOR assembly only understands a *scalar*
     * diffusion coefficient. ProjectLORCoefficient() reads it via
     * DiffusionIntegrator::GetCoefficient(), which returns the scalar member Q;
     * for an integrator built from a MatrixCoefficient that member is nullptr and
     * LOR silently falls back to SetConstant(1.0) (see fem/lor/lor_batched.hpp).
     * The resulting preconditioner then models a completely different operator --
     * for cardiac conductivities dt*sigma ~ 1e-4, so the substituted 1.0 is off by
     * ~4 orders of magnitude and CG stagnates.
     *
     * The LOR operator only has to be spectrally equivalent, not exact, so an
     * isotropic surrogate is fine: it is exact when Sigma is isotropic, and a
     * reasonable approximation otherwise. */
    class MatrixTraceCoefficient : public Coefficient
    {
    private:
        MatrixCoefficient *MQ; //< NOT OWNED
        mutable DenseMatrix M;

    public:
        MatrixTraceCoefficient(MatrixCoefficient *MQ_)
            : MQ(MQ_), M(MQ_->GetHeight(), MQ_->GetWidth()) {}

        real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
        {
            MQ->SetTime(GetTime());
            MQ->Eval(M, T, ip);
            return M.Trace() / static_cast<real_t>(M.Height());
        }
    };

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
    // where Top = M + dt*K
    class ImplicitSolverFA : public ImplicitSolverBase
    {
    private:
        HypreParMatrix *M, *K;
        HypreParMatrix *T, *Te;
        int prec_type;

        // Assembles the operator T = M + dt*K
        void BuildOperator() override;

    public:
        // Constructor: assemble the operator T and setup linear solver
        ImplicitSolverFA(Array<int> &ess_tdof_list_, int dim, real_t dt_,
                         HypreParMatrix *M_, HypreParMatrix *K_,
                         int prec_type = 0);

        void EliminateBC(const Vector &x, Vector &b) const override;

        void Mult(const Vector &x, Vector &y) const override;

        ~ImplicitSolverFA();
    };

    // Solver for implicit time integration Top du/dt = -K(T) + f
    // where Top = M + dt*K
    class ImplicitSolverPA : public ImplicitSolverBase
    {
    private:
        MPI_Comm comm;
        OperatorHandle opT;
        ParFiniteElementSpace *fes; //< NOT OWNED
        std::unique_ptr<ParBilinearForm> T_form;
        std::unique_ptr<ParLORDiscretization> lor;
        std::unique_ptr<ScalarMatrixProductCoefficient> dt_diff_coeff;
        // Separate form + scalar coefficients used only to build the LOR
        // preconditioner; see MatrixTraceCoefficient above.
        std::unique_ptr<ParBilinearForm> lor_form;
        std::unique_ptr<MatrixTraceCoefficient> diff_trace_coeff;
        std::unique_ptr<ProductCoefficient> dt_diff_scalar_coeff;
        Coefficient *mass_coeff;       //< NOT OWNED
        MatrixCoefficient *diff_coeff; //< NOT OWNED
        BCHandler *bcs; //< NOT OWNED
        int prec_type;

        // Assembles PA operator opT, recreate linear solver and preconditioner
        void BuildOperator() override;

    public:
        // Constructor: assemble the (partially assembled) operator T and setup linear solver 
        ImplicitSolverPA(ParFiniteElementSpace *fes_, real_t dt_,
                         BCHandler *bcs_, Array<int> &ess_tdof_list_,
                         MatrixCoefficient *diff_coeff_, Coefficient *mass_coeff_,
                         int prec_type = 0);

        void EliminateBC(const Vector &x, Vector &b) const override;

        void Mult(const Vector &x, Vector &y) const override;

        ~ImplicitSolverPA();
    };

} // namespace electrophysiology

} // namespace mfem

