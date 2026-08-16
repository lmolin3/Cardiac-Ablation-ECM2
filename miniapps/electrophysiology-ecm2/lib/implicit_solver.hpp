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
        real_t rel_tol_base = 1e-6;
        bool warm_start = false;
        MPI_Comm comm = MPI_COMM_WORLD;

        virtual void BuildOperator() = 0;

    public:
        ImplicitSolverBase(Array<int> &ess_tdof_list_)
            : ess_tdof_list(ess_tdof_list_), cached_dt(0.0) {}

        virtual void SetOperator(const Operator &op);
        virtual void EliminateBC(const Vector &x, Vector &b) const = 0;
        virtual void Mult(const Vector &x, Vector &y) const = 0;

        /// Warm-start the CG solve from the previous step's solution.
        ///
        /// du/dt changes little between steps (the wavefront advances a fraction of a
        /// node spacing), so the previous solution is a good initial guess. A purely
        /// *relative* stopping test throws that away: MFEM's CG stops at
        /// nom < max(nom0*rel_tol^2, abs_tol^2), so a smaller initial residual just
        /// rescales the target and costs the same iterations. Warm starts therefore
        /// need an absolute tolerance.
        ///
        /// The absolute target is set to what a cold solve would have achieved this
        /// step: abs_tol = rel_tol * sqrt(<B b, b>), recomputed every step because the
        /// residual scale varies by orders of magnitude over an activation (calibrating
        /// once on the first step -- with the tissue still at rest -- is far too tight
        /// and makes the solve *slower*). Cost is one preconditioner apply plus one dot,
        /// against ~10 CG iterations saved.
        ///
        /// Measured at 216k dofs over 400 steps: 17.9 -> 9.4 ms/step, with ||u||_2
        /// matching a rel_tol=1e-8 reference to the same order as the cold default.
        void EnableWarmStart(const Vector &b)
        {
            if (!linear_solver || !prec) { return; }
            Vector Bb(b.Size()); Bb.UseDevice(true);
            prec->Mult(b, Bb);
            const real_t nom0_cold = InnerProduct(comm, Bb, b);
            if (!(nom0_cold > 0.0)) { return; }   // degenerate rhs: stay cold
            warm_start = true;
            linear_solver->iterative_mode = true;
            linear_solver->SetRelTol(rel_tol_base);          // keep as a floor
            linear_solver->SetAbsTol(rel_tol_base * std::sqrt(nom0_cold));
        }

        void DisableWarmStart()
        {
            if (!linear_solver) { return; }
            warm_start = false;
            linear_solver->iterative_mode = false;
            linear_solver->SetRelTol(rel_tol_base);
            linear_solver->SetAbsTol(0.0);
        }

        bool UsingWarmStart() const { return warm_start; }

        /// Iterations taken by the last solve (diagnostic).
        int GetNumIterations() const
        { return linear_solver ? linear_solver->GetNumIterations() : 0; }

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
        real_t rel_tol;

        // Assembles the operator T = M + dt*K
        void BuildOperator() override;

    public:
        // Constructor: assemble the operator T and setup linear solver
        ImplicitSolverFA(Array<int> &ess_tdof_list_, int dim, real_t dt_,
                         HypreParMatrix *M_, HypreParMatrix *K_,
                         int prec_type = 0, real_t rel_tol_ = 1e-6);

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
        real_t rel_tol;

        // Assembles PA operator opT, recreate linear solver and preconditioner
        void BuildOperator() override;

    public:
        // Constructor: assemble the (partially assembled) operator T and setup linear solver 
        ImplicitSolverPA(ParFiniteElementSpace *fes_, real_t dt_,
                         BCHandler *bcs_, Array<int> &ess_tdof_list_,
                         MatrixCoefficient *diff_coeff_, Coefficient *mass_coeff_,
                         int prec_type = 0, real_t rel_tol_ = 1e-6);

        void EliminateBC(const Vector &x, Vector &b) const override;

        void Mult(const Vector &x, Vector &y) const override;

        ~ImplicitSolverPA();
    };

} // namespace electrophysiology

} // namespace mfem

