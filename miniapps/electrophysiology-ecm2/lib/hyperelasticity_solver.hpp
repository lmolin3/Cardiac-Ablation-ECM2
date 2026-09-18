#pragma once

#include "mfem.hpp"
#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

#include <memory>

namespace mfem
{
    namespace electrophysiology
    {

        /**
         * @brief How the cell model's activation enters the mechanics.
         *
         * ActiveStress adds an active term to the strain-energy density, so the
         * total stress is the passive stress plus a fiber-directed active
         * stress. ActiveStrain instead splits the deformation multiplicatively,
         * F = F_e F_a, and evaluates the *passive* law on the elastic part only.
         */
        enum class ActiveFormulation : int
        {
            ActiveStress = 0,
            ActiveStrain = 1
        };

        /**
         * @brief Preconditioner for the Newton (Hessian) solves.
         *
         * Jacobi is matrix-free but its CG iteration count grows with the mesh,
         * which is what caps the tractable problem size. AMG assembles the
         * Hessian into a HypreParMatrix once per Newton iteration and builds
         * BoomerAMG with elasticity options on it: it pays an assembly cost to
         * buy a mesh-independent iteration count, and is the default because
         * that trade turns strongly favourable at tissue scale.
         */
        enum class MechanicsPreconditioner : int
        {
            None = 0,
            Jacobi = 1,
            AMG = 2
        };

        /**
         * @brief Parameters of the transversely isotropic Holzapfel-type passive law.
         *
         * Stresses are in kPa, matching the Land 2017 tension output.
         */
        struct HolzapfelParameters
        {
            real_t c = 50.0;      ///< Isotropic (matrix) stiffness [kPa]
            real_t kappa = 100.0; ///< Bulk penalty on log(J) [kPa]
            real_t k1 = 10.0;     ///< Fiber stiffness [kPa]
            real_t k2 = 20.0;     ///< Fiber exponential stiffening [-]
        };


        /**
         * @brief Quasi-static finite-strain cardiac mechanics driven by an active field.
         *
         * The material response is specified *only* as a scalar strain-energy
         * density Psi. The nonlinear residual is the first variation of
         * Pi_int(u) = int Psi dx, obtained from
         * DifferentiableOperator::GetDerivative, and Newton's Hessian-vector
         * products come from GetSecondDerivative -- so neither the stress nor
         * the tangent modulus is ever written out by hand.
         *
         * The active field arrives as a true-dof vector from the reaction
         * solver and reaches the quadrature points through the standard MFEM
         * pipeline (T-vector -> L-vector -> E-vector -> Q-vector), which is what
         * registering it as a dFEM input field with Value<> does.
         */
        class HyperelasticitySolver
        {
        public:
            /**
             * @param fes         Displacement space (vector H1, vdim = dim).
             * @param active_fes  Scalar space carrying the active field.
             * @param fiber_fes   Vector space (vdim = dim) carrying the fiber direction f0.
             * @param ir          Integration rule for the domain integrator.
             */
            HyperelasticitySolver(ParFiniteElementSpace &fes,
                                  ParFiniteElementSpace &active_fes,
                                  ParFiniteElementSpace &fiber_fes,
                                  const IntegrationRule &ir,
                                  ActiveFormulation formulation = ActiveFormulation::ActiveStress,
                                  const HolzapfelParameters &params = {});

            ~HyperelasticitySolver();

            /**
             * @brief Choose the active stress / active strain formulation.
             *
             * Rebuilds the differentiable operator, because the two use
             * different quadrature functions.
             */
            void SetFormulation(ActiveFormulation formulation);

            ActiveFormulation GetFormulation() const { return formulation_; }

            /// Set the passive material parameters (rebuilds the operator).
            void SetMaterialParameters(const HolzapfelParameters &params);

            /**
             * @brief Set the fiber direction field, as true dofs of @a fiber_fes.
             *
             * Need not be exactly unit: the q-functions renormalise at the
             * quadrature point. That matters for any rotating fiber field --
             * H1 interpolation of a rotating unit field is shorter than unit
             * between nodes, which would otherwise leave the reference
             * configuration carrying residual stress.
             */
            void SetFiberField(const Vector &f0_tvector);

            /// Convenience overload projecting a coefficient onto the fiber space.
            void SetFiberField(VectorCoefficient &f0_coeff);

            /// Homogeneous Dirichlet (fully clamped) boundary attributes.
            void SetEssentialAttributes(const Array<int> &ess_bdr);

            /**
             * @brief Update the active field driving the mechanics.
             *
             * For ActiveStress this is the active tension Ta [kPa] produced by
             * ReactionSolver::GetActiveTension(). For ActiveStrain it is the
             * dimensionless fiber shortening factor gamma in [0, 1).
             */
            void UpdateActiveField(const Vector &active_tvector);

            /**
             * @brief Solve the equilibrium problem with Newton-Raphson.
             *
             * The active field is frozen for the whole solve: it is an input
             * field of the operator and is not differentiated, so every Newton
             * iteration sees the same activation.
             */
            void SolveQuasiStaticStep();

            /// Current displacement, as a grid function (distributed after each solve).
            ParGridFunction &GetDisplacement() { return u_gf_; }

            /// Current displacement true dofs.
            Vector &GetDisplacementTrueDofs() { return u_; }

            /**
             * @brief Newton controls.
             *
             * @a abs_tol matters more than it looks. Before the tissue is
             * activated the exact solution is u = 0, so the initial residual is
             * at round-off; a purely relative test can then never be satisfied
             * and Newton burns max_iter iterations at every quiet step. Whether
             * that residual is exactly zero or merely 1e-19 depends on the
             * reduction order, i.e. on the backend -- so the absolute floor is
             * what makes CPU and GPU runs agree.
             */
            void SetNewtonOptions(int max_iter, real_t rel_tol, int print_level = 1,
                                  real_t abs_tol = 1e-12);
            void SetLinearSolverOptions(int max_iter, real_t rel_tol, int print_level = -1,
                                        real_t abs_tol = 1e-14);

            /**
             * @brief Choose the Newton preconditioner (default AMG).
             *
             * AMG requires the Hessian to be assemblable into a HypreParMatrix;
             * the solver falls back to Jacobi with a warning if it is not.
             */
            void SetPreconditioner(MechanicsPreconditioner prec) { prec_ = prec; }
            MechanicsPreconditioner GetPreconditioner() const { return prec_; }

            /// Total CG iterations across the last SolveQuasiStaticStep().
            int GetNumLinearIterations() const { return last_linear_iterations_; }

            /// Number of Newton iterations taken by the last solve.
            int GetNumNewtonIterations() const { return last_newton_iterations_; }

            /// Whether the last Newton solve converged.
            bool GetConverged() const { return last_converged_; }

            /**
             * @brief Phenomenological map from active tension to shortening factor.
             *
             * The Land model produces a tension, while the active strain
             * formulation needs a kinematic shortening. There is no unique
             * conversion; this is the usual saturating first guess,
             * gamma = gamma_max * Ta / (Ta + Ta_half), and it is provided so a
             * driver can run both formulations from the same cell model. A
             * quantitative study should instead calibrate gamma directly.
             */
            static real_t ShorteningFromTension(real_t Ta, real_t Ta_half = 40.0,
                                                real_t gamma_max = 0.15)
            {
                const real_t t = std::max(Ta, real_t(0.0));
                return gamma_max * t / (t + Ta_half);
            }

            /// Apply ShorteningFromTension() elementwise, T-vector to T-vector.
            static void ShorteningFromTension(const Vector &Ta, Vector &gamma,
                                              real_t Ta_half = 40.0,
                                              real_t gamma_max = 0.15);

        private:
            void BuildOperator();

            ParFiniteElementSpace &fes_;
            ParFiniteElementSpace &active_fes_;
            ParFiniteElementSpace &fiber_fes_;
            const IntegrationRule &ir_;

            ActiveFormulation formulation_;
            HolzapfelParameters params_;

            ParGridFunction u_gf_;
            Vector u_;
            Vector active_tvector_;
            Vector fiber_tvector_;
            Vector mesh_nodes_tdofs_;

            Array<int> ess_bdr_;

            int newton_max_iter_ = 20;
            real_t newton_rel_tol_ = 1e-8;
            real_t newton_abs_tol_ = 1e-12;
            int newton_print_level_ = 1;
            int linear_max_iter_ = 1000;
            real_t linear_rel_tol_ = 1e-6;
            real_t linear_abs_tol_ = 1e-14;
            int linear_print_level_ = -1;

            MechanicsPreconditioner prec_ = MechanicsPreconditioner::AMG;
            int last_newton_iterations_ = 0;
            int last_linear_iterations_ = 0;
            bool last_converged_ = false;

            // The dFEM operator and its Newton wrapper are held by pointer so
            // that SetFormulation() can rebuild them without reconstructing the
            // solver.
            class Impl;
            std::unique_ptr<Impl> impl_;
        };

    } // namespace electrophysiology
} // namespace mfem
