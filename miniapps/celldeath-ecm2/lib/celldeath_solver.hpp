#ifndef MFEM_CELLDEATH_SOLVER
#define MFEM_CELLDEATH_SOLVER

#include "../../common/mesh_extras.hpp"
#include "../../common/pfem_extras.hpp"
#include "../../common-ecm2/utils.hpp"
#include "ThreeStateCellDeath.h"

#include <functional>
#include <variant>

#ifdef MFEM_USE_MPI

namespace mfem
{

    using common::H1_ParFESpace;

    namespace celldeath
    {
        /// An Arrhenius parameter can be supplied as either a constant
        /// (real_t) or a temperature-dependent function (T [K] -> value).
        using ArrheniusParam = std::variant<real_t,
                                            std::function<real_t(real_t)>>;

        /// Convert an ArrheniusParam to a callable.  A constant value c is
        /// wrapped into [c](real_t){ return c; }.
        inline std::function<real_t(real_t)>
        to_arrhenius_fn(const ArrheniusParam &p)
        {
            if (auto *c = std::get_if<real_t>(&p))
            {
                real_t val = *c;
                return [val](real_t) { return val; };
            }
            return std::get<std::function<real_t(real_t)>>(p);
        }

        class CellDeathSolver
        {
        public:
            /// Construct without Arrhenius parameters (set later via
            /// SetParameters or SetArrheniusFunctions).
            CellDeathSolver(int order,
                            ParGridFunction *T_,
                            bool verbose = false);

            /// Construct with constant Arrhenius parameters.
            CellDeathSolver(int order,
                            ParGridFunction *T_,
                            real_t A1_, real_t A2_, real_t A3_,
                            real_t deltaE1_, real_t deltaE2_, real_t deltaE3_,
                            bool verbose = false);

            virtual ~CellDeathSolver();

            // Setup the projection
            virtual void SetupProjection();

            // Project the temperature field
            virtual void ProjectTemperature(Vector &Tin, Vector &Tout);

            // Solve the system
            virtual void Solve(real_t t, real_t dt) = 0;

            // Visualization and Postprocessing
            virtual void RegisterVisItFields(VisItDataCollection &visit_dc_);

            virtual void RegisterParaviewFields(ParaViewDataCollection &paraview_dc_);

            virtual void AddParaviewField(const std::string &field_name, ParGridFunction *gf);

            virtual void AddVisItField(const std::string &field_name, ParGridFunction *gf);

            virtual void WriteFields(const int &it = 0, const real_t &time = 0);

            ParaViewDataCollection &GetParaViewDc() { return *paraview_dc; }
            VisItDataCollection &GetVisItDc() { return *visit_dc; }

            HYPRE_BigInt GetProblemSize();

            void display_banner(std::ostream &os);

            void SetVerbose(bool verbose_) { verbose = verbose_; }

            // Getters
            ParGridFunction &GetAliveCellsGf() { return N_gf; }
            ParGridFunction &GetVulnerableCellsGf() { return U_gf; }
            ParGridFunction &GetDeadCellsGf() { return D_gf; }

            ParGridFunction &GetDamageVariableGf() { return G_gf; }

            /// Update constant Arrhenius parameters at runtime.
            /// Also resets the T-dependent functions to return these constants.
            void SetParameters(real_t A1_, real_t A2_, real_t A3_,
                               real_t deltaE1_, real_t deltaE2_, real_t deltaE3_)
            {
                A1 = A1_; A2 = A2_; A3 = A3_;
                deltaE1 = deltaE1_; deltaE2 = deltaE2_; deltaE3 = deltaE3_;
                A1_fn  = [this](real_t) { return A1; };
                A2_fn  = [this](real_t) { return A2; };
                A3_fn  = [this](real_t) { return A3; };
                dE1_fn = [this](real_t) { return deltaE1; };
                dE2_fn = [this](real_t) { return deltaE2; };
                dE3_fn = [this](real_t) { return deltaE3; };
            }

            /// Provide temperature-dependent Arrhenius pre-factors and
            /// activation energies.  Each argument can be either a plain
            /// real_t (constant) or a std::function<real_t(real_t T)>.
            /// Example:
            /// @code
            ///   solver->SetArrheniusFunctions(
            ///       [](real_t T){ return T > 328.15 ? 3.56e19 : 8.87e70; },
            ///       5.35e8,   // constant A2
            ///       1.6e12,   // constant A3
            ///       [](real_t T){ return T > 328.15 ? 144.7e3 : 467.6e3; },
            ///       85.9e3,   // constant dE2
            ///       105.1e3); // constant dE3
            /// @endcode
            void SetArrheniusFunctions(
                ArrheniusParam A1_p,  ArrheniusParam A2_p,  ArrheniusParam A3_p,
                ArrheniusParam dE1_p, ArrheniusParam dE2_p, ArrheniusParam dE3_p)
            {
                A1_fn  = to_arrhenius_fn(A1_p);
                A2_fn  = to_arrhenius_fn(A2_p);
                A3_fn  = to_arrhenius_fn(A3_p);
                dE1_fn = to_arrhenius_fn(dE1_p);
                dE2_fn = to_arrhenius_fn(dE2_p);
                dE3_fn = to_arrhenius_fn(dE3_p);
            }

        protected:
            // Shared pointer to Mesh
            ParMesh *pmesh = nullptr; /// NOT OWNED
            int dim;

            // FE spaces
            FiniteElementCollection *fec;
            ParFiniteElementSpace *fes;
            ParFiniteElementSpace *fesT;
            int fes_truevsize;
            int fesT_truevsize;
            int order;
            int orderT;
            TrueTransferOperator *transferOp;

            // Grid functions and Vectors
            ParGridFunction N_gf; // Alive cells grid function
            ParGridFunction U_gf; // Vulnerable cells grid function
            ParGridFunction D_gf; // Dead cells grid function
            ParGridFunction G_gf; // Damage variable grid function (U+D)

            Vector N, U, D, G, T, Tsrc;

            // Coefficients
            ParGridFunction *T_gf;
            static constexpr real_t R = 8.31446261815324; // J/(mol*K)
            const real_t invR = 1.0 / R;
            real_t k1, k2, k3;

            // Temperature-dependent Arrhenius parameter functions.
            // Defaults (set in constructor) return the constant members.
            std::function<real_t(real_t)> A1_fn, A2_fn, A3_fn;
            std::function<real_t(real_t)> dE1_fn, dE2_fn, dE3_fn;
            real_t A1, A2, A3;
            real_t deltaE1, deltaE2, deltaE3;
            
            // Postprocessing
            VisItDataCollection *visit_dc;       // To prepare fields for VisIt viewing
            ParaViewDataCollection *paraview_dc; // To prepare fields for ParaView viewing

            bool verbose;
        };

        class CellDeathSolverEigen : public CellDeathSolver
        {
        public:
            /// Construct without Arrhenius parameters.
            CellDeathSolverEigen(int order,
                                 ParGridFunction *T_,
                                 bool verbose = false);

            /// Construct with constant Arrhenius parameters.
            CellDeathSolverEigen(int order,
                                 ParGridFunction *T_,
                                 real_t A1_, real_t A2_, real_t A3_,
                                 real_t deltaE1_, real_t deltaE2_, real_t deltaE3_,
                                 bool verbose = false);

            ~CellDeathSolverEigen();

            // Solve the system
            void Solve(real_t t, real_t dt) override;

        private:
            // Private methods
            // Compute eigenvalues/eigenvectors given the coefficients ki (handle cases in which ki = 0)
            inline void EigenSystem(real_t k1, real_t k2, real_t k3, Vector &lambda, DenseMatrix &P);

            // Eigenvalue problem
#ifndef MFEM_THREAD_SAFE
            Vector Xn;            // initial conditions
            Vector X;             // solution
            DenseMatrix P, Plu;   // eigenvector matrix
            Vector lambda;        // eigenvalues
            Vector exp_lambda_dt; // exp(lambda * dt)
#endif
        };

        class CellDeathSolverGotran : public CellDeathSolver
        {
        public:
            /// Construct without Arrhenius parameters.
            CellDeathSolverGotran(int order_, ParGridFunction *T_,
                                  bool verbose = false);

            /// Construct with constant Arrhenius parameters.
            CellDeathSolverGotran(int order_, ParGridFunction *T_,
                                  real_t A1_, real_t A2_, real_t A3_,
                                  real_t deltaE1_, real_t deltaE2_, real_t deltaE3_,
                                  bool verbose = false);

            ~CellDeathSolverGotran();
            // Solve the system
            void Solve(real_t t, real_t dt) override;
            void Solve(real_t t, real_t dt, int method = 1, int substeps = 1);

        private:
            // ODE model
            static const int num_param = 3;
            static const int num_states = 3;

            real_t parameters[num_param];
            real_t (*parameters_nodes)[num_param];
            real_t init_states[num_states];
            real_t (*states)[num_states];
        };

    } // namespace celldeath

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_CELLDEATH_SOLVER