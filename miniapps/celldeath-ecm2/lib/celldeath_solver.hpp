#ifndef MFEM_CELLDEATH_SOLVER
#define MFEM_CELLDEATH_SOLVER

#include "mesh_extras.hpp"
#include "pfem_extras.hpp"
#include "utils.hpp"
#include "ThreeStateCellDeath.h"

#ifdef MFEM_USE_MPI

namespace mfem
{

    using common::H1_ParFESpace;

    namespace celldeath
    {

        class CellDeathSolver
        {
        public:
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

            Vector N, U, D, T, Tsrc;

            // Coefficients
            ParGridFunction *T_gf;
            real_t A1, A2, A3;
            real_t deltaE1, deltaE2, deltaE3;
            static constexpr real_t R = 8.31446261815324; // J/(mol*K)
            const real_t invR = 1.0 / R;
            real_t k1, k2, k3;

            // Postprocessing
            VisItDataCollection *visit_dc;       // To prepare fields for VisIt viewing
            ParaViewDataCollection *paraview_dc; // To prepare fields for ParaView viewing

            bool verbose;
        };

        class CellDeathSolverEigen : public CellDeathSolver
        {
        public:
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