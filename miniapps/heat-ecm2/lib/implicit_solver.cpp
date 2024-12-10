#include "implicit_solver.hpp"

using namespace std;
namespace mfem
{
    namespace heat
    {

        // Class for solver used in implicit time integration (base)
        ImplicitSolverBase::~ImplicitSolverBase()
        {
            delete linear_solver;
            linear_solver = nullptr;
            delete prec;
            prec = nullptr;
        }

        void ImplicitSolverBase::SetOperator(const Operator &op)
        {
            linear_solver->SetOperator(op);
        }

        // Class for solver used in implicit time integration
        ImplicitSolverFA::ImplicitSolverFA(HypreParMatrix *M_, HypreParMatrix *K_,
                                           Array<int> &ess_tdof_list_, int dim, bool use_advection)
            : ImplicitSolverBase(ess_tdof_list_), M(M_), K(K_), RobinMass(nullptr),
              T(nullptr), Te(nullptr)
        {
            // prec = new HypreSmoother();
            // prec->SetType(HypreSmoother::Jacobi); // See hypre.hpp for more options --> use default l1-scaled block Gauss-Seidel/SSOR
            prec = new HypreBoomerAMG();
            prec->iterative_mode = false;

            if (use_advection)
            {
                linear_solver = new GMRESSolver(M->GetComm());
                static_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
                static_cast<HypreBoomerAMG *>(prec)->SetAdvectiveOptions(15, "", "FFC"); // HypreBoomerAMG with advective options (AIR) default used, check hypre.hpp for more options
            }
            else
            {
                linear_solver = new CGSolver(M->GetComm());
                static_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
            }

            linear_solver->iterative_mode = false;
            linear_solver->SetRelTol(1e-8);
            linear_solver->SetAbsTol(0.0);
            linear_solver->SetMaxIter(500);
            linear_solver->SetPrintLevel(0);
            linear_solver->SetPreconditioner(*prec);
        };

        void ImplicitSolverFA::SetOperators(HypreParMatrix *M_, HypreParMatrix *K_, HypreParMatrix *RobinMass_)
        {
            M = M_;
            K = K_;
            RobinMass = RobinMass_;
        }

        void ImplicitSolverFA::SetTimeStep(real_t dt_)
        {
            if (dt_ == current_dt)
                return;

            current_dt = dt_;

            delete T;
            delete Te;

            BuildOperator();
        }

        void ImplicitSolverFA::Mult(const Vector &x, Vector &y) const
        {
            linear_solver->Mult(x, y);
        }

        void ImplicitSolverFA::BuildOperator()
        {
            if (T)
                delete T;
            if (Te)
                delete Te;

            MFEM_VERIFY((M != nullptr) && (K != nullptr), "Operator M and K not set");

            // T = M + dt*K + dt RobinMass = M + dt*(D + A - R) + dt RobinMass
            T = new HypreParMatrix(*M);
            if (RobinMass)
            {
                auto tmp = ParAdd(K, RobinMass);
                T->Add(current_dt, *tmp);
                delete tmp;
                tmp = nullptr;
            }
            else
            {
                T->Add(current_dt, *K);
            }

            Te = T->EliminateRowsCols(ess_tdof_list);
            linear_solver->SetOperator(*T);
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
                                           MatrixCoefficient *Kappa_, Coefficient *rhoC_,
                                           real_t alpha_, VectorCoefficient *u_,
                                           Coefficient *beta_, int prec_type_)
            : ImplicitSolverBase(ess_tdof_list_), fes(fes_), T(nullptr),
              lor(nullptr), dtKappa(nullptr), dtBeta(nullptr),
              dtConv(0.0), rhoC(rhoC_), Beta(beta_), u(u_), Kappa(Kappa_), alpha(alpha_),
              bcs(bcs_), has_diffusion(false), has_advection(false), has_reaction(false),
              prec_type(prec_type_)
        {
            // Check contributions
            has_reaction = Beta ? true : false;
            has_diffusion = Kappa ? true : false;
            has_advection = alpha_ != 0 && u ? true : false;

            // Create product coefficients (dependent on timestep)
            if (has_diffusion)
                dtKappa = new ScalarMatrixProductCoefficient(current_dt, *Kappa);
            if (has_advection)
                dtConv = current_dt * alpha;
            if (has_reaction)
                dtBeta = new ProductCoefficient(current_dt, *Beta);

            // Create bilinear form for operator T = M + dt*K + dt*RobinMass
            T = new ParBilinearForm(fes);
            T->AddDomainIntegrator(new MassIntegrator(*rhoC));
            if (has_diffusion)
                T->AddDomainIntegrator(new DiffusionIntegrator(*dtKappa));
            if (has_advection)
                T->AddDomainIntegrator(new ConvectionIntegrator(*u, dtConv));
            if (has_reaction)
                T->AddDomainIntegrator(new MassIntegrator(*dtBeta));

            for (auto &robin_bc : bcs->GetRobinBcs())
            {
                // Add a Mass integrator on the Robin boundary
                ProductCoefficient dtH(current_dt, *robin_bc.h_coeff);
                T->AddBoundaryIntegrator(new MassIntegrator(dtH), robin_bc.attr);
            }

            T->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            T->Assemble();
            T->FormSystemMatrix(ess_tdof_list, opT);

            // Preconditioner
            switch (prec_type)
            {
            case 0: // Jacobi Smoother
                prec = new OperatorJacobiSmoother(*T, ess_tdof_list);
                break;
            case 1: // LOR
                lor = new ParLORDiscretization(*T, ess_tdof_list);
                prec = new HypreBoomerAMG(lor->GetAssembledMatrix());
                static_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
                break;
            default:
                MFEM_ABORT("Unknown preconditioner type.");
            }

            // Linear solver
            if (has_advection)
            {
                linear_solver = new GMRESSolver(fes->GetComm());
            }
            else
            {
                linear_solver = new CGSolver(fes->GetComm());
            }

            linear_solver->iterative_mode = false;
            linear_solver->SetRelTol(1e-8);
            linear_solver->SetAbsTol(0.0);
            linear_solver->SetMaxIter(500);
            linear_solver->SetPrintLevel(0);
            linear_solver->SetOperator(*opT);
            linear_solver->SetPreconditioner(*prec);
        };

        void ImplicitSolverPA::SetTimeStep(double dt_)
        {
            // If the timestep has not changed, do nothing
            if (dt_ == current_dt)
            {
                return;
            }

            // If the timestep has changed:
            // 1. Update the coefficients
            // 2. Reassemble the operator
            // 3. Reset the solver

            current_dt = dt_;

            // Update the coefficients
            if (has_diffusion)
                dtKappa->SetAConst(current_dt);
            if (has_advection)
                dtConv = current_dt * alpha;
            if (has_reaction)
                dtBeta->SetAConst(current_dt);

            // Reassemble the operator
            delete T;
            opT.Clear();
            T = new ParBilinearForm(fes);
            T->AddDomainIntegrator(new MassIntegrator(*rhoC));
            if (has_diffusion)
                T->AddDomainIntegrator(new DiffusionIntegrator(*dtKappa));
            if (has_advection)
                T->AddDomainIntegrator(new ConvectionIntegrator(*u, dtConv));
            if (has_reaction)
                T->AddDomainIntegrator(new MassIntegrator(*dtBeta));

            for (auto &robin_bc : bcs->GetRobinBcs())
            {
                // Add a Mass integrator on the Robin boundary
                ProductCoefficient dtH(current_dt, *robin_bc.h_coeff);
                T->AddBoundaryIntegrator(new MassIntegrator(dtH), robin_bc.attr);
            }

            T->SetAssemblyLevel(AssemblyLevel::PARTIAL);
            T->Assemble();
            T->FormSystemMatrix(ess_tdof_list, opT);

            // Recreate the preconditioner
            delete prec;
            switch (prec_type)
            {
            case 0: // Jacobi Smoother
                prec = new OperatorJacobiSmoother(*T, ess_tdof_list);
                break;
            case 1: // LOR
                delete lor;
                lor = new ParLORDiscretization(*T, ess_tdof_list);
                prec = new HypreBoomerAMG(lor->GetAssembledMatrix());
                static_cast<HypreBoomerAMG *>(prec)->SetPrintLevel(0);
                break;
            default:
                MFEM_ABORT("Unknown preconditioner type.");
            }

            // Reset the solver
            delete linear_solver;
            if (has_advection)
            {
                linear_solver = new GMRESSolver(fes->GetComm());
            }
            else
            {
                linear_solver = new CGSolver(fes->GetComm());
            }
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
            delete T;
            delete dtKappa;
            delete dtBeta;
            delete lor;
        }

    } // namespace heat

} // namespace mfem