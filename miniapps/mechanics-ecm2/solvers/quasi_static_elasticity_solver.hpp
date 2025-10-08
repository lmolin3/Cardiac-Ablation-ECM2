#pragma once

#include "../operators/elasticity_operator.hpp" 
#include "../preconditioners/elasticity_jacobian_preconditioner.hpp" 
#include "../definitions/materials.hpp"

namespace mfem
{

   namespace elasticity_ecm2
   {

      // Forward declaration of templated ElasticityOperator
      template <int dim> class ElasticityOperator;

#ifdef MFEM_USE_SUNDIALS
      enum class ElasticityKINSolverType : int
      {
         NONE = KIN_NONE,
         LINESEARCH = KIN_LINESEARCH,
         PICARD = KIN_PICARD,
         FIXEDPOINT = KIN_FP
      };
#endif

      /** @brief A wrapper class for second order ODE solvers for elasticity problems.
          This class holds a pointer to a SecondOrderODESolver, which is selected at runtime,
          and delegates all solver-related calls to it.
       */
      template <int dim>
      class ElasticitySolver
      {
      public:
         /** @brief Construct an ElasticitySolver.
             @param solver_type The type of SecondOrderODESolver to use (see ode.hpp).
             @param op_ The ElasticityOperator to use (not owned).
          */
         ElasticitySolver(ParMesh *pmesh_, int order, bool verbose_ = true);

         ~ElasticitySolver();

         /// Setup the jacobian preconditioner
         /// @param prec_type The type of preconditioner to use for the Jacobian solver (default AMG).
         /// @param Args Additional arguments to pass to the preconditioner.
         template <typename PrecTag, typename... Args>
         void SetupJacobianPreconditioner(Args &&...args)
         {
            if constexpr (std::is_same_v<PrecTag, AMG>)
            {
               prec_type = PreconditionerType::AMG;
               j_prec = std::make_unique<AMGElasticityPreconditioner<dim>>(std::forward<Args>(args)...);
               j_solver = std::make_unique<GMRESSolver>(comm);
            }
            else if constexpr (std::is_same_v<PrecTag, NESTED>)
            {
               prec_type = PreconditionerType::NESTED;
               j_prec = std::make_unique<NestedElasticityPreconditioner<dim>>(comm, std::forward<Args>(args)...);
               j_solver = std::make_unique<FGMRESSolver>(comm);
            }
            else
            {
               MFEM_ABORT("Unknown PreconditionerType in ElasticitySolver::Setup");
            }

            j_solver->iterative_mode = false;
            j_solver->SetAbsTol(0.0);
            j_solver->SetRelTol(1e-4);
            // j_solver->SetKDim(500);
            j_solver->SetMaxIter(500);
            j_solver->SetPrintLevel(2);
            j_solver->SetPreconditioner(*j_prec);
         }

         /// @brief Configure the nonlinear solver for the elasticity problem.
         void SetupNonlinearSolver(int k_grad_update = 1);
#ifdef MFEM_USE_SUNDIALS
         // Setup for KINSOL nonlinear solver
         // kinsol_nls_type: Type of KINSOL solver (NONE = full Newton, LINESEARCH = newton with globalization, PICARD = Picard, FIXEDPOINT = Fixed Point)
         // enable_jfnk: Whether to use JFNK (Jacobian-Free Newton-Krylov). Only works if kinsol_nls_type is not PICARD. 
         // kinsol_aa_n: Number of previous solutions to use for Anderson Acceleration (only for PICARD and FIXEDPOINT)
         // kinsol_damping: Damping factor for the update (only for PICARD and FIXEDPOINT)
         // max_setup_calls: Maximum number of times to call Setup(), i.e. how frequently to rebuild the Jacobian.
         //
         // TODO: 
         // 1. For now it fails in some cases, need to check if it's due to need for load ramping
         // (unlikely tho as MFEM Newton converges). One failing example is -pt 0, with prescribed displacement -0.1.
         // Also not sure why in those cases with JFNK it converges.
         // 2. Add more overloads to SetupNonlinearSolver for specific solver types (ensure safer selection of parameters)
         void SetupNonlinearSolver(ElasticityKINSolverType kinsol_nls_type, bool enable_jfnk = false, real_t kinsol_aa_n = 0.0, real_t kinsol_damping = 0.0, int max_setup_calls = 4);
#endif

         void Setup()
         {
            ///----- Setup the operator
            op->Setup();
         }

         /// @brief Enable load ramping for the simulation.
         /// @param num_steps Number of load increments to use (default 10).
         // This enables load ramping for all BCs applied (prescribed displacement, traction, body force).
         // If you want to keep some BCs from being ramped, you can inform it when setting the BCs.
         void EnableLoadRamping(int num_steps = 10);

         /// Set the material model for the elasticity operator.
         void SetMaterial(const MaterialVariant<dim> &material)
         {
            op->SetMaterial(material);
         }

         /// Perform the entire solve.
         void Solve();

         /// Solve a single step of load ramping. Useful when user wants to control and save the load steps.
         /// Returns the current step stored in the operator after solving
         // (incremented by 1 from the input current step)
         int SolveCurrentStep();

         /// Get the FE space used by the operator.
         ParFiniteElementSpace *GetFESpace() { return fes; }

         /// Get the displacement grid function
         ParGridFunction *GetDisplacementGridFunction() { return &u_gf; }

         HYPRE_BigInt GetProblemSize();

         // Expose the interface for adding bcs to the operator
         void AddFixedConstraint(const Array<int> &fixed_bdr, int component = -1);
         void AddFixedConstraint(int attr, int component = -1);

         void AddPrescribedDisplacement(VectorCoefficient *disp, Array<int> &attr, real_t scaling = 1.0, bool own = true);
         void AddPrescribedDisplacement(VecFuncT disp, Array<int> &attr, real_t scaling = 1.0);
         void AddPrescribedDisplacement(VectorCoefficient *disp, int attr, real_t scaling = 1.0, bool own = true);
         void AddPrescribedDisplacement(VecFuncT disp, int attr, real_t scaling = 1.0);

         void AddBoundaryLoad(VectorCoefficient *coeff, Array<int> &attr, real_t scaling = 1.0, bool own = true);
         void AddBoundaryLoad(VecFuncT func, Array<int> &attr, real_t scaling = 1.0);
         void AddBoundaryLoad(VectorCoefficient *coeff, int attr, real_t scaling = 1.0, bool own = true);
         void AddBoundaryLoad(VecFuncT func, int attr, real_t scaling = 1.0);

         void AddBodyForce(VectorCoefficient *coeff, Array<int> &attrs, bool own = true);
         void AddBodyForce(VecFuncT func, Array<int> &attrs);
         void AddBodyForce(VectorCoefficient *coeff, int attr, bool own = true);
         void AddBodyForce(VecFuncT func, int attr); 
      
      private:
         void SetupNonlinearSolverCommon(int pl = 0);

         MPI_Comm comm;                                                 //< NOT OWNED
         ParMesh *pmesh;                                                //< NOT OWNED

         FiniteElementCollection *fec = nullptr;                        //< OWNED
         ParFiniteElementSpace *fes = nullptr;                          //< OWNED


         bool verbose = true;
         mutable ParGridFunction u_gf;   ///< Grid function for displacement field.

         std::unique_ptr<ElasticityOperator<dim>> op;                   //< OWNED

         std::unique_ptr<NewtonSolver> nonlinear_solver;                //< OWNED

         PreconditionerType prec_type;
         std::unique_ptr<IterativeSolver> j_solver;                     //< OWNED
         std::unique_ptr<Solver> j_prec; //< OWNED

         std::unique_ptr<Vector> zero, U;                               //< OWNED

         // Load ramping
         bool load_ramping = false;
         int load_ramping_steps = 1;                               //< Number of load increments (Default: 1, i.e. no ramping)
         int cached_step = 1;                                   
      };

      // Alias for 2D and 3D ElasticitySolver
      using ElasticitySolver2D = ElasticitySolver<2>;
      using ElasticitySolver3D = ElasticitySolver<3>;

   } // namespace elasticity_ecm2

} // namespace mfem


