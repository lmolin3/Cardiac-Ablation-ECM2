#pragma once

#include "../operators/elasticity_operator.hpp" 
#include "../preconditioners/elasticity_jacobian_preconditioner.hpp" 


namespace mfem
{

   namespace elasticity_ecm2
   {

      // Forward declaration of templated ElasticityOperator
      template <int dim> class ElasticityOperator;
      
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

         /// Setup the underlying solver with the operator.
         /// @param k_grad_update The gradient update frequency for the FrozenNewtonSolver.
         /// k_grad_update = 1 corresponds to the standard NewtonSolver.
         /// @param prec_type The type of preconditioner to use for the Jacobian solver (default AMG).
         /// @param Args Additional arguments to pass to the preconditioner.
         template <typename PrecTag, typename... Args>
         void Setup(int k_grad_update, Args &&...args)
         {
            op->Setup();

            if constexpr (std::is_same_v<PrecTag, AMG>)
            {
               j_prec = std::make_unique<AMGElasticityPreconditioner<dim>>(std::forward<Args>(args)...);
            }
            else if constexpr (std::is_same_v<PrecTag, NESTED>)
            {
               j_prec = std::make_unique<NestedElasticityPreconditioner<dim>>(comm, std::forward<Args>(args)...);
            }
            else
            {
               MFEM_ABORT("Unknown PreconditionerType in ElasticitySolver::Setup");
            }

            // j_prec->SetOperator(*op); // Should be called already inside the NewtonSolver-->GMRESolver

            if constexpr (std::is_same_v<PrecTag, AMG>)
               j_solver = std::make_unique<GMRESSolver>(comm);
            else if constexpr (std::is_same_v<PrecTag, NESTED>)
               j_solver = std::make_unique<FGMRESSolver>(comm);
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

            nonlinear_solver = std::make_unique<FrozenNewtonSolver>(comm, k_grad_update);
            //nonlinear_solver->iterative_mode = false;
            nonlinear_solver->SetOperator(*op);
            nonlinear_solver->SetAbsTol(0.0);
            nonlinear_solver->SetRelTol(1e-6);
            nonlinear_solver->SetMaxIter(500);
            nonlinear_solver->SetSolver(*j_solver);
            nonlinear_solver->SetPrintLevel(1);
            // nonlinear_solver->SetAdaptiveLinRtol(2, 0.5, 0.9);
         }

         /// Set the material model for the elasticity operator.
         void SetMaterial(const MaterialVariant<dim> &material)
         {
            op->SetMaterial(material);
         }

         /// Perform a single time step.
         void Solve();

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

         MPI_Comm comm;                                                //< NOT OWNED
         ParMesh *pmesh;                                                //< NOT OWNED

         FiniteElementCollection *fec = nullptr;                        //< OWNED
         ParFiniteElementSpace *fes = nullptr;                          //< OWNED

         mutable ParGridFunction u_gf;   ///< Grid function for displacement field.

         std::unique_ptr<ElasticityOperator<dim>> op;                   //< OWNED

         std::unique_ptr<NewtonSolver> nonlinear_solver;                //< OWNED
         std::unique_ptr<IterativeSolver> j_solver;                     //< OWNED
         std::unique_ptr<ElasticityJacobianPreconditioner<dim>> j_prec; //< OWNED

         std::unique_ptr<Vector> zero, U;                               //< OWNED
      };


      // Alias for 2D and 3D ElasticitySolver
      using ElasticitySolver2D = ElasticitySolver<2>;
      using ElasticitySolver3D = ElasticitySolver<3>;

   } // namespace elasticity_ecm2

} // namespace mfem


