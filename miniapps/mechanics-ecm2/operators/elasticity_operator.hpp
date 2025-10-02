#pragma once

// General mfem lib
#include "mfem.hpp"

// Common utilities
#include "../../common-ecm2/common_utils.hpp"

// Elasticity specific dependencies
#include "../definitions/materials.hpp"
#include "elasticity_utils.hpp"
#include "../solvers/quasi_static_elasticity_solver.hpp"
#include "elasticity_jacobian_operator.hpp"
#include "../preconditioners/elasticity_jacobian_preconditioner.hpp"

// NOTE: All dfem features are experimental and under the mfem::future namespace.
// They might change their interface or behavior in upcoming releases until they have stabilized.
using namespace mfem::future;
using mfem::future::tensor;
using namespace mfem::ecm2_utils;


namespace mfem
{
   namespace elasticity_ecm2
   {      
      // Forward declarations
      template<int dim> class ElasticitySolver;
      template<int dim> class ElasticityJacobianOperator;
      template<int dim> class ElasticityJacobianPreconditioner;
      template<int dim> class AMGElasticityPreconditioner;
      template<int dim> class NestedElasticityPreconditioner;

      // CRTP Base class for QFunction
      template <typename Derived, int dim>
      struct QFunctionBase
      {
         MFEM_HOST_DEVICE inline auto operator()(
             const tensor<real_t, dim, dim> &dudxi,
             const tensor<real_t, dim, dim> &J,
             const real_t &w) const -> tuple<tensor<real_t, dim, dim>>
         {
            return static_cast<const Derived *>(this)->operator()(dudxi, J, w);
         }
      };

      template <int dim>
      struct StressQFunction : public QFunctionBase<StressQFunction<dim>, dim>
      {
         StressQFunction(const MaterialVariant<dim> &mat) : material(mat) {}

         MFEM_HOST_DEVICE inline auto operator()(
             const tensor<real_t, dim, dim> &dudxi,
             const tensor<real_t, dim, dim> &J,
             const real_t &w) const -> tuple<tensor<real_t, dim, dim>>
         {
            auto invJ = inv(J);
            auto dudX = dudxi * invJ;
            auto P = std::visit([&dudX](const auto &mat)
                                { return mat(dudX); }, material);
            auto JxW = det(J) * w * transpose(invJ);
            return tuple{P * JxW};
         }

         MaterialVariant<dim> material;
      };

      /**
       * @brief ElasticityOperator for quasi-static non-linear elasticity problems.
       *
       * This class encapsulates the setup and solution of the non-linear elasticity equations
       * using MFEM's parallel finite element infrastructure. It supports essential and
       * Neumann boundary conditions, body traction, and different nonlinear hyperelastic materials
       * defined using the DifferentiableOperator interface.
       */
      template <int dim>
      class ElasticityOperator : public Operator
      {

         // Field identifiers for dfem
         static constexpr int Displacement = 1;
         static constexpr int Coordinates = 2;

         friend class ElasticitySolver<dim>; // Allow these classes to access private members
         friend class ElasticityJacobianOperator<dim>; 
         //friend class ElasticityJacobianPreconditioner<dim>;
         friend class AMGElasticityPreconditioner<dim>;
         friend class NestedElasticityPreconditioner<dim>;
      public:
         /**
          * @brief Construct a new ElasticityOperator.
          * @param fes_ The ParFiniteElementSpace to use (not owned).
          * @param verbose_ Verbosity flag for debugging (default false).
          */
         ElasticityOperator(ParFiniteElementSpace *fes_, bool verbose_ = true);

         //////////////////////////////////////////////////////
         //---/ BC interface /-------------------------------//
         //////////////////////////////////////////////////////

         /**
          * @brief Set the boundaries to be fixed.
          * @param fixed_bdr Array marking the boundary attributes for fixed constraints.
          * @param component The component to apply the fixed constraint to (-1 for all components).
          */
         void AddFixedConstraint(const Array<int> &fixed_bdr, int component = -1);
         void AddFixedConstraint(int attr, int component = -1);

         /**
          * @brief Set the prescribed an analytic (time-dependent) displacement (Dirichlet BC) on specified boundary attributes.
          * API allows setting the prescribed displacement:
          * 1. Using either VectorCoefficient/VecFuncT (alias for void(const Vector &x, real_t t, Vector &u))
          * 2. Specifying single boundary attribute or an array of attributes.
          * 3. Velocity and acceleration should also be specified.
          */
         void AddPrescribedDisplacement(VectorCoefficient *disp, Array<int> &attr, real_t scaling = 1.0, bool own = true);
         void AddPrescribedDisplacement(VecFuncT disp, Array<int> &attr, real_t scaling = 1.0);
         void AddPrescribedDisplacement(VectorCoefficient *disp, int attr, real_t scaling = 1.0, bool own = true);
         void AddPrescribedDisplacement(VecFuncT disp, int attr, real_t scaling = 1.0);

         /**
          * @brief Set the BoundaryLoad (Traction BC) on specified boundary attributes.
          * API allows setting the BoundaryLoad:
          * 1. Using either VectorCoefficient/VecFuncT (alias for void(const Vector &x, real_t t, Vector &u))
          * 2. Specifying single boundary attribute or an array of attributes.
          */
         void AddBoundaryLoad(VectorCoefficient *coeff, Array<int> &attr, real_t scaling = 1.0, bool own = true);
         void AddBoundaryLoad(VecFuncT func, Array<int> &attr, real_t scaling = 1.0);
         void AddBoundaryLoad(VectorCoefficient *coeff, int attr, real_t scaling = 1.0, bool own = true);
         void AddBoundaryLoad(VecFuncT func, int attr, real_t scaling = 1.0);
         /**
          * @brief Set the body force (volumetric load) on specified domain attributes.
          * API allows setting the BodyForce:
          * 1. Using either VectorCoefficient/VecFuncT (alias for void(const Vector &x, real_t t, Vector &u))
          * 2. Specifying single domain attribute or an array of attributes.
          */
         void AddBodyForce(VectorCoefficient *coeff, Array<int> &attrs, real_t scaling = 1.0, bool own = true);
         void AddBodyForce(VecFuncT func, Array<int> &attrs, real_t scaling = 1.0);
         void AddBodyForce(VectorCoefficient *coeff, int attr, real_t scaling = 1.0, bool own = true);
         void AddBodyForce(VecFuncT func, int attr, real_t scaling = 1.0);

         ///////////////////////////////////////////////////////////
         //---/ Setup and computing interface /-------------------//
         ///////////////////////////////////////////////////////////

         /**
          * @brief Assemble and initialize the system matrices and solvers.
            * @param ir_order The order of the integration rule to use (default is 2).
          */
         void Setup(int ir_order = 2);

         /**
          * @brief Compute the residual Y = R(U) representing the elasticity equation
          * with a material model chosen by calling SetMaterial.
          *
          * The output vector @a Y has essential degrees of freedom applied by setting
          * them to zero. This ensures R(U)_i = 0 being satisfied for each essential
          * dof i.
          *
          * @param U U
          * @param Y Residual R(U)
          */
         virtual void Mult(const Vector &U, Vector &Y) const override;

         /**
          * @brief Get the Gradient object
          *
          * Update and cache the state vector @a U, used to compute the linearization
          * dR(U)/dU.
          *
          * @param U
          * @return Operator&
          */
         Operator &GetGradient(const Vector &U) const override;


         /**
          * @brief Set the material type.
          *
          * This method sets the material type by instantiating the material lambda used
          * in the QFunction with the provided material model.
          *
          * @tparam material_type
          * @param[in] material
          */
         void SetMaterial(const MaterialVariant<dim> &material);

         /**
          * @brief Project the essential boundary conditions onto the solution vector X.
          * @param X The solution T-vector to project the essential boundary conditions onto.
          */
         void ProjectEssentialBCs(Vector &X) const;

         /**
          * @brief Update the system after mesh and/or coefficient changes.
          * For now it just updates the right-hand side vector.
          */
         void Update();

         /**
          * @brief Get the list of essential true degrees of freedom.
          * @return Reference to the array of essential true DOFs.
          */
         const Array<int>& GetEssTdofList() const { return ess_tdof_list; }

         /// Get the FE space used by the operator.
         ParFiniteElementSpace *GetFESpace() { return fes; }

      private:
         ParMesh *pmesh; ///< Reference to the parallel mesh.     // NOT OWNED
         const int sdim;  ///< Spatial dimension.
         int fes_truevsize;

         ParFiniteElementSpace *fes;               ///< NOT OWNED

         bool setup = false;   ///< Flag indicating if the system is set up.
         bool verbose = false; ///< Verbosity flag for debugging.

         mutable Vector B;       ///< rhs t-vector.

         // Linear form for rhs and bcs
         std::unique_ptr<ParLinearForm> Fform;    // OWNED

         // Material storage
         std::string material_type_name;           ///< Material type name
         bool material_set = false;                ///< Material set flag
         std::unique_ptr<StressQFunction<dim>> qfunction; ///< Type-erased QFunction

         // DifferentiableOperator for nonlinear mechanics
         std::unique_ptr<DifferentiableOperator> residual = nullptr;        ///< Residual operator.
         mutable std::unique_ptr<ElasticityJacobianOperator<dim>> jacobian_op = nullptr; ///< Jacobian operator.

         // Boundary conditions
         int max_bdr_attributes;
         int max_domain_attributes;
         std::vector<FixedConstraint> fixed_constraints;             ///< Fixed constraints.
         std::vector<VectorCoeffContainer> prescribed_displacements; ///< Prescribed displacements (Dirichlet conditions).
         std::vector<VectorCoeffContainer> boundary_loads;           ///< Boundary loads (Neumann conditions).
         std::vector<VectorCoeffContainer> body_forces;              ///< Body forces (on domain).

         Array<int> tmp_bdr_attrs;        ///< Temporary boundary attributes.
         Array<int> tmp_domain_attrs;     ///< Temporary domain attributes.
         Array<int> fixed_attrs;          ///< Fixed boundary attributes.
         Array<int> fixed_attrs_xyz;      ///< Fixed boundary attributes for each component.
         Array<int> prescribed_disp_attr; ///< Prescribed displacement attributes.
         Array<int> boundary_load_attr;   ///< Boundary load attributes.
         Array<int> body_force_attr;      ///< Body force attributes.  --> Domain attributes

         Array<int> fixed_tdof_list;           ///< List of true DOFs for fixed constraint (subset ess_tdof_list).
         Array<int> prescribed_disp_tdof_list; ///< List of true DOFs for prescribed displacements (subset ess_tdof_list).
         Array<int> ess_tdof_list;             ///< List of essential DOFs for essential BCs.
      };

      // Type aliases for convenience
      using ElasticityOperator2D = ElasticityOperator<2>;
      using ElasticityOperator3D = ElasticityOperator<3>;

   }
}
