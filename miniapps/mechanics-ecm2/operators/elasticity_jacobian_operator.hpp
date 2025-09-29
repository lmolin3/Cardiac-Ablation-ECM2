#pragma once
#include "mfem.hpp"
#include "elasticity_operator.hpp"

// NOTE: All dfem features are experimental and under the mfem::future namespace.
// They might change their interface or behavior in upcoming releases until they have stabilized.
using namespace mfem::future;
using mfem::future::tensor;

namespace mfem
{
   namespace elasticity_ecm2
   {

      // Forward declaration of templated ElasticityOperator
      template <int dim>
      class ElasticityOperator;
      // Forward declaration of templated ElasticityJacobianPreconditioner
      template <int dim>
      class ElasticityJacobianPreconditioner;
      template <int dim>
      class AMGElasticityPreconditioner;
      template <int dim>
      class NestedElasticityPreconditioner;

      template <int dim>
      class ElasticityJacobianOperator : public Operator
      {

         //friend class ElasticityJacobianPreconditioner<dim>; // Allow Preconditioner to access private members
         friend class AMGElasticityPreconditioner<dim>;
         friend class NestedElasticityPreconditioner<dim>;
      public:
         ElasticityJacobianOperator(const ElasticityOperator<dim> *elasticity, const Vector &x);

         void Mult(const Vector &x, Vector &y) const override;

      private:
         const ElasticityOperator<dim> *elasticity;
         std::shared_ptr<DerivativeOperator> jacobian;
         mutable Vector z;
      };

      // Type aliases for convenience
      using ElasticityJacobianOperator2D = ElasticityJacobianOperator<2>;
      using ElasticityJacobianOperator3D = ElasticityJacobianOperator<3>;


   } // namespace elasticity_ecm2

} // namespace mfem
