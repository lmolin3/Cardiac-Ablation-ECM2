#pragma once
#include "mfem.hpp"

namespace mfem
{
   // TODO:
   // - implement PA preconditioner (LOR+AMG, or jacobi)
   //
   // * For jacobi we can just use the assemble_diagonal method
   // * For LOR we need to check if the API can work with DifferentiableOperator
   // (could create ParLORDiscretization given just ParFiniteElementSpace)

   namespace elasticity_ecm2
   {
      template <int dim>
      class ElasticityJacobianPreconditioner : public Solver
      {
      public:
         ElasticityJacobianPreconditioner();

         ~ElasticityJacobianPreconditioner();

         void SetOperator(const Operator &op) override;
         void Mult(const Vector &x, Vector &y) const override;

      private:
         HypreParMatrix* A = nullptr;
         std::unique_ptr<HypreBoomerAMG> amg;
      };
   } // namespace elasticity_ecm2

}