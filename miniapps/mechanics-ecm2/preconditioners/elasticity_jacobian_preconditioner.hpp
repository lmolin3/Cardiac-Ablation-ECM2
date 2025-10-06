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

      // --- Type def for runtime preconditioner selection ---
      enum class PreconditionerType : int
      {
         AMG=0,
         NESTED=1,
      };

      // --- Dummy argument for compile-time Tag Dispatch ---
      struct AMG {};
      struct NESTED {};

      /// AMG preconditioner for the elasticity Jacobian
      /// Uses HypreBoomerAMG internally, requiring the Jacobian to be assembled
      /// into a HypreParMatrix.
      template <int dim>
      class AMGElasticityPreconditioner : public Solver
      {
      public:
         AMGElasticityPreconditioner(bool amg_elast = false);
         ~AMGElasticityPreconditioner();

         void SetOperator(const Operator &op) override;
         void Mult(const Vector &x, Vector &y) const override;

      private:
         HypreParMatrix* A = nullptr;
         std::unique_ptr<HypreBoomerAMG> amg;
         bool amg_elast = false;
      };


      /// Nested iteration preconditioner for the elasticity Jacobian
      /// Uses an iterative solver for the inner solves (GMRES).
      /// This triggers the outer solver to be a flexible solver (FGMRES).
      template <int dim>
      class NestedElasticityPreconditioner : public Solver
      {
      public:
         NestedElasticityPreconditioner(int inner_iter_max_ = 5, real_t inner_tol_ = 1e-2);

         #ifdef MFEM_USE_MPI
            NestedElasticityPreconditioner(MPI_Comm comm_, int inner_iter_max_ = 5, real_t inner_tol_ = 1e-2);
         #endif

         ~NestedElasticityPreconditioner();

         void SetOperator(const Operator &op) override;
         void Mult(const Vector &x, Vector &y) const override;
      private:
         std::unique_ptr<IterativeSolver> inner_solver;
         int inner_iter_max;
         real_t inner_tol;
      };

   } // namespace elasticity_ecm2
}