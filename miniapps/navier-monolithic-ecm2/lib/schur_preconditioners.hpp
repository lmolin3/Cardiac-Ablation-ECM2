/**
 * @file preconditioners_ns.hpp
 * @brief File containing declarations for various preconditioners for Navier Stokes.
 */

#pragma once

#ifndef SCHUR_PRECONDITIONERS_NAVIER_HPP
#define SCHUR_PRECONDITIONERS_NAVIER_HPP

#include <mfem.hpp>

namespace mfem
{
   namespace navier
   {
      /**
       * @class SchurComplementPreconditioner
       * @brief Abstract class for Schur Complement Preconditioner.
       */
      class SchurComplementPreconditioner : public Solver
      {
      public:
         SchurComplementPreconditioner(int s) : Solver(s) {};

         virtual ~SchurComplementPreconditioner() {};

         void Mult(const Vector &x, Vector &y) const override = 0;

         // Set operator for the preconditioner
         void SetOperator(const Operator &op) override {};

         // Rebuild the preconditioner
         void Rebuild() {};
      };

      /**
       * @class PMass
       * @brief Pressure mass preconditioner: P^{-1} = kin_vis Mp^{-1}
       */
      class PMass : public SchurComplementPreconditioner
      {
      public:
         PMass(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs, real_t kin_vis_);

         ~PMass() override;

         void SetCoefficients(real_t kin_vis_) { kin_vis = kin_vis_; }

         void Mult(const Vector &x, Vector &y) const override;

      private:
         OperatorHandle Mp;
         ParBilinearForm* mp_form = nullptr;
         Solver *Mp_inv = nullptr;
         real_t kin_vis;
      };

      /**
       * @class PLap
       * @brief Pressure laplacian preconditioner: P^{-1} = sigma Lp^{-1}
       */
      class PLap : public SchurComplementPreconditioner
      {

      public:
         PLap(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs, real_t sigma_);

         ~PLap() override;

         void Mult(const Vector &x, Vector &y) const override;

         void SetCoefficients(real_t sigma_) { sigma = sigma_; }

      private:
         OperatorHandle Lp;
         ParBilinearForm *lp_form = nullptr;
         Solver *Lp_inv = nullptr;
         real_t sigma;
      };

      /**
       * @class PCD
       * @brief PCD Preconditioner: P^{-1} = Mp^{-1} Fp Lp^{-1} with Fp = sigma Mp + kin_vis Lp + Np
       *
       * See:
       *  1. Elman H. C., Silvester D. J., Wathen A. J. (2014) Finite Elements and Fast Iterative Solvers: With Applications in Incompressible Fluid Dynamics.
       */

      class PCD : public SchurComplementPreconditioner
      {

      public:
         PCD(ParFiniteElementSpace *pres_fes_, Array<int> &pres_ess_tdofs_,
             Coefficient *mass_coeff, Coefficient *diff_coeff, VectorCoefficient *conv_coeff_ = nullptr);

         ~PCD() override;

         void Mult(const Vector &x, Vector &y) const override;

         void SetCoefficients(VectorCoefficient *velocity_) { conv_coeff = velocity_; }

         void Rebuild();

      private:
         ParFiniteElementSpace *pres_fes = nullptr;
         Array<int> pres_ess_tdofs;
         Array<int> ess_tdofs_pcd; // Needed for the Fp operator

         Coefficient *mass_coeff = nullptr;
         Coefficient *diff_coeff = nullptr;
         VectorCoefficient *conv_coeff = nullptr;

         ParBilinearForm *mp_form = nullptr;
         OperatorHandle Mp;

         ParBilinearForm *lp_form = nullptr;
         OperatorHandle Lp;

         ParBilinearForm *fp_form = nullptr;
         OperatorHandle Fp;

         Solver *Mp_inv;
         Solver *Lp_inv;

         mutable Vector z, w;
      };

      /**
       * @class CahouetChabard
       * @brief Cahouet Chabard preconditioner: P^{-1} = 1/dt Lp^{-1} + kin_vis Mp^{-1}
       *
       * See:
       * 1. Cahouet, J., and J‐P. Chabard. "Some fast 3D finite element solvers for the generalized Stokes problem." International Journal for Numerical Methods in Fluids 8.8 (1988): 869-895.
       * 2. Veneziani, Alessandro. "Block factorized preconditioners for high‐order accurate in time approximation of the Navier‐Stokes equations." Numerical Methods for Partial Differential Equations: An International Journal 19.4 (2003): 487-510.
       */
      class CahouetChabard : public SchurComplementPreconditioner
      {

      public:
         CahouetChabard(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs, real_t dt, real_t kin_vis);

         ~CahouetChabard() override;

         void Mult(const Vector &x, Vector &y) const override;

         void SetCoefficients(real_t dt_, real_t kin_vis_)
         {
            dt = dt_;
            kin_vis = kin_vis_;
         }

      private:
         Array<int> pres_ess_tdofs;

         ParBilinearForm *mp_form = nullptr;
         OperatorHandle Mp;

         ParBilinearForm *lp_form = nullptr;
         OperatorHandle Lp;
         
         Solver *Lp_inv;
         Solver *Mp_inv;

         real_t dt;
         real_t kin_vis;

         mutable Vector z;
      };

      /**
       * @class LSC
       * @brief Least Squares Commutator preconditioner: P^{-1} = (D T^{-1} G)^{-1}  ( D T^{-1} C T^{-1} G )   (D T^{-1} G)^{-1},  with T = diag(Mv), C = 1/dt Mv + kin_vis K  + N(u*)
       *
       * See:
       * 1. Elman, Howard C., David J. Silvester, and Andrew J. Wathen. Finite elements and fast iterative solvers: with applications in incompressible fluid dynamics. Vol. 22. Oxford university press, 2014.
       */
      class LSC : public SchurComplementPreconditioner
      {

      public:
         LSC(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs_, HypreParMatrix *D_, HypreParMatrix *G_, HypreParMatrix *Mv);

         ~LSC() override;

         void Mult(const Vector &x, Vector &y) const override;

         void SetOperator(const Operator &op) override { opC = const_cast<Operator *>(&op); };

      private:
         HypreParMatrix *D = nullptr;
         HypreParMatrix *G = nullptr;
         Operator *opC = nullptr;

         OperatorHandle S;
         Solver *invS = nullptr;
         Vector* diagT;

         Array<int> pres_ess_tdofs;

         mutable Vector z1, z2, q;
      };

      /**
       * @class ApproximateDiscreteLaplacian
       * @brief Wrapper for approximate Discrete Laplacian preconditioner: P^{-1} = sigma (D diag{M}^{-1} G)^{-1}
       */
      class ApproximateDiscreteLaplacian : public SchurComplementPreconditioner
      {
      public:
         ApproximateDiscreteLaplacian(ParFiniteElementSpace *pres_fes, Array<int> &pres_ess_tdofs_, const HypreParMatrix *D, const HypreParMatrix *G, const HypreParMatrix *Mv, real_t sigma_); // @note : for PA we'll need to pass the BilinearForm

         ~ApproximateDiscreteLaplacian() override;

         void Mult(const Vector &x, Vector &y) const override;

         void SetCoefficients(real_t sigma_) { sigma = sigma_; }

      private:
         OperatorHandle S;
         Solver *invS = nullptr;
         real_t sigma;

         Array<int> pres_ess_tdofs;
      };

   } // namespace navier
} // namespace mfem

#endif // SCHUR_PRECONDITIONERS_NAVIER_HPP
