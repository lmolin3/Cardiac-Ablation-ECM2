/**
 * @file preconditioners_ns.hpp
 * @brief File containing declarations for various preconditioners for Navier Stokes.
 */

#ifndef PRECONDITIONERS_NS_HPP
#define PRECONDITIONERS_NS_HPP

#include <mfem.hpp>

namespace mfem
{
   namespace navier
   {
   /**
    * @class Builder
    * @brief Abstract class for navier stokes preconditioner.
    */
   class NavierStokesPC : public Solver
   {
   public:
      NavierStokesPC(int s) : Solver(s) {};

      virtual ~NavierStokesPC() {};

      void Mult(const Vector &x, Vector &y) const override = 0;

      virtual void SetCoefficients() = 0;

      void SetOperator(const Operator &op) override {};
   };

   /**
    * @class PCBuilder
    * @brief Abstract class for preconditioner builders.
    */
   class PCBuilder
   {
   public:
      virtual ~PCBuilder() {}

      /**
       * @brief Get the solver built by the builder.
       * @return The built solver.
       */
      virtual NavierStokesPC &GetSolver() = 0;
   };
   
   /**
    * @class PCD
    * @brief PCD Preconditioner: P^{-1} = Mp^{-1} Fp Lp^{-1} with Fp = a Mp + b Lp
    */
   class PCD : public NavierStokesPC
   {

   public:
      PCD(Solver &Mp_inv, Solver &Lp_inv, OperatorHandle &Fp);

      void Mult(const Vector &x, Vector &y) const override;

      void SetCoefficients() override {};

   private:
      Solver &Mp_inv;
      Solver &Lp_inv;
      OperatorHandle Fp;
      mutable Vector z, w;
   };

   /**
    * @class PCDBuilder
    * @brief Builder for the PCD preconditioner.
    */
   class PCDBuilder : public PCBuilder
   {
   public:
      PCDBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs,
                 Coefficient *mass_coeff, Coefficient *diff_coeff);

      PCDBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs);

      ~PCDBuilder();

      NavierStokesPC &GetSolver();

   private:
      ParBilinearForm mp_form;
      OperatorHandle Mp;

      ParBilinearForm lp_form;
      OperatorHandle Lp;

      ParBilinearForm fp_form;
      OperatorHandle Fp;

      Solver *Lp_inv;
      Solver *Mp_inv;

      PCD *pcd = nullptr;
   };

   /**
    * @class CahouetChabardPC
    * @brief Cahouet Chabard preconditioner: P^{-1} = 1/dt Lp^{-1} + Mp^{-1}
    */
   class CahouetChabardPC : public NavierStokesPC
   {

   public:
      CahouetChabardPC(Solver &Mp_inv, Solver &Lp_inv, Array<int> pres_ess_tdofs, real_t dt);

      void Mult(const Vector &x, Vector &y) const override;

      void SetCoefficients() override {};

      void SetCoefficients(real_t new_dt){ dt = new_dt; }

   private:
      Solver &Mp_inv;
      Solver &Lp_inv;
      mutable Vector z;
      real_t dt;
      Array<int> pres_ess_tdofs;
   };

   /**
    * @class CahouetChabardBuilder
    * @brief Builder for the CahouetChabard preconditioner.
    */
   class CahouetChabardBuilder : public PCBuilder
   {
   public:
      CahouetChabardBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs, real_t dt);

      ~CahouetChabardBuilder();

      NavierStokesPC &GetSolver();

   private:
      ParBilinearForm mp_form;
      OperatorHandle Mp;

      ParBilinearForm lp_form;
      OperatorHandle Lp;

      Solver *Lp_inv;
      Solver *Mp_inv;

      CahouetChabardPC *cahouet_chabard = nullptr;
   };

   /**
    * @class SchurApproxInvPC
    * @brief Approximate inverse preconditioner: P^{-1} = dt Mp^{-1} + Lp^{-1}
    */
   class SchurApproxInvPC : public NavierStokesPC
   {

   public:
      SchurApproxInvPC(Solver &Mp_inv, Solver &Lp_inv, Array<int> pres_ess_tdofs, real_t dt);

      void Mult(const Vector &x, Vector &y) const override;

      void SetCoefficients() override {};

      void SetCoefficients(real_t new_dt) { dt = new_dt; }

   private:
      real_t dt;
      Solver &Mp_inv;
      Solver &Lp_inv;
      mutable Vector z;
      Array<int> pres_ess_tdofs;
   };

   /**
    * @class SchurApproxInvBuilder
    * @brief Builder for the SchurApproxInv preconditioner.
    */
   class SchurApproxInvBuilder : public PCBuilder
   {
   public:
      SchurApproxInvBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs, real_t dt);

      ~SchurApproxInvBuilder();

      NavierStokesPC &GetSolver();

   private:
      ParBilinearForm mp_form;
      OperatorHandle Mp;
      // SparseMatrix M_local;
      // UMFPackSolver *Mp_inv;

      ParBilinearForm lp_form;
      OperatorHandle Lp;

      Solver *Lp_inv;
      Solver *Mp_inv;

      SchurApproxInvPC *schur_approx_inv = nullptr;
   };

   /**
    * @class PMassPC
    * @brief Pressure mass preconditioner: P^{-1} = 1/dt Mp^{-1}
    */
   class PMassPC : public NavierStokesPC
   {

   public:
      PMassPC(Solver &Mp_inv, real_t dt);

      void Mult(const Vector &x, Vector &y) const override;

      void SetCoefficients() override {};

      void SetCoefficients(real_t new_dt) { dt = new_dt; }

   private:
      Solver &Mp_inv;
      real_t dt;
   };

   /**
    * @class PMassBuilder
    * @brief Builder for the PMassPC preconditioner.
    */
   class PMassBuilder : public PCBuilder
   {
   public:
      PMassBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs, real_t dt);

      ~PMassBuilder();

      NavierStokesPC &GetSolver();

   private:
      ParBilinearForm mp_form;
      OperatorHandle Mp;
      SparseMatrix M_local;
      Solver *Mp_inv;
      PMassPC *pmass = nullptr;
   };

   /**
    * @class PLapPC
    * @brief Pressure laplacian preconditioner: P^{-1} = Lp^{-1}
    */
   class PLapPC : public NavierStokesPC
   {

   public:
      PLapPC(Solver &Lp_inv);

      void Mult(const Vector &x, Vector &y) const override;

      void SetCoefficients() override {};

   private:
      Solver &Lp_inv;
   };

   /**
    * @class PLapBuilder
    * @brief Builder for the Pressure Laplacian preconditioner.
    */
   class PLapBuilder : public PCBuilder
   {
   public:
      PLapBuilder(ParFiniteElementSpace &pres_fes, Array<int> pres_ess_tdofs);

      ~PLapBuilder();

      NavierStokesPC &GetSolver();

   private:
      ParBilinearForm lp_form;
      OperatorHandle Lp;
      SparseMatrix L_local;
      Solver *Lp_inv;
      PLapPC *plap = nullptr;
   };

   } // namespace navier
} // namespace mfem

#endif // PRECONDITIONERS_NS_HPP
