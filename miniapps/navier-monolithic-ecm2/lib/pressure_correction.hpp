/**
 * @file pressure_correction.hpp
 * @brief File containing declarations for various pressure corrections used in block preconditioners for Navier Stokes.
 */

#ifndef PRESSURE_CORRECTION_NAVIER_HPP
#define PRESSURE_CORRECTION_NAVIER_HPP

#include <mfem.hpp>
#include "utils.hpp"

namespace mfem
{
   using namespace ecm2_utils;

   namespace navier
   {
      ////////////////////////////////////////////////////////////////////
      //                    Pressure Correction Solver                  //
      ////////////////////////////////////////////////////////////////////

      /**
       * @brief Class for the pressure correction solver.
       *
       * This class is used to define the pressure correction solver for the Navier Stokes equations.
       */
      class PressureCorrectionSolver : public Solver
      {
         public:
            PressureCorrectionSolver(Array<int> &offsets); 

            ~PressureCorrectionSolver();
      
            void SetOperator(const Operator &op) override;

            void Mult(const Vector &x, Vector &y) const override;

            virtual void SetSchurSolver(Solver *invS_, bool own_schur_ = false);

            virtual void SetH1Solver(Solver *H1_, bool own_H1_ = false);

            virtual void SetH1Operator(Operator *H2Op);

            protected:
               Array<int> block_offsets;
               int usize, psize;
               Solver *invS = nullptr;
               Solver *H1 = nullptr;
               BlockOperator *nsOp = nullptr;
               bool own_schur = false;
               bool own_H1 = false;

               mutable Vector tmp1, tmp2, rhs; // Auxiliary vectors
      };


      /////////////////////////////////////////////////////////////////////////////
      ///                 High Order Pressure Correction Solver                 ///
      /////////////////////////////////////////////////////////////////////////////

      /**
       * @brief Class for the high order pressure correction solver.
       *
       * This class is used to define the high order pressure correction solver.
       *
       * see:
       * --> A. Veneziani and U. Villa. Aladins: An algebraic splitting time adaptive solver for the incompressible navier–stokes equations. Journal of Computational Physics, 238:359–375, 2013.
       */
      class HighOrderPressureCorrectionSolver : public PressureCorrectionSolver
      {
         public:
            HighOrderPressureCorrectionSolver(Array<int> &offsets, int q_order_ = 2); 
      
            // Set Operator A = K + N(u). This is required since SetOperator gives access only to C = alpha/dt M + K + N(u).
            void SetMomentumOperator(const Operator &op)
            {
               opA = &op;
            }

            void SetOrder(int q_order_);

            void Mult(const Vector &x, Vector &y) const override;

            // Get the latest pressure correction. This will be used for time adaptivity.
            Vector GetLatestPressureCorrection() const { return zq; }

            private:
               const Operator *opA = nullptr; // NOT OWNED
               int q_order; // Pressure Correction Order
               mutable std::vector<Vector> ZData; // Pressure Correction Data
               mutable Vector z, zq; // Temporary solution, final update
      };

   } // namespace navier
} // namespace mfem

#endif // PRESSURE_CORRECTION_NAVIER_HPP
