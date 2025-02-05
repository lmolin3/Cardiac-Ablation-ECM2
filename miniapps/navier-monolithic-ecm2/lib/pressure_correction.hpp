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

            void SetSchurSolver(Solver *invS_, bool own_schur_ = false);

            void SetH1Solver(Solver *H1_, bool own_H1_ = false);

            void SetH1Operator(Operator *H2Op);

            private:
               Array<int> block_offsets;
               int usize, psize;
               Solver *invS = nullptr;
               Solver *H1 = nullptr;
               BlockOperator *nsOp = nullptr;
               bool own_schur = false;
               bool own_H1 = false;

               mutable Vector tmp1, tmp2, rhs; // Auxiliary vectors
      };

   } // namespace navier
} // namespace mfem

#endif // PRESSURE_CORRECTION_NAVIER_HPP
