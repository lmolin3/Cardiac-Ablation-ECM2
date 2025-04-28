/**
 * @file navier_preconditioners.hpp
 * @brief File containing declarations for various block preconditioners for Navier Stokes.
 */

#pragma once

#ifndef PRECONDITIONERS_NAVIER_HPP
#define PRECONDITIONERS_NAVIER_HPP

#include <mfem.hpp>
#include "../../common-ecm2/utils.hpp"
#include "pressure_correction.hpp"

namespace mfem
{
   using namespace ecm2_utils;

   namespace navier
   {
      ////////////////////////////////////////////////////////////////////
      ///                         Abstract                             ///
      ////////////////////////////////////////////////////////////////////

      /**
       * @class NavierBlockPreconditioner
       * @brief Abstract class for Navier Stokes Block preconditioners.
       * The Navier Stokes operator has the form:
       * 
       *  A = [ C  G ]
       *      [ D  0 ] 
       */     

      class NavierBlockPreconditioner : public Solver
      {
      public:
         NavierBlockPreconditioner(Array<int> block_offsets_);

         virtual ~NavierBlockPreconditioner();

         virtual void SetOperator(const Operator &op)
         {
            nsOp = (BlockOperator *) &op;
         }

         virtual void Mult(const Vector &x, Vector &y) const = 0;

         virtual void SetSchurSolver(Solver *invS_, bool own_schur_ = false)
         {
            if (own_schur)
               delete invS;

            invS = invS_;
            own_schur = own_schur_;
         }

         virtual void SetMomentumSolver(Solver *invC_, bool own_momentum_ = false)
         {
            if (own_momentum)
               delete invC;

            invC = invC_;
            own_momentum = own_momentum_;
         }

      protected:
         Array<int> block_offsets;
         int usize, psize;
         BlockOperator *nsOp = nullptr;
         Solver *invC = nullptr;
         Solver *invS = nullptr;
         bool own_momentum = false;
         bool own_schur = false;
      };

      ////////////////////////////////////////////////////////////////////////////
      ///                         Block Diagonal Preconditioner                ///
      ////////////////////////////////////////////////////////////////////////////

      /**
       * @class NavierBlockDiagonalPreconditioner
       * @brief Block Diagonal Preconditioner for Navier Stokes. 
       * 
       * The block diagonal preconditioner is defined as:
       * 
       *       P = [ C     ]    
       *           [    -S ]  
       * 
       * with inverse :
       *              
       *    P^-1 = [ C^-1       ]
       *           [      -S^-1 ]
       *
       * where C is an approximation of the momentum block and S is an approximation of the Schur complement.
       * 
       */
      class NavierBlockDiagonalPreconditioner : public NavierBlockPreconditioner
      {
      public:
         NavierBlockDiagonalPreconditioner(Array<int> block_offsets_);

         void SetOperator(const Operator &op) override;

         void Mult(const Vector &x, Vector &y) const override;

      private:
      };


      /////////////////////////////////////////////////////////////////////////////////////
      ///                         Block Lower Triangular Preconditioner                 ///
      /////////////////////////////////////////////////////////////////////////////////////

      /**
       * @class NavierBlockLowerTriangularPreconditioner
       * @brief Block Lower Triangular Preconditioner for Navier Stokes.
       * 
       * The block lower triangular preconditioner is defined a:
       *    
       *       P = [ C     ]    
       *           [ D  -S ]  
       * 
       * with inverse :
       *              
       *    P^-1 = [ I       ][  I     ][ C^-1    ]
       *           [   -S^-1 ][ -D   I ][       I ]
       *
       * where C is an approximation of the momentum block and S is an approximation of the Schur complement.
       *  
       * see:
       * --> Veneziani, Alessandro. "Block factorized preconditioners for high‐order accurate in time approximation of the Navier‐Stokes equations." Numerical Methods for Partial Differential Equations: An International Journal 19.4 (2003): 487-510.
       */
      class NavierBlockLowerTriangularPreconditioner : public NavierBlockPreconditioner
      {
      public:
         NavierBlockLowerTriangularPreconditioner(Array<int> block_offsets_);

         void SetOperator(const Operator &op) override;

         void Mult(const Vector &x, Vector &y) const override;

      private:
      };


      /////////////////////////////////////////////////////////////////////////////////////
      ///                        Block Upper Triangular Preconditioner                 ///
      /////////////////////////////////////////////////////////////////////////////////////

      /**
       * @class NavierBlockUpperTriangularPreconditioner
       * @brief Block Upper Triangular Preconditioner for Navier Stokes.
       * 
       * The block upper triangular preconditioner is defined a:
       * 
       *      P = [ C   G ]
       *          [    -S ]
       * 
       * with inverse :
       * 
       *    P^-1 = [ C^-1   ][  I  -G ][ I       ]
       *           [      I ][      I ][   -S^-1 ]
       * 
       * where C is an approximation of the momentum block and S is an approximation of the Schur complement.
       * 
       * see: 
       * --> Elman, Howard C., et al. "A parallel block multi-level preconditioner for the 3D incompressible Navier–Stokes equations." Journal of Computational Physics 187.2 (2003): 504-523.
       */

      class NavierBlockUpperTriangularPreconditioner : public NavierBlockPreconditioner
      {
      public:
         NavierBlockUpperTriangularPreconditioner(Array<int> block_offsets_);

         void SetOperator(const Operator &op) override;

         void Mult(const Vector &x, Vector &y) const override;

      private:
      };

      
      //////////////////////////////////////////////////////////////////////////////////
      ///                     Algebraic Chorin Temam Preconditioner                  ///
      //////////////////////////////////////////////////////////////////////////////////

      /**
       * @class ChorinTemamPreconditioner
       * @brief Preconditioner based on Chorin-Temam (mass preserving) Algebraic Factorization of Navier Stokes operator.
       * 
       * The Chorin-Temam preconditioner is defined as:
       * 
       *    P = [ C     ][ I   H2 G ] = L U
       *        [ D  -S ][       I  ]
       * 
       * with inverse :
       * 
       *   P^-1 = U^-1 L^-1
       * 
       * where C is an approximation of the momentum block, S is an approximation of the Schur complement and H2 = (alpha/dt M)^-1.
       * 
       * see:
       * --> Veneziani, Alessandro. "Block factorized preconditioners for high‐order accurate in time approximation of the Navier‐Stokes equations." Numerical Methods for Partial Differential Equations: An International Journal 19.4 (2003): 487-510.
       */ 

      class ChorinTemamPreconditioner : public NavierBlockPreconditioner
      {
      public:
         ChorinTemamPreconditioner(Array<int> block_offsets_);

         ~ChorinTemamPreconditioner() override;
         
         void SetOperator(const Operator &op) override;

         void SetMomentumSolver(Solver *invC_, bool own_momentum_ = false) override;

         void SetSchurSolver(Solver *invS_, bool own_schur_ = false) override;

         void SetH2Solver(Solver *H2_, bool own_H2_ = false);

         void SetH2Operator(Operator *H2Op);

         void Mult(const Vector &x, Vector &y) const override;

      protected:
         Solver *H2 = nullptr;
         bool own_H2 = false;
         TransposeOperator *H2Gt = nullptr;
         ProductOperator *H2G = nullptr;
         BlockLowerTriangularPreconditioner *L = nullptr;
         BlockLowerTriangularPreconditioner *U = nullptr; // Define U^T as Lower Triangular, but we will use MultTranspose
         mutable BlockVector tmp;
      };


      /////////////////////////////////////////////////////////////////////////////////////
      ///                        Algebraic Yosida Preconditioner                        ///
      /////////////////////////////////////////////////////////////////////////////////////

      /**
       * @class YosidaPreconditioner
       * @brief Preconditioner based on Yosida (momentum preserving) Algebraic Factorization of Navier Stokes operator.
       * 
       * The Yosida preconditioner is defined as:
       * 
       *    P = [ C     ][ I   C^-1 G ] = L U
       *        [ D  -S ][         I  ]
       * 
       * with inverse :
       * 
       *   P^-1 = U^-1 L^-1
       * 
       * where C is an approximation of the momentum block, S is an approximation of the Schur complement.
       * 
       * see:
       * --> Veneziani, Alessandro. "Block factorized preconditioners for high‐order accurate in time approximation of the Navier‐Stokes equations." Numerical Methods for Partial Differential Equations: An International Journal 19.4 (2003): 487-510.
       */

      class YosidaPreconditioner : public NavierBlockPreconditioner
      {
      public:
         YosidaPreconditioner(Array<int> block_offsets_);

         ~YosidaPreconditioner() override;

         void SetOperator(const Operator &op) override;

         void SetMomentumSolver(Solver *invC_, bool own_momentum_ = false) override;

         void SetSchurSolver(Solver *invS_, bool own_schur_ = false) override;

         void SetH2Solver(Solver *H2_, bool own_H2_ = false);

         void SetH2Operator(Operator *H2Op);

         void Mult(const Vector &x, Vector &y) const override;

      protected:
         Solver *H2 = nullptr;
         bool own_H2 = false;
         TransposeOperator *H2Gt = nullptr;
         ProductOperator *H2G = nullptr;
         BlockLowerTriangularPreconditioner *L = nullptr;
         BlockLowerTriangularPreconditioner *U = nullptr; // Define U^T as Lower Triangular, but we will use MultTranspose
         mutable BlockVector tmp;
      };


      ////////////////////////////////////////////////////////////////////////////
      ///                 Yosida Pressure Corrected Preconditioner             ///
      ////////////////////////////////////////////////////////////////////////////

      /**
       * @class YosidaPressureCorrectedPreconditioner
       * @brief Preconditioner based on Yosida (momentum preserving) Algebraic Factorization of Navier Stokes operator with Pressure Correction.
       * 
       * The Yosida preconditioner is defined as:
       * 
       *    P = [ C     ][ I   C^-1 G ] = L U,    where Q = B^-1 S,   B = (D H1 C H1 G)
       *        [ D  -S ][         Q  ]
       * 
       * with inverse :
       * 
       *   P^-1 = U^-1 L^-1
       * 
       * where C is an approximation of the momentum block, S is an approximation of the Schur complement.
       * 
       * see:
       * --> Saleri, Fausto, and Alessandro Veneziani. "Pressure correction algebraic splitting methods for the incompressible Navier--Stokes equations." SIAM journal on numerical analysis 43.1 (2005): 174-194.
       * --> Badia, Santiago, and Ramon Codina. "Algebraic pressure segregation methods for the incompressible Navier-Stokes equations." Archives of Computational Methods in Engineering 15 (2007): 1-52.
       * --> Gauthier, Alain, Fausto Saleri, and Alessandro Veneziani. "A fast preconditioner for the incompressible Navier Stokes Equations." Computing and Visualization in science 6 (2004): 105-112.
       * --> Gervasio, Paola, and Fausto Saleri. "Algebraic fractional-step schemes for time-dependent incompressible Navier–Stokes equations." Journal of Scientific Computing 27.1 (2006): 257-269.
       */

      class YosidaPressureCorrectedPreconditioner : public YosidaPreconditioner
      {
      public:
         YosidaPressureCorrectedPreconditioner(Array<int> block_offsets_);

         ~YosidaPressureCorrectedPreconditioner() override;

         void SetOperator(const Operator &op) override;

         void SetSchurSolver(Solver *invS_, bool own_schur_ = false) override;

         using YosidaPreconditioner::Mult;

         // Methods to forward the SetH1Solver and SetH1Operator to the PressureCorrectionSolver
         void SetH1Solver(Solver *H1_, bool own_H1_ = false);

         void SetH1Operator(Operator *H1Op);

      protected:
         TransposeOperator *Qt = nullptr;
         PressureCorrectionSolver *Q = nullptr;
      };



   /////////////////////////////////////////////////////////////////////////////
   ///              Chorin Temam Pressure Corrected Preconditioner           ///
   /////////////////////////////////////////////////////////////////////////////

   /** 
    * @class ChorinTemamPressureCorrectedPreconditioner
    * @brief Preconditioner based on Chorin-Temam (mass preserving) Algebraic Factorization of Navier Stokes operator with Pressure Correction.
    * 
    * The Chorin-Temam preconditioner is defined as:
    * 
    *    P = [ C     ][ I   H2 G Q ] = [ C     ][ I   H2 G ] [ I    ] = L U T   where Q = B^-1 S,   B = (D H1 C H1 G)
    *        [ D  -S ][        Q   ]   [ D  -S ][       I  ] [    Q ]
    * 
    * with inverse :
    * 
    *   P^-1 = T^-1 U^-1 L^-1
    * 
    * where C is an approximation of the momentum block, S is an approximation of the Schur complement and H2 = (alpha/dt M)^-1.
    * 
    * see:
    * --> Saleri, Fausto, and Alessandro Veneziani. "Pressure correction algebraic splitting methods for the incompressible Navier--Stokes equations." SIAM journal on numerical analysis 43.1 (2005): 174-194.
    * --> Badia, Santiago, and Ramon Codina. "Algebraic pressure segregation methods for the incompressible Navier-Stokes equations." Archives of Computational Methods in Engineering 15 (2007): 1-52.
    * --> Gauthier, Alain, Fausto Saleri, and Alessandro Veneziani. "A fast preconditioner for the incompressible Navier Stokes Equations." Computing and Visualization in science 6 (2004): 105-112.
    * --> Gervasio, Paola, and Fausto Saleri. "Algebraic fractional-step schemes for time-dependent incompressible Navier–Stokes equations." Journal of Scientific Computing 27.1 (2006): 257-269.
    */

   class ChorinTemamPressureCorrectedPreconditioner : public ChorinTemamPreconditioner
   {
   public:
      ChorinTemamPressureCorrectedPreconditioner(Array<int> block_offsets_);

      ~ChorinTemamPressureCorrectedPreconditioner() override;

      void SetOperator(const Operator &op) override;

      void SetSchurSolver(Solver *invS_, bool own_schur_ = false) override;

      void Mult(const Vector &x, Vector &y) const override;

      // Methods to forward the SetH1Solver and SetH1Operator to the PressureCorrectionSolver
      void SetH1Solver(Solver *H1_, bool own_H1_ = false);

      void SetH1Operator(Operator *H1Op);

   private:
      PressureCorrectionSolver *Q = nullptr;
      BlockDiagonalPreconditioner *T = nullptr;
      mutable BlockVector tmp2;
   };

   /**
    * @class HOYPressureCorrectedPreconditioner
    * @brief Preconditioner based on Yosida (momentum preserving) Algebraic Factorization of Navier Stokes operator with High Order Pressure Correction.
    *
    * The Yosida preconditioner is defined as:
    *
    *    P = [ C     ][ I   C^-1 G ] = L U,    where Q is a high-order (q) pressure correction
    *        [ D  -S ][         Q  ]
    *
    * with inverse :
    *
    *   P^-1 = U^-1 L^-1
    *
    * where C is an approximation of the momentum block, S is an approximation of the Schur complement.
    *
    * Note: The default order for the high order pressure correction is 2.
    *       Alternatively this can be set at any time using MonolithicNavierSolver::SetPressureCorrectionOrder,
    *       after the preconditioner has been created.
    * 
    * see:
    * --> A. Veneziani and U. Villa. Aladins: An algebraic splitting time adaptive solver for the incompressible navier–stokes equations. Journal of Computational Physics, 238:359–375, 2013.
   */
   class HOYPressureCorrectedPreconditioner : public YosidaPreconditioner
   {
   public:
      HOYPressureCorrectedPreconditioner(Array<int> block_offsets_);

      ~HOYPressureCorrectedPreconditioner() override;

      void SetOrder(int q_order_) { Q->SetOrder(q_order_); }

      void SetMomentumOperator(const Operator *op) { Q->SetMomentumOperator(*op); }

      void SetOperator(const Operator &op) override;

      void SetSchurSolver(Solver *invS_, bool own_schur_ = false) override;

      using YosidaPreconditioner::Mult;

      // Methods to forward the SetH1Solver and SetH1Operator to the PressureCorrectionSolver
      void SetH1Solver(Solver *H1_, bool own_H1_ = false);

      void SetH1Operator(Operator *H1Op);

      const Vector& GetLastPressureCorrection() const { return Q->zq; }
      
      int GetPressureCorrectionOrder() const { return Q->q_order; }

   private:
      HighOrderPressureCorrectionSolver *Q = nullptr; // High Order Pressure Correction
      TransposeOperator *Qt = nullptr;
   };

   } // namespace navier
} // namespace mfem

#endif // PRECONDITIONERS_NS_HPP
