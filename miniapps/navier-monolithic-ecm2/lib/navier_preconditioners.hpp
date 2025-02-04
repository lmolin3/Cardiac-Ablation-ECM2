/**
 * @file preconditioners_ns.hpp
 * @brief File containing declarations for various preconditioners for Navier Stokes.
 */

#ifndef PRECONDITIONERS_NAVIER_HPP
#define PRECONDITIONERS_NAVIER_HPP

#include <mfem.hpp>
#include "utils.hpp"

namespace mfem
{
   using namespace ecm2_utils;

   namespace navier
   {
      ////////////////////////////////////////////////////////////////////
      ///                         Abstract                             ///
      ////////////////////////////////////////////////////////////////////

      /**
       * @class MonolithicNavierPreconditioner
       * @brief Abstract class for Navier Stokes Block preconditioners.
       * The Navier Stokes operator has the form:
       * 
       *  A = [ C  G ]
       *      [ D  0 ] 
       */     

      class MonolithicNavierPreconditioner : public Solver
      {
      public:
         MonolithicNavierPreconditioner(Array<int> block_offsets_);

         virtual ~MonolithicNavierPreconditioner();

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
      class NavierBlockDiagonalPreconditioner : public MonolithicNavierPreconditioner
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
      class NavierBlockLowerTriangularPreconditioner : public MonolithicNavierPreconditioner
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
       *      P = [ C  G ]
       *          [    D ]
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

      class NavierBlockUpperTriangularPreconditioner : public MonolithicNavierPreconditioner
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

      class ChorinTemamPreconditioner : public MonolithicNavierPreconditioner
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

      private:
         Solver *H2 = nullptr;
         bool own_H2 = false;
         TransposeOperator *H2Gt = nullptr;
         ProductOperator *H2G = nullptr;
         BlockLowerTriangularPreconditioner *L = nullptr;
         BlockLowerTriangularPreconditioner *U = nullptr; // Define it as Lower Triangular, but we will use MultTranspose
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

      class YosidaPreconditioner : public MonolithicNavierPreconditioner
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

      private:
         Solver *H2 = nullptr;
         bool own_H2 = false;
         TransposeOperator *H2Gt = nullptr;
         ProductOperator *H2G = nullptr;
         BlockLowerTriangularPreconditioner *L = nullptr;
         BlockLowerTriangularPreconditioner *U = nullptr; // Define it as Lower Triangular, but we will use MultTranspose
         mutable BlockVector tmp;
      };



   } // namespace navier
} // namespace mfem

#endif // PRECONDITIONERS_NS_HPP
