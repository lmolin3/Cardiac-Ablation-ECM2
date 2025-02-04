#include "navier_preconditioners.hpp"

namespace mfem
{

    namespace navier
    {

        /////////////////////////////////////////////////////////////////////////////
        ///                     Abstract Monolithic Preconditioner                ///
        /////////////////////////////////////////////////////////////////////////////

        // Implementation of MonolithicNavierPreconditioner constructor
        MonolithicNavierPreconditioner::MonolithicNavierPreconditioner(Array<int> block_offsets_) : block_offsets(block_offsets_)
        {
            // Compute sizes
            usize = block_offsets[1] - block_offsets[0];
            psize = block_offsets[2] - block_offsets[1];

            // Set ownership flags
            own_momentum = false;
            own_schur = false;
        }

        // Implementation of MonolithicNavierPreconditioner destructor
        MonolithicNavierPreconditioner::~MonolithicNavierPreconditioner()
        {
            if (own_momentum)
                delete invC;

            if (own_schur)
                delete invS;
        }

        ////////////////////////////////////////////////////////////////////////////////////////
        ///                       BlockDiagonal Preconditioner                               ///
        ////////////////////////////////////////////////////////////////////////////////////////

        // Implementation of NavierBlockDiagonalPreconditioner constructor
        NavierBlockDiagonalPreconditioner::NavierBlockDiagonalPreconditioner(Array<int> block_offsets_) : MonolithicNavierPreconditioner(block_offsets_)
        {
            invC = new HypreBoomerAMG();
            static_cast<HypreBoomerAMG *>(invC)->SetPrintLevel(0);
            invC->iterative_mode = false;
            own_momentum = true;
        }


        // Implementation of NavierBlockDiagonalPreconditioner RebuildPreconditioner
        void NavierBlockDiagonalPreconditioner::SetOperator(const Operator &op)
        {   
            MonolithicNavierPreconditioner::SetOperator(op);
            invC->SetOperator(nsOp->GetBlock(0, 0));
        }

        // Implementation of NavierBlockDiagonalPreconditioner Mult
        void NavierBlockDiagonalPreconditioner::Mult(const Vector &x, Vector &y) const
        {
            Vector u, p;
            u.MakeRef(y, block_offsets[0], block_offsets[1] - block_offsets[0]);
            p.MakeRef(y, block_offsets[1], block_offsets[2] - block_offsets[1]);

            Vector ru, rp;
            ru.MakeRef(const_cast<Vector &>(x), block_offsets[0], block_offsets[1] - block_offsets[0]);
            rp.MakeRef(const_cast<Vector &>(x), block_offsets[1], block_offsets[2] - block_offsets[1]);

            invC->Mult(ru, u);
            invS->Mult(rp, p);
            p.Neg();
        }


        //////////////////////////////////////////////////////////////////////////////
        ///                   Block Lower Triangular Preconditioner                ///
        //////////////////////////////////////////////////////////////////////////////

        // Implementation of NavierBlockLowerTriangularPreconditioner constructor
        NavierBlockLowerTriangularPreconditioner::NavierBlockLowerTriangularPreconditioner(Array<int> block_offsets_) : MonolithicNavierPreconditioner(block_offsets_)
        {
            // Set the momentum solver
            invC = new HypreBoomerAMG();
            static_cast<HypreBoomerAMG *>(invC)->SetPrintLevel(0);
            invC->iterative_mode = false;
            own_momentum = true;
        }

        // Implementation of NavierBlockLowerTriangularPreconditioner SetOperator
        void NavierBlockLowerTriangularPreconditioner::SetOperator(const Operator &op)
        {
            MonolithicNavierPreconditioner::SetOperator(op);
            invC->SetOperator(nsOp->GetBlock(0, 0));
        }

        // Implementation of NavierBlockLowerTriangularPreconditioner Mult
        void NavierBlockLowerTriangularPreconditioner::Mult(const Vector &x, Vector &y) const
        {
            // Unpack the block vectors
            Vector u, p;
            u.MakeRef(y, block_offsets[0], usize);
            p.MakeRef(y, block_offsets[1], psize);

            Vector ru, rp; // careful as changing ru, rp will change r
            ru.MakeRef(const_cast<Vector &>(x), block_offsets[0], usize);
            rp.MakeRef(const_cast<Vector &>(x), block_offsets[1], psize);

            Vector tmp(psize);
            Vector tmp2(psize);
            
            // Perform the block elimination for the preconditioner
            invC->Mult(ru, u);

            nsOp->GetBlock(1,0).Mult(u, tmp);
            subtract(rp, tmp, tmp2);

            invS->Mult(tmp2, p);
            p.Neg();  
        }        



        /////////////////////////////////////////////////////////////////////////////////////
        ///                      Block Upper Triangular Preconditioner                    ///
        /////////////////////////////////////////////////////////////////////////////////////

        // Implementation of NavierBlockUpperTriangularPreconditioner constructor
        NavierBlockUpperTriangularPreconditioner::NavierBlockUpperTriangularPreconditioner(Array<int> block_offsets_) : MonolithicNavierPreconditioner(block_offsets_)
        {
            // Set the momentum solver
            invC = new HypreBoomerAMG();
            static_cast<HypreBoomerAMG *>(invC)->SetPrintLevel(0);
            invC->iterative_mode = false;
            own_momentum = true;
        }

        // Implementation of NavierBlockUpperTriangularPreconditioner SetOperator
        void NavierBlockUpperTriangularPreconditioner::SetOperator(const Operator &op)
        {
            MonolithicNavierPreconditioner::SetOperator(op);
            invC->SetOperator(nsOp->GetBlock(0, 0));
        }

        // Implementation of NavierBlockUpperTriangularPreconditioner Mult
        void NavierBlockUpperTriangularPreconditioner::Mult(const Vector &x, Vector &y) const
        {
            // Unpack the block vectors
            Vector u, p;
            u.MakeRef(y, block_offsets[0], usize);
            p.MakeRef(y, block_offsets[1], psize);

            Vector ru, rp; // careful as changing ru, rp will change r
            ru.MakeRef(const_cast<Vector &>(x), block_offsets[0], usize);
            rp.MakeRef(const_cast<Vector &>(x), block_offsets[1], psize);

            Vector tmp(usize);
            Vector tmp2(usize);

            // Perform the block elimination for the preconditioner
            invS->Mult(rp, p);
            p.Neg();

            nsOp->GetBlock(0,1).Mult(p, tmp);
            subtract(ru, tmp, tmp2);

            invC->Mult(tmp2, u);
        }


        ////////////////////////////////////////////////////////////////////////////
        ///                     Chorin Temam Preconditioner                      ///
        ////////////////////////////////////////////////////////////////////////////

        // Implementation of ChorinTemamPreconditioner constructor
        ChorinTemamPreconditioner::ChorinTemamPreconditioner(Array<int> block_offsets_) : MonolithicNavierPreconditioner(block_offsets_)
        {
            // Set the momentum solver
            invC = new HypreBoomerAMG();
            static_cast<HypreBoomerAMG *>(invC)->SetPrintLevel(0);
            invC->iterative_mode = false;
            own_momentum = true;

            // Create default H2 solver (can be set externally with SetH2Solver)
            H2 = new HypreSmoother();
            static_cast<HypreSmoother *>(H2)->SetType(HypreSmoother::Jacobi, 1);
            own_H2 = true;

            // Create Lower Block Triangular Preconditioner
            L = new BlockLowerTriangularPreconditioner(block_offsets);

            // Create Upper Block Triangular Preconditioner
            U = new BlockLowerTriangularPreconditioner(block_offsets);            
        }

        // Implementation of ChorinTemamPreconditioner destructor
        ChorinTemamPreconditioner::~ChorinTemamPreconditioner()
        {
            delete H2G;  
            delete H2Gt; // Transpose operator doesn't have ownership of H2G
            delete L;
            delete U;
            if (own_H2)
                delete H2;
        }

        // Implementation of ChorinTemamPreconditioner SetMomentumSolver
        void ChorinTemamPreconditioner::SetMomentumSolver(Solver *invC_, bool own_momentum_)
        {
            MonolithicNavierPreconditioner::SetMomentumSolver(invC_, own_momentum_);
            L->SetDiagonalBlock(0, invC);
        }

        // Implementation of ChorinTemamPreconditioner SetSchurSolver
        void ChorinTemamPreconditioner::SetSchurSolver(Solver *invS_, bool own_schur_)
        {
            MonolithicNavierPreconditioner::SetSchurSolver(invS_, own_schur_);
            L->SetDiagonalBlock(1, invS);
        }

        // Implementation of ChorinTemamPreconditioner SetH2Solver
        void ChorinTemamPreconditioner::SetH2Solver(Solver *H2_, bool own_H2_)
        {
            if (own_H2)
                delete H2;

            if (H2G != nullptr) // Delete the previous H2G, will be recreated in SetOperator
            {
                delete H2G;
                H2G = nullptr;
                delete H2Gt;
                H2Gt = nullptr;
            }

            H2 = H2_;
            own_H2 = own_H2_;
        }

        // Implementation of ChorinTemamPreconditioner SetH2Operator
        void ChorinTemamPreconditioner::SetH2Operator(Operator *H2Op)
        {
            if (H2G != nullptr) // Delete the previous H2G, will be recreated in SetOperator
            {
                delete H2G;
                H2G = nullptr;
                delete H2Gt;
                H2Gt = nullptr;
            }

            H2->SetOperator(*H2Op);
        }

        // Implementation of ChorinTemamPreconditioner SetOperator
        void ChorinTemamPreconditioner::SetOperator(const Operator &op)
        {
            MonolithicNavierPreconditioner::SetOperator(op);

            // Set the momentum operator
            invC->SetOperator(nsOp->GetBlock(0, 0));
            
            // Set the Lower Block Triangular Preconditioner
            L->SetDiagonalBlock(0, invC);
            L->SetBlock(1, 0, &nsOp->GetBlock(1, 0));

            // Set the Upper Block Triangular Preconditioner
            if (!H2G)
            {
                MFEM_ASSERT(H2 != nullptr, "H2 solver is not set! Call ChorinTemamPreconditioner::SetH2Solver() first.");
                H2G = new ProductOperator(H2, &nsOp->GetBlock(0, 1), false, false); // Does not own H2 or G
                H2Gt = new TransposeOperator( *H2G );
            }
            U->SetBlock(1, 0, H2Gt); // We're building the transpose of the upper block triangular
        }

        // Implementation of ChorinTemamPreconditioner Mult
        void ChorinTemamPreconditioner::Mult(const Vector &x, Vector &y) const
        {
            tmp.Update(block_offsets); tmp = 0.0;

            // Unpack the temporary block vector
            Vector upred, ppred;
            upred.MakeRef(tmp, block_offsets[0], usize);
            ppred.MakeRef(tmp, block_offsets[1], psize);
 
            // Perform the block elimination for the preconditioner
            L->Mult(x, tmp);
            ppred.Neg(); // Because we are applying the negative of the Schur complement
            U->MultTranspose(tmp, y);
        }


        /////////////////////////////////////////////////////////////////////////////////////
        ///                        Algebraic Yosida Preconditioner                        ///
        /////////////////////////////////////////////////////////////////////////////////////

        // Implementation of YosidaPreconditioner constructor
        YosidaPreconditioner::YosidaPreconditioner(Array<int> block_offsets_) : MonolithicNavierPreconditioner(block_offsets_)
        {
            // Set the momentum solver
            invC = new HypreBoomerAMG();
            static_cast<HypreBoomerAMG *>(invC)->SetPrintLevel(0);
            invC->iterative_mode = false;
            own_momentum = true;

            // Create default H2 solver (can be set externally with SetH2Solver)
            H2 = new HypreBoomerAMG();
            static_cast<HypreBoomerAMG *>(H2)->SetPrintLevel(0);
            H2->iterative_mode = false;
            own_H2 = true;

            // Create Lower Block Triangular Preconditioner
            L = new BlockLowerTriangularPreconditioner(block_offsets);

            // Create Upper Block Triangular Preconditioner
            U = new BlockLowerTriangularPreconditioner(block_offsets);
        }

        // Implementation of YosidaPreconditioner destructor
        YosidaPreconditioner::~YosidaPreconditioner()
        {
            delete H2G;  
            delete H2Gt; // Transpose operator doesn't have ownership of H2G
            delete L;
            delete U;
            if (own_H2)
                delete H2;
        }

        // Implementation of YosidaPreconditioner SetMomentumSolver
        void YosidaPreconditioner::SetMomentumSolver(Solver *invC_, bool own_momentum_)
        {
            MonolithicNavierPreconditioner::SetMomentumSolver(invC_, own_momentum_);
            L->SetDiagonalBlock(0, invC);
        }

        // Implementation of YosidaPreconditioner SetSchurSolver
        void YosidaPreconditioner::SetSchurSolver(Solver *invS_, bool own_schur_)
        {
            MonolithicNavierPreconditioner::SetSchurSolver(invS_, own_schur_);
            L->SetDiagonalBlock(1, invS);
        }

        // Implementation of YosidaPreconditioner SetH2Solver
        void YosidaPreconditioner::SetH2Solver(Solver *H2_, bool own_H2_)
        {
            if (own_H2)
                delete H2;

            if (H2G != nullptr) // Delete the previous H2G, will be recreated in SetOperator
            {
                delete H2G;
                H2G = nullptr;
                delete H2Gt;
                H2Gt = nullptr;
            }

            H2 = H2_;
            own_H2 = own_H2_;
        }

        // Implementation of YosidaPreconditioner SetH2Operator
        void YosidaPreconditioner::SetH2Operator(Operator *H2Op)
        {
            if (H2G != nullptr) // Delete the previous H2G, will be recreated in SetOperator
            {
                delete H2G;
                H2G = nullptr;
                delete H2Gt;
                H2Gt = nullptr;
            }

            H2->SetOperator(*H2Op);
        }

        // Implementation of YosidaPreconditioner SetOperator
        void YosidaPreconditioner::SetOperator(const Operator &op)
        {
            MonolithicNavierPreconditioner::SetOperator(op);

            // Set the momentum operator
            invC->SetOperator(nsOp->GetBlock(0, 0));
            H2->SetOperator(nsOp->GetBlock(0, 0));
            
            // Set the Lower Block Triangular Preconditioner
            L->SetDiagonalBlock(0, invC);
            L->SetBlock(1, 0, &nsOp->GetBlock(1, 0));

            // Set the Upper Block Triangular Preconditioner
            if (!H2G)
            {
                MFEM_ASSERT(H2 != nullptr, "H2 solver is not set! Call YosidaPreconditioner::SetH2Solver() first.");
                H2G = new ProductOperator(H2, &nsOp->GetBlock(0, 1), false, false); // Does not own H2 or G
                H2Gt = new TransposeOperator( *H2G );
            }
            U->SetBlock(1, 0, H2Gt); // We're building the transpose of the upper block triangular
        }

        // Implementation of YosidaPreconditioner Mult
        void YosidaPreconditioner::Mult(const Vector &x, Vector &y) const
        {
            tmp.Update(block_offsets); tmp = 0.0;

            // Unpack the temporary block vector
            Vector upred, ppred;
            upred.MakeRef(tmp, block_offsets[0], usize);
            ppred.MakeRef(tmp, block_offsets[1], psize);
 
            // Perform the block elimination for the preconditioner
            L->Mult(x, tmp);
            ppred.Neg(); // Because we are applying the negative of the Schur complement
            U->MultTranspose(tmp, y);
        }

        

    } // namespace navier

} // namespace mfem