#include "pressure_correction.hpp"

namespace mfem
{

    namespace navier
    {

        ///////////////////////////////////////////////////////////////////////
        ///                   Pressure Correction Solver                    ///
        ///////////////////////////////////////////////////////////////////////

        PressureCorrectionSolver::PressureCorrectionSolver(Array<int> &block_offsets_)
            : block_offsets(block_offsets_),
              Solver(block_offsets_[2] - block_offsets_[1])
        {
            // Compute sizes
            usize = block_offsets[1] - block_offsets[0];
            psize = block_offsets[2] - block_offsets[1];

            // Initialize Vectors
            tmp1.SetSize(usize);
            tmp1 = 0.0;
            tmp2.SetSize(usize);
            tmp2 = 0.0;
            rhs.SetSize(psize);
            rhs = 0.0;

            // Create default H1 solver (can be set externally with SetH1Solver)
            H1 = new HypreSmoother();
            static_cast<HypreSmoother *>(H1)->SetType(HypreSmoother::Jacobi, 1);
            H1->iterative_mode = false;
            own_H1 = true;
        }

        // Implementation of PressureCorrectionSolver destructor
        PressureCorrectionSolver::~PressureCorrectionSolver()
        {
            if (own_H1)
                delete H1;

            if (own_schur)
                delete invS;
        }

        // Implementation of PressureCorrectionSolver SetOperator
        void PressureCorrectionSolver::SetOperator(const Operator &op)
        {
            nsOp = (BlockOperator *)&op;
        }

        // Implementation of PressureCorrectionSolver SetSchurSolver
        void PressureCorrectionSolver::SetSchurSolver(Solver *invS_, bool own_schur_)
        {
            if (own_schur)
                delete invS;

            invS = invS_;
            own_schur = own_schur_;
        }

        // Implementation of PressureCorrectionSolver SetH1Solver
        void PressureCorrectionSolver::SetH1Solver(Solver *H1_, bool own_H1_)
        {
            if (own_H1)
                delete H1;

            H1 = H1_;
            own_H1 = own_H1_;
        }

        // Implementation of PressureCorrectionSolver SetH1Operator
        void PressureCorrectionSolver::SetH1Operator(Operator *H1Op)
        {
            H1->SetOperator(*H1Op);
        }

        // Implementation of PressureCorrectionSolver Mult
        void PressureCorrectionSolver::Mult(const Vector &x, Vector &y) const
        {
            const Operator &C = nsOp->GetBlock(0, 0);
            const Operator &D = nsOp->GetBlock(1, 0);
            const Operator &G = nsOp->GetBlock(0, 1);

            // Compute the rhs
            G.Mult(x, tmp1);         //         G x = tmp1
            H1->Mult(tmp1, tmp2);    //       H G x = tmp2
            C.Mult(tmp2, tmp1);      //     C H G x = tmp1
            H1->Mult(tmp1, tmp2);    //   H C H G x = tmp2
            D.Mult(tmp2, rhs);      // D H C H G x = rhs

            // Solve the Schur complement
            invS->Mult(rhs, y);
        }

    } // namespace navier

} // namespace mfem