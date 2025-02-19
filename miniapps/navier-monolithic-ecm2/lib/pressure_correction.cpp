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
            D.Mult(tmp2, rhs);       // D H C H G x = rhs

            // Solve the Schur complement
            invS->Mult(rhs, y);
        }


        /////////////////////////////////////////////////////////////////////////////
        ///                 High Order Pressure Correction Solver                 ///
        /////////////////////////////////////////////////////////////////////////////

        HighOrderPressureCorrectionSolver::HighOrderPressureCorrectionSolver(Array<int> &block_offsets_, int q_order_)
            : PressureCorrectionSolver(block_offsets_),
              q_order(q_order_)
        {
            // Initialize Vectors
            z.SetSize(psize); z = 0.0;
            zq.SetSize(psize); zq = 0.0;

            // Initialize structure for the high order pressure correction: q x q matrix of Vector of size usize
            int size = q_order * q_order;
            ZData.resize(size);
            for (int i = 0; i < size; i++)
            {
                ZData[i].SetSize(usize);
                ZData[i] = 0.0;
            }
        }


        // Implementation of HighOrderPressureCorrectionSolver SetOrder
        void HighOrderPressureCorrectionSolver::SetOrder(int q_order_)
        {
            MFEM_VERIFY(q_order_ >= 1, "Pressure Correction order must be greater than 0.");

            q_order = q_order_;

            // Resize ZData
            int size = q_order * q_order;
            ZData.resize(size);
            for (int i = 0; i < size; i++)
            {
                ZData[i].SetSize(usize);
                ZData[i] = 0.0;
            }
        }

        // Implementation of HighOrderPressureCorrectionSolver Mult
        void HighOrderPressureCorrectionSolver::Mult(const Vector &x, Vector &y) const
        { 
            const Operator &D = nsOp->GetBlock(1, 0);
            const Operator &G = nsOp->GetBlock(0, 1);

            // Reshape data to easily access 
            Vector* ZData_ = ZData.data();
            auto Z3 = Reshape(ZData_, q_order, q_order);

            // Initialize solution with z0 = x
            z = x;  // z0 = x
            y = z;  // y = z0

            // High order pressure correction
            for (int i = 0; i < q_order; i++)
            {
                // 1. Compute rhs

                // Z3[i][0] = HAHG zi
                G.Mult(z, tmp1);           //         G z = tmp1
                H1->Mult(tmp1, tmp2);      //       H G x = tmp2
                opA->Mult(tmp2, tmp1);     //     A H G x = tmp1
                H1->Mult(tmp1, Z3(0,i));   //   H A H G x = tmp2
                // rhs = D Z3[0][i]
                D.Mult(tmp2, rhs);         // D H A H G z = rhs

                for (int j = 1; j <= i; j++)
                {
                    // Z3[i][j] = -(HA) Z3[i][j-1]
                    opA->Mult(Z3(j-1,i-j), tmp1);    //      A Z3[j-1][i-j] = tmp1
                    H1->Mult(tmp1, tmp2);            //    H A Z3[j-1][i-j] = tmp2
                    tmp2.Neg();                      // - (H A) Z3[j-1][i-j])
                    Z3(j,i-j) = tmp2;
                    D.AddMult(Z3(j,i-j), rhs, 1.0);  // rhs += D Z3[j][i-j]
                }

                // 2. Solve the Schur complement
                invS->Mult(rhs, z);

                // 3. Update final pressure with correction
                y.Add(1.0, z); // y += z
            }

            // Save the latest pressure correction
            zq = z;
        }



    } // namespace navier

} // namespace mfem