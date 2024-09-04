#include <mfem.hpp>
#include "navierstokes_srk.hpp"

namespace mfem
{

   NavierStokesSRKSolver::NavierStokesSRKSolver(NavierStokesOperator *op, int &method, bool own) : op(op),
                                                                                   method(method),
                                                                                   own(own),
                                                                                   b(op->GetOffsets()),
                                                                                   z(op->GetOffsets()),
                                                                                   w(op->GetOffsets())
   {
      // Parse the corresponding RK order based on the method
      if(method == 0) // First-order methods
      {
         rk_order = 1;
      }
      else if ( method == 1 || method == 2 || method == 3 ) // Second-order methods
      {
         rk_order = 2;
      }
      else if (method == 4) // Third-order methods
      {
         rk_order = 3;
      } 

      // Set the Runge Kutta order in the NavierStokesOperator (for extrapolation)
      op->SetOrder(rk_order);
   }

   void NavierStokesSRKSolver::GetMethod() const
   {
      // Based on the method, print the corresponding SRK method
      switch (method)
      {
      case 0:
         std::cout << "Forward-Backward Euler" << std::endl;
         break;
      case 1:
         std::cout << "Implicit-Explicit Midpoint" << std::endl;
         break;   
      case 2:
         std::cout << "Two-stage, second-order, L-stable DIRK (v1)" << std::endl;
         break;
      case 3:
         std::cout << "Two-stage, second-order, L-stable DIRK (v2)" << std::endl;
         break;
      case 4:
         std::cout << "Four-stage, third-order, L-stable DIRK" << std::endl;
         break;
      default:
         std::cout << "Selected RK scheme is not implemented yet, choose one of the following: \n"
                   << "1- Forward-Backward Euler \n"
                   << "2- Implicit-Explicit Midpoint \n"
                   << "3- Two-stage, second-order, L-stable DIRK (v1) \n"
                   << "4- Two-stage, second-order, L-stable DIRK (v2) \n"
                   << "5- Four-stage, third-order, L-stable DIRK" << std::endl;
         break;
      }
   }

   void NavierStokesSRKSolver::Step(Vector &xb, double &t, double &dt)
   {
      // Set timestep for the NavierStokesOperator
      op->SetTimeStep(dt);
      op->SetTime(t, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1); // Setting  EvalMode to ADDITIVE_TERM_1 (i.e. Explicit) enables assembling forcing term fu_rhs

      // Call the appropriate SRK solver based on method
      switch (method)
      {
      case 0:
         Step_FEBE(xb, t, dt); // 1 stage, first-order
         break;
      case 1:
         Step_IMEX_Midpoint(xb, t, dt); // 1 stage, second-order
         break;
      case 2:
         Step_DIRK_2_3_2(xb, t, dt); // 2 stages, second-order (v1)
         break;
      case 3:
         Step_DIRK_2_2_2(xb, t, dt); // 2 stages, second-order (v2)
         break;
      case 4:
         Step_DIRK_4_4_3(xb, t, dt); // 4 stages, third-order
         break;
      default:
         mfem_error("Selected RK scheme is not implemented yet, choose one of the following: \n"
                    "1- Forward-Backward Euler \n"
                    "2- Implicit-Explicit Midpoint \n"
                    "3- Two-stage, second-order, L-stable DIRK (v1) \n"
                    "4- Two-stage, second-order, L-stable DIRK (v2) \n"
                    "5- Four-stage, third-order, L-stable DIRK");
         break;
      }

      // Advance time   TODO: This should check if accept solution or not if we use adaptive timestepping
      t += dt;
   }

   void NavierStokesSRKSolver::Step_FEBE(Vector &xb, double &t, double &dt)
   {
      // Solve SRK step using Forward-Backward Euler, corresponding to the following Butcher's tableaus:
      //
      // Implicit:  0 | 0 0        Explicit:  0 | 0 0
      //            1 | 0 1                   1 | 1 0
      //            --------                  --------
      //              | 0 1                     | 0 1

      BlockVector y(xb.GetData(), op->GetOffsets());

      // RK Coefficients definitions
      const double aI_22 = 1.0;
      const double cI_2 = 1.0;
      const double bI_2 = 1.0;

      const double aE_21 = 1.0;
      const double cE_2 = 1.0;
      const double bE_2 = 1.0;

      // Temporary vectors for residuals and solution
      BlockVector FE1(op->GetOffsets());
      BlockVector FE2(op->GetOffsets());
      BlockVector FI2(op->GetOffsets());
      BlockVector Y1(op->GetOffsets());
      BlockVector Y2(op->GetOffsets());

      // Stage 1
      {
         Y1 = y;                // U1 = Un
         op->SolvePressure(Y1); // solve for P1

         // Residuals
         op->SetTime(t, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y1, FE1); // FE1 = f(t, Y1)
      }

      // Stage 2
      {
         // Compute rhs
         op->MassMult(xb, b); // b = M * Un
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_21;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_21 * FE1

         // Implicit solve
         op->SetImplicitCoefficient(aI_22);
         Y2 = Y1; // initial guess
         op->ImplicitSolve(b, Y2);

         // Solve for pressure
         op->SolvePressure(Y2);

         // Residuals
         op->SetTime(t + cE_2 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y2, FE2); // FE2 = f(t + cE_2 * dt, Y2)
         op->SetTime(t + cI_2 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_2);
         op->ImplicitMult(Y2, FI2); // FI2 = g(t + cI_2 * dt, Y2)
      }

      // Final update velocity
      for (int i = 0; i < y.BlockSize(0); i++) // TODO: do we need mass matrix inversion?
      {
         y[i] += dt * (bE_2 * FE2[i] + bE_2 * FI2[i]);
      }

      op->SolvePressure(y);
   }

   void NavierStokesSRKSolver::Step_IMEX_Midpoint(Vector &xb, double &t, double &dt)
   {
      // Solve SRK step using Implicit-Midpoint IMEX, corresponding to the following Butcher's tableaus:
      //
      // Implicit:  0   | 0  0        Explicit:  0   | 0   0
      //            1/2 | 0 1/2                  1/2 | 1/2 0
      //            --------------               --------------
      //                | 0  1                       | 0   1

      BlockVector y(xb.GetData(), op->GetOffsets());

      // RK Coefficients definitions
      const double aI_22 = 1.0 / 2.0;
      const double cI_2 = 1.0 / 2.0;
      const double bI_2 = 1.0;

      const double aE_21 = 1.0 / 2.0;
      const double cE_2 = 1.0 / 2.0;
      const double bE_2 = 1.0;

      // Temporary vectors for residuals and solution
      BlockVector FE1(op->GetOffsets());
      BlockVector FE2(op->GetOffsets());
      BlockVector FI2(op->GetOffsets());
      BlockVector Y1(op->GetOffsets());
      BlockVector Y2(op->GetOffsets());

      // Stage 1
      {
         Y1 = y;                // U1 = Un
         op->SolvePressure(Y1); // solve for P1

         // Residuals
         op->SetTime(t, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y1, FE1); // FE1 = f(t, Y1)
      }

      // Stage 2
      {
         // Compute rhs
         op->MassMult(xb, b); // b = M * Un
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_21;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_21 * FE1

         // Implicit solve
         op->SetImplicitCoefficient(aI_22);
         Y2 = Y1; // initial guess
         op->ImplicitSolve(b, Y2);

         // Solve for pressure
         op->SolvePressure(Y2);

         // Residuals
         op->SetTime(t + cE_2 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y2, FE2); // FE2 = f(t + cE_2 * dt, Y2)
         op->SetTime(t + cI_2 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_2);
         op->ImplicitMult(Y2, FI2); // FI2 = g(t + cI_2 * dt, Y2)
      }

      // Final update velocity
      for (int i = 0; i < y.BlockSize(0); i++) // TODO: do we need mass matrix inversion?
      {
         y[i] += dt * (bE_2 * FE2[i] + bE_2 * FI2[i]);
      }

      op->SolvePressure(y);
   }

   void NavierStokesSRKSolver::Step_DIRK_2_3_2(Vector &xb, double &t, double &dt)
   {
      // Solve SRK step using L-stable, two-stage, second-order DIRK, corresponding to the following Butcher's tableaus:
      //
      // Implicit:  0    | 0      0          0          Explicit:  0    | 0          0          0
      //          &gamma | 0    &gamma       0                   &gamma | &gamma     0          0
      //            1    | 0   1- &gamma   &gamma                  1    | &delta  1- &delta     0
      //          --------------------------------               -------------------------------------
      //                 | 0   1- &gamma   &gamma                       | 0       1- &gamma    &gamma
      //
      // where &gamma = (2 - sqrt(2))/2, &delta = - 2 sqrt(2) / 3
      //

      BlockVector y(xb.GetData(), op->GetOffsets());

      // RK Coefficients definitions
      const double gamma = (2.0 - sqrt(2.0)) / 2.0;
      const double delta = -2.0 * sqrt(2.0) / 3.0;

      const double aE_21 = 1.0;
      const double aE_31 = delta;
      const double aE_32 = 1.0 - delta;
      const double cE_2 = 1.0;
      const double cE_3 = 1.0;
      const double bE_2 = 1.0 - gamma;
      const double bE_3 = gamma;

      const double aI_22 = gamma;
      const double aI_32 = 1.0 - gamma;
      const double aI_33 = gamma;
      const double cI_2 = gamma;
      const double cI_3 = 1.0;
      const double bI_2 = 1.0 - gamma;
      const double bI_3 = gamma;

      // Temporary vectors for residuals and solution
      BlockVector FE1(op->GetOffsets());
      BlockVector FE2(op->GetOffsets());
      BlockVector FE3(op->GetOffsets());
      BlockVector FI2(op->GetOffsets());
      BlockVector FI3(op->GetOffsets());
      BlockVector Y1(op->GetOffsets());
      BlockVector Y2(op->GetOffsets());
      BlockVector Y3(op->GetOffsets());

      // Stage 1
      {
         Y1 = y;                // U1 = Un
         op->SolvePressure(Y1); // solve for P1

         // Residuals
         op->SetTime(t, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y1, FE1); // FE1 = f(t, Y1)
      }

      // Stage 2
      {
         // Compute rhs
         op->MassMult(xb, b); // b = M * Un
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_21;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_21 * FE1

         // Implicit solve
         op->SetImplicitCoefficient(aI_22);
         Y2 = Y1; // initial guess
         op->ImplicitSolve(b, Y2);

         // Solve for pressure
         op->SolvePressure(Y2);

         // Residuals
         op->SetTime(t + cE_2 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y2, FE2); // FE2 = f(t + cE_2 * dt, Y2)
         op->SetTime(t + cI_2 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_2);
         op->ImplicitMult(Y2, FI2); // FI2 = g(t + cI_2 * dt, Y2)
      }

      // Stage 3
      {
         // Compute rhs
         op->MassMult(xb, b); // b = M * Un
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_31;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_31 * FE1

         w.GetBlock(0) = FE2.GetBlock(0);
         w.GetBlock(0) *= dt * aE_32;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_31 * FE1 + dt * aE_32 * FE2

         w.GetBlock(0) = FI2.GetBlock(0);
         w.GetBlock(0) *= dt * aI_32;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_31 * FE1 + dt * aE_32 * FE2 + dt * aI_32 * FI2

         // Implicit solve
         op->SetImplicitCoefficient(aI_33);
         Y3 = Y2; // initial guess
         op->ImplicitSolve(b, Y3);

         // Solve for pressure
         op->SolvePressure(Y3);

         // Residuals
         op->SetTime(t + cE_3 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y3, FE3); // FE3 = f(t + cE_3 * dt, Y3)
         op->SetTime(t + cI_3 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_2);
         op->ImplicitMult(Y3, FI3); // FI3 = g(t + cI_3 * dt, Y3)
      }

      // Final update velocity
      for (int i = 0; i < y.BlockSize(0); i++) // TODO: do we need mass matrix inversion?
      {
         y[i] += dt * (bE_2 * FE2[i] + bE_3 * FE3[i] + bI_2 * FI2[i] + +bI_2 * FI3[i]);
      }

      op->SolvePressure(y);
   }

   void NavierStokesSRKSolver::Step_DIRK_2_2_2(Vector &xb, double &t, double &dt)
   {
      // Solve SRK step using L-stable, two-stage, second-order DIRK, corresponding to the following Butcher's tableaus:
      //
      // Implicit:  0    | 0      0          0          Explicit:  0    | 0          0          0
      //          &gamma | 0    &gamma       0                   &gamma | &gamma     0          0
      //            1    | 0   1- &gamma   &gamma                  1    | &delta  1- &delta     0
      //          --------------------------------              ----------------------------------
      //                 | 0   1- &gamma   &gamma                       | &delta  1- &delta     0
      //
      // where &gamma = (2 - sqrt(2))/2, &delta = 1 - 1/(2 &gamma)

      BlockVector y(xb.GetData(), op->GetOffsets());

      // RK Coefficients definitions
      const double gamma = (2.0 - sqrt(2.0)) / 2.0;
      const double delta = 1.0 - 1.0 / (2.0 * gamma);

      const double aE_21 = 1.0;
      const double aE_31 = delta;
      const double aE_32 = 1.0 - delta;
      const double cE_2 = 1.0;
      const double cE_3 = 1.0;
      const double bE_1 = delta;
      const double bE_2 = 1.0 - delta;

      const double aI_22 = gamma;
      const double aI_32 = 1.0 - gamma;
      const double aI_33 = gamma;
      const double cI_2 = gamma;
      const double cI_3 = 1.0;
      const double bI_2 = 1.0 - gamma;
      const double bI_3 = gamma;

      // Temporary vectors for residuals and solution
      BlockVector FE1(op->GetOffsets());
      BlockVector FE2(op->GetOffsets());
      BlockVector FI2(op->GetOffsets());
      BlockVector Y1(op->GetOffsets());
      BlockVector Y2(op->GetOffsets());
      BlockVector Y3(op->GetOffsets());

      // Stage 1
      {
         Y1 = y;                // U1 = Un
         op->SolvePressure(Y1); // solve for P1

         // Residuals
         op->SetTime(t, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y1, FE1); // FE1 = f(t, Y1)
      }

      // Stage 2
      {
         // Compute rhs
         op->MassMult(xb, b); // b = M * Un
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_21;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_21 * FE1

         // Implicit solve
         op->SetImplicitCoefficient(aI_22);
         Y2 = Y1; // initial guess
         op->ImplicitSolve(b, Y2);

         // Solve for pressure
         op->SolvePressure(Y2);

         // Residuals
         op->SetTime(t + cE_2 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y2, FE2); // FE2 = f(t + cE_2 * dt, Y2)
         op->SetTime(t + cI_2 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_2);
         op->ImplicitMult(Y2, FI2); // FI2 = g(t + cI_2 * dt, Y2)
      }

      // Stage 3
      {
         // Compute rhs
         op->MassMult(xb, b); // b = M * Un
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_31;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_31 * FE1

         w.GetBlock(0) = FE2.GetBlock(0);
         w.GetBlock(0) *= dt * aE_32;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_31 * FE1 + dt * aE_32 * FE2

         w.GetBlock(0) = FI2.GetBlock(0);
         w.GetBlock(0) *= dt * aI_32;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_31 * FE1 + dt * aE_32 * FE2 + dt * aI_32 * FI2

         // Implicit solve
         op->SetImplicitCoefficient(aI_33);
         Y3 = Y2; // initial guess
         op->ImplicitSolve(b, Y3);

         // Solve for pressure
         op->SolvePressure(Y3);
      }

      // Final update velocity
      y = Y3;               // Since bE_j = aE_s,j for j=1,2,...,s, we can directly update the solution to the last stage
      op->SolvePressure(y); // TODO: check if we need to solve for pressure here
   }

   void NavierStokesSRKSolver::Step_DIRK_4_4_3(Vector &xb, double &t, double &dt)
   {
      // Solve SRK step four-stage, third-order, L-stable DIRK, coupled with a four-stage ERK, corresponding to the following Butcher's tableaus:
      //
      // Implicit:  0   | 0    0    0    0    0      Explicit:  0   | 0      0    0    0    0
      //            1/2 | 0   1/2   0    0    0                 1/2 | 1/2    0    0    0    0
      //            2/3 | 0   1/6  1/2   0    0                 2/3 | 11/8  1/18  0    0    0
      //            1/2 | 0  -1/2  1/2  1/2   0                 1/2 | 5/6  -5/6  1/2  1/2   0
      //            1   | 0   3/2 -3/2  1/2  1/2                1   | 1/4   7/4  3/4 -7/4   0
      //            ----------------------------               ----------------------------
      //                | 0   3/2 -3/2  1/2  1/2                    | 1/4   7/4  3/4 -7/4   0

      BlockVector y(xb.GetData(), op->GetOffsets());

      // RK Coefficients definitions
      const double aI_22 = 1.0 / 2.0;
      const double aI_32 = 1.0 / 6.0;
      const double aI_33 = 1.0 / 2.0;
      const double aI_42 = -1.0 / 2.0;
      const double aI_43 = 1.0 / 2.0;
      const double aI_44 = 1.0 / 2.0;
      const double aI_52 = 3.0 / 2.0;
      const double aI_53 = -3.0 / 2.0;
      const double aI_54 = 1.0 / 2.0;
      const double aI_55 = 1.0 / 2.0;
      const double cI_2 = 1.0 / 2.0;
      const double cI_3 = 2.0 / 3.0;
      const double cI_4 = 1.0 / 2.0;
      const double cI_5 = 1.0;
      const double bI_2 = 3.0 / 2.0;
      const double bI_3 = -3.0 / 2.0;
      const double bI_4 = 1.0 / 2.0;
      const double bI_5 = 1.0 / 2.0;

      const double aE_21 = 1.0 / 2.0;
      const double aE_31 = 11.0 / 8.0;
      const double aE_32 = 1.0 / 18.0;
      const double aE_41 = 5.0 / 6.0;
      const double aE_42 = -5.0 / 6.0;
      const double aE_43 = 1.0 / 2.0;
      const double aE_51 = 1.0 / 4.0;
      const double aE_52 = 7.0 / 4.0;
      const double aE_53 = 3.0 / 4.0;
      const double aE_54 = -7.0 / 4.0;
      const double cE_2 = 1.0 / 2.0;
      const double cE_3 = 2.0 / 3.0;
      const double cE_4 = 1.0 / 2.0;
      const double cE_5 = 1.0;
      const double bE_1 = 1.0 / 4.0;
      const double bE_2 = 7.0 / 4.0;
      const double bE_3 = 3.0 / 4.0;
      const double bE_4 = -7.0 / 4.0;

      // Temporary vectors for residuals and solution
      BlockVector FE1(op->GetOffsets());
      BlockVector FE2(op->GetOffsets());
      BlockVector FE3(op->GetOffsets());
      BlockVector FE4(op->GetOffsets());
      BlockVector FE5(op->GetOffsets());
      BlockVector FI2(op->GetOffsets());
      BlockVector FI3(op->GetOffsets());
      BlockVector FI4(op->GetOffsets());
      BlockVector FI5(op->GetOffsets());
      BlockVector Y1(op->GetOffsets());
      BlockVector Y2(op->GetOffsets());
      BlockVector Y3(op->GetOffsets());
      BlockVector Y4(op->GetOffsets());
      BlockVector Y5(op->GetOffsets());

      // Stage 1
      {
         Y1 = y;                // U1 = Un
         op->SolvePressure(Y1); // solve for P1

         // Residuals
         op->SetTime(t, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y1, FE1); // FE1 = f(t, Y1)
      }

      // Stage 2
      {
         // Compute rhs
         op->MassMult(xb, b); // b = M * Un
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_21;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_21 * FE1

         // Implicit solve
         op->SetImplicitCoefficient(aI_22);
         Y2 = Y1; // initial guess
         op->ImplicitSolve(b, Y2);

         // Solve for pressure
         op->SolvePressure(Y2);

         // Residuals
         op->SetTime(t + cE_2 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_1);
         op->ExplicitMult(Y2, FE2); // FE2 = f(t + cE_2 * dt, Y2)
         op->SetTime(t + cI_2 * dt, TimeDependentOperator::EvalMode::ADDITIVE_TERM_2);
         op->ImplicitMult(Y2, FI2); // FI2 = g(t + cI_2 * dt, Y2)
      }

      // Stage 3
      {
         // Compute rhs
         op->MassMult(xb, b); // b = M * Un
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_31;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_31 * FE1

         w.GetBlock(0) = FE2.GetBlock(0);
         w.GetBlock(0) *= dt * aE_32;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_31 * FE1 + dt * aE_32 * FE2

         w.GetBlock(0) = FI2.GetBlock(0);
         w.GetBlock(0) *= dt * aI_32;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_31 * FE1 + dt * aE_32 * FE2 + dt * aI_32 * FI2

         // Implicit solve
         op->SetImplicitCoefficient(aI_33);
         Y3 = Y2; // initial guess
         op->ImplicitSolve(b, Y3);

         // Solve for pressure
         op->SolvePressure(Y3);
      }

      // Stage 4
      {
         // Compute rhs
         op->MassMult(xb, b); // b = M * Un
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_41;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_41 * FE1

         w.GetBlock(0) = FE2.GetBlock(0);
         w.GetBlock(0) *= dt * aE_42;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_41 * FE1 + dt * aE_42 * FE2

         w.GetBlock(0) = FE3.GetBlock(0);
         w.GetBlock(0) *= dt * aE_43;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_41 * FE1 + dt * aE_42 * FE2 + dt * aE_43 * FE3

         w.GetBlock(0) = FI2.GetBlock(0);
         w.GetBlock(0) *= dt * aI_42;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_41 * FE1 + dt * aE_42 * FE2 + dt * aE_43 * FE3 + dt * aI_42 * FI2

         w.GetBlock(0) = FI3.GetBlock(0);
         w.GetBlock(0) *= dt * aI_43;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_41 * FE1 + dt * aE_42 * FE2 + dt * aE_43 * FE3 + dt * aI_42 * FI2 + dt * aI_43 * FI3

         // Implicit solve
         op->SetImplicitCoefficient(aI_44);
         Y4 = Y3; // initial guess
         op->ImplicitSolve(b, Y4);

         // Solve for pressure
         op->SolvePressure(Y4);
      }

      // Stage 5
      {
         // Compute rhs
         op->MassMult(xb, b); // b = M * Un
         w.GetBlock(0) = FE1.GetBlock(0);
         w.GetBlock(0) *= dt * aE_51;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_51 * FE1

         w.GetBlock(0) = FE2.GetBlock(0);
         w.GetBlock(0) *= dt * aE_52;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_51 * FE1 + dt * aE_52 * FE2

         w.GetBlock(0) = FE3.GetBlock(0);
         w.GetBlock(0) *= dt * aE_53;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_51 * FE1 + dt * aE_52 * FE2 + dt * aE_53 * FE3

         w.GetBlock(0) = FE4.GetBlock(0);
         w.GetBlock(0) *= dt * aE_54;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_51 * FE1 + dt * aE_52 * FE2 + dt * aE_53 * FE3 + dt * aE_54 * FE4

         w.GetBlock(0) = FI2.GetBlock(0);
         w.GetBlock(0) *= dt * aI_52;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_51 * FE1 + dt * aE_52 * FE2 + dt * aE_53 * FE3 + dt * aE_54 * FE4 + dt * aI_52 * FI2

         w.GetBlock(0) = FI3.GetBlock(0);
         w.GetBlock(0) *= dt * aI_53;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_51 * FE1 + dt * aE_52 * FE2 + dt * aE_
         // 53 * FE3 + dt * aE_54 * FE4 + dt * aI_52 * FI2 + dt * aI_53 * FI3

         w.GetBlock(0) = FI4.GetBlock(0);
         w.GetBlock(0) *= dt * aI_54;
         b.GetBlock(0) += w.GetBlock(0); // b = M Un + dt * aE_51 * FE1 + dt * aE_52 * FE2 + dt * aE_53 * FE3 + dt * aE_54 * FE4 + dt * aI_52 * FI2 + dt * aI_53 * FI3 + dt * aI_54 * FI4

         // Implicit solve
         op->SetImplicitCoefficient(aI_55);
         Y5 = Y4; // initial guess
         op->ImplicitSolve(b, Y5);

         // Solve for pressure
         op->SolvePressure(Y5);
      }

      // Final update velocity
      y = Y5;               // Since bE_j = aE_s,j for j=1,2,...,s, we can directly update the solution to the last stage
      op->SolvePressure(y); // TODO: check if we need to solve for pressure here
   }

} // namespace mfem
