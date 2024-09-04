#ifndef MFEM_NAVIER_UNSTEADY_HPP
#define MFEM_NAVIER_UNSTEADY_HPP

#define MFEM_NAVIER_SRK_VERSION 0.1

#include <mfem.hpp>
#include "navierstokes_operator.hpp"

namespace mfem
{

   /**
    * @brief Segregated Runge-Kutta solver for the incompressible Navier-Stokes equations.
    *
    * This class implements the Segregated Runge-Kutta (SRK) solver, specifically designed
    * for the incompressible Navier-Stokes equations. The solver supports both implicit-explicit
    * (IMEX) and semi-implicit formulations.
    *
    * Refs:
    * [1] Colomés, Oriol, and Santiago Badia. "Segregated Runge–Kutta methods for the incompressible Navier–Stokes equations."
    *     International Journal for Numerical Methods in Engineering 105.5 (2016): 372-400.
    */
   class NavierStokesSRKSolver
   {
   public:
      /**
       * @brief Constructor for the NavierStokesSRKSolver class.
       *
       * @param op NavierStokesOperator for incompressible Navier-Stokes equations semi-discretized in space.
       * @param method Integer specifying the SRK method to use (1 for IMEX, 2 for semi-implicit).
       */
      NavierStokesSRKSolver(NavierStokesOperator *op, int &method, bool own = false);

      ~NavierStokesSRKSolver()
      {
         if (own)
         {
            delete op;
         }
      }

      /**
       * @brief Time-stepping method for the NavierStokesSRKSolver class.
       *
       * This method performs a single time step using the Segregated Runge-Kutta solver.
       * Type of time advancing depends on the method selected.
       *
       * @param X Vector representing the solution at the current time step.
       * @param t Current time.
       * @param dt Time step size.
       */
      void Step(Vector &xb, double &t, double &dt);

      // Getters
      NavierStokesOperator *GetOperator() const { return op; }
      void GetMethod() const;
      int GetOrder() const { return rk_order; }

   private:
      // Implementation of the different SRK methods
      void Step_FEBE(Vector &xb, double &t, double &dt);
      void Step_IMEX_Midpoint(Vector &xb, double &t, double &dt);
      void Step_DIRK_2_3_2(Vector &xb, double &t, double &dt);
      void Step_DIRK_2_2_2(Vector &xb, double &t, double &dt);
      void Step_DIRK_4_4_3(Vector &xb, double &t, double &dt);

      NavierStokesOperator *op = nullptr;   ///< Pointer to the NavierStokesOperator.
      int rk_order;                         ///< Order of the Runge-Kutta method. (useful for extrapolation)
      bool own = false;                     ///< Boolean determining if SRK solver owns NavierStokesOperator.
      int &method;                          ///< selected SRK method identifier.
      mutable BlockVector b, z, w;          ///< Temporary vector for solution and rhs.
   };

} // namespace mfem

#endif