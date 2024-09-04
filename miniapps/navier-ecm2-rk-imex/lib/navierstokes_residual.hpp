#pragma once

#include "utils.hpp"
#include <mfem.hpp>

namespace mfem
{

class NavierStokesOperator;

class NavierStokesResidual : public Operator
{

public:
   NavierStokesResidual(NavierStokesOperator &nav);

   virtual ~NavierStokesResidual() { delete Jacobian; Jacobian = nullptr;}

   /// Compute Residual y = R(x)
   void Mult(const Vector &x, Vector &y) const override = 0;

   /// Return an Operator for the linearization dR(x)/dx.
   Operator &GetGradient(const Vector &x) const override = 0;

   // Set the time step for the Runge-Kutta time integrator
   void SetTimeStep(const double new_dt);

   // Set the implicit coefficient for the Runge-Kutta time integrator
   void SetImplicitCoefficient(const double coeff);

   NavierStokesOperator &nav;
   //mutable std::shared_ptr<FDJacobian> fd_linearized;
   mutable Vector z, w, Hdot;
   mutable HypreParMatrix *Jacobian = nullptr;
   double dt;
   double aI_ii;
};

// Class specifying Residual/Gradient for SRK-IMEX solver
class NavierStokesResidualIMEX : public NavierStokesResidual
{

   //friend class NavierStokesResidualIMEX;

public:
   NavierStokesResidualIMEX(NavierStokesOperator &nav);

   void Mult(const Vector &x, Vector &y) const override;

   Operator &GetGradient(const Vector &x) const override;
};


// Class specifying Residual/Gradient for SRK-Implicit solver
class NavierStokesResidualImplicit : public NavierStokesResidual
{

   //friend class NavierStokesResidualImplicit;

public:
   NavierStokesResidualImplicit(NavierStokesOperator &nav);

   void Mult(const Vector &x, Vector &y) const override;

   Operator &GetGradient(const Vector &x) const override;
};

}