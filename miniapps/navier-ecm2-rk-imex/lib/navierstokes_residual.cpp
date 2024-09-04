#include "navierstokes_residual.hpp"
#include "navierstokes_operator.hpp"

using namespace mfem;

// Implementation of NavierStokesResidual
NavierStokesResidual::NavierStokesResidual(NavierStokesOperator &nav) : Operator(nav.udim),
                                                                        nav(nav),
                                                                        z(nav.GetUdim()) {}

void NavierStokesResidual::SetTimeStep(const double new_dt)
{
   dt = new_dt;
}

void NavierStokesResidual::SetImplicitCoefficient(const double coeff)
{
   aI_ii = coeff;
}

// Implementation of NavierStokesResidualIMEX
NavierStokesResidualIMEX::NavierStokesResidualIMEX(NavierStokesOperator &nav)
    : NavierStokesResidual(nav)
{
   // Check if the NavierStokesOperator is of the correct type
   MFEM_VERIFY(nav.splitting_type == SplittingType::IMEX,
               "NavierStokesResidualIMEX requires a NavierStokesOperator of type IMEX");
}

void NavierStokesResidualIMEX::Mult(const Vector &xb, Vector &yb) const
{ // Compute y = M x + dt aI_ii K x = M x - aI_ii dt ImplicitMult(x),  y(ess_dofs) = 0
   const BlockVector x(xb.GetData(), nav.offsets);
   BlockVector y(yb.GetData(), nav.offsets);
   Vector &yu = y.GetBlock(0);
   Vector &yp = y.GetBlock(1);

   nav.MassMult(x, y);
   nav.ImplicitMult(x, z);
   add(yu, -dt*aI_ii, z, yu);
   yu.SetSubVector(nav.vel_ess_tdof, 0.0); // Apply bcs
   yp = 0.0;
}

Operator &NavierStokesResidualIMEX::GetGradient(const Vector &xb) const
{
   //Compute dR(x)/dx = M + dt aI_ii K,   modified with bcs
   delete Jacobian;
   Jacobian = Add(1.0, *((nav.M).As<HypreParMatrix>()), dt * aI_ii, *((nav.K).As<HypreParMatrix>()) );
   HypreParMatrix *Je = Jacobian->EliminateRowsCols(nav.vel_ess_tdof); // Apply bcs
   delete Je;
   return *Jacobian;
}

// Implementation of NavierStokesResidualImplicit
NavierStokesResidualImplicit::NavierStokesResidualImplicit(NavierStokesOperator &nav)
    : NavierStokesResidual(nav)
{
   // Check if the NavierStokesOperator is of the correct type
   MFEM_VERIFY(nav.splitting_type == SplittingType::IMPLICIT,
               "NavierStokesResidualImplicit requires a NavierStokesOperator of type IMPLICIT");
}

void NavierStokesResidualImplicit::Mult(const Vector &xb, Vector &yb) const
{ 
   // Compute y = M x + dt aI_ii ( K + C(x) ) x = M x - dt aI_ii ImplicitMult(x), ,  y(ess_dofs) = 0
   const BlockVector x(xb.GetData(), nav.offsets);
   BlockVector y(yb.GetData(), nav.offsets);
   
   Vector &yu = y.GetBlock(0);
   Vector &yp = y.GetBlock(1);

   nav.MassMult(x, y);
   nav.ImplicitMult(x, z);
   add(yu, -dt*aI_ii, z, yu);
   yu.SetSubVector(nav.vel_ess_tdof, 0.0); // Apply bcs
   yp = 0.0;
}

Operator &NavierStokesResidualImplicit::GetGradient(const Vector &xb) const
{
   // Compute dR(x)/dx = M + dt aii K + dt aI_ii C(x)',  modified with bcs
   const BlockVector x(xb.GetData(), nav.offsets);
   const Vector &xu = x.GetBlock(0);
   const Vector &xp = x.GetBlock(1);

   delete Jacobian;
   Jacobian = Add(1.0, *((nav.M).As<HypreParMatrix>()), dt * aI_ii, *((nav.K).As<HypreParMatrix>()) );
   auto *grad_C = dynamic_cast<const HypreParMatrix *>(&nav.NL_form->GetGradient(xu));
   Jacobian->Add(dt * aI_ii, *grad_C);
   HypreParMatrix *Je = Jacobian->EliminateRowsCols(nav.vel_ess_tdof); // Apply bcs
   delete Je;
   return *Jacobian;
}
