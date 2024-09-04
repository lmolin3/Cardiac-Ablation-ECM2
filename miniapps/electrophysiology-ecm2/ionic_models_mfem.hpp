#include "mfem.hpp"

#ifndef MFEM_ECM2_IONIC_HPP
#define MFEM_ECM2_IONIC_HPP

namespace mfem
{

/** Discretized system of ODEs for Fenton Karma:
 *     du/dt = F(u,h)
 *             { (1-h)/tau_open    u < u_gate    
 *     dh/dt = {
 *             { -h/tau_close      u >= u_gate 
 * 
 *     F(u,h) = 1/tau_in h u^2(1-u) - u/tau_out
 * 
 *  where u is the vector representing the transmembrane potential, h is the gate variable.
 *  F(u,h) is a nonlinear operator.
 *
 *  Class MitchellSchaeffer represents the right-hand side of the above
 *  system of ODEs. */
class MitchellSchaeffer : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   ParBlockNonlinearForm H;

public:
   MitchellSchaeffer(ParFiniteElementSpace &f,
                     double tau_in, double tau_out,
                     double tau_open, double tau_close,
                     double v_gate);

   /// Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &uh, Vector &duh_dt) const;
   
   // Compute Ionic current to be used in tissue simulations
   void GetIonicCurrent(const ParGridFunction &u, ParGridFunction &h,
                              ParGridFunction &I) const;

   virtual ~MitchellSchaeffer();

};


class MitchellSchaefferIntegrator : public BlockNonlinearFormIntegrator
{
public:
   MitchellSchaefferIntegrator(double tau_in, double tau_out,
                               double tau_open, double tau_close,
                               double v_gate);

   const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                   const FiniteElement &test_fe, ElementTransformation &Trans);

   virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                       ElementTransformation &Tr,
                                       const Array<const Vector *> &elfun,
                                       const Array<Vector *> &elvec) override;

private:
   double tau_in;
   double tau_out;
   double tau_open;
   double tau_close;
   double u_gate;
};


}






