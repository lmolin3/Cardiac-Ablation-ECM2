#include "ionic_models.hpp"

namespace mfem
{

MitchellSchaeffer::MitchellSchaeffer(ParFiniteElementSpace &f,
                                     double tau_in, double tau_out,
                                     double tau_open, double tau_close,
                                     double v_gate)
    : TimeDependentOperator(2*f.TrueVSize(), 0.0), fespace(f),
{
    H.AddDomainIntegrator(new MitchellSchaefferIntegrator(tau_in, tau_out, tau_open, tau_close, u_gate));
}

void MitchellSchaeffer::Mult(const Vector &uh, Vector &duh_dt) const
{
    // Compute rhs of duh/dt
    H.Mult(uh, duh_dt);       // Call to ParBlockNonlinearForm
}


void MitchellSchaeffer::GetIonicCurrent(const ParGridFunction &uh, ParGridFunction &I) const
{
    // Implement the logic to compute the ionic current based on u and h
    // ...
}




MitchellSchaefferIntegrator::MitchellSchaefferIntegrator(double tau_in, double tau_out,
                                                         double tau_open, double tau_close,
                                                         double v_gate)
    : tau_in(tau_in), tau_out(tau_out), tau_open(tau_open), tau_close(tau_close),
{}


const IntegrationRule &MitchellSchaefferIntegrator::GetRule(const FiniteElement &trial_fe,
                                                            const FiniteElement &test_fe, ElementTransformation &Trans)
{
    const int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
    return IntRules.Get(trial_fe.GetGeomType(), order);
}

void MitchellSchaefferIntegrator::AssembleElementVector(const
                                                   Array<const FiniteElement *> &el,
                                                   ElementTransformation &Tr,
                                                   const Array<const Vector *> &elfun,
                                                   const Array<Vector *> &elvec)
{
    int nd = el.GetDof();       // same for both u and h
    int dim = el[0]->GetDim();

    Vector shape(nd);
    
    elvec[0]->SetSize(nd);
    elvec[1]->SetSize(nd);

    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Tr);

   *elvec[0] = 0.0;
   *elvec[1] = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr.SetIntPoint(&ip);
        double w = Tr.Weight() * ip.weight;

        el.CalcShape(ip, shape);
        double u = shape * *elfun[0];    
        double h = shape * *elfun[1];
        
        // Take inner product with jth test function,
        for (int j=0; j<nd; j++)
        {
            // Compute rhs of du/dt
            (*elvect[0])[j] += 1.0/tau_in * w*shape[j]*h*u*u*(1.0 - u); // 1/tau_in h u^2 (1-u)
            (*elvect[0])[j] -= 1.0/tau_out * w*shape[j]*u;                       // -1/tau_out u

            // Compute rhs of dh/dt
            if( u < u_gate ) { 1.0/tau_open * w * shape[j] *(1.0 - h);} // 1/tau_open (1-h)
            else { -1.0/tau_close * w * shape[j] * h; }                 // -h/tau_close 
        } 
    }
}

}// namespace mfem
