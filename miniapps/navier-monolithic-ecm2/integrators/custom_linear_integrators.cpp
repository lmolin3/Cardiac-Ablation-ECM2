#include "custom_linear_integrators.hpp"

namespace mfem
{

// Linear Integrator for Neumann BC
void VectorNeumannLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = U -> FESpace() -> GetMesh() -> SpaceDimension();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);
   nor.SetSize(vdim);
   pn.SetSize(vdim);
   gradUn.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint (&ip);

      // Compute normal and normalize
      CalcOrtho(Tr.Jacobian(), nor);
      nor /= nor.Norml2();

      // Compute pn
      pn = nor;
      real_t beta = Q2.Eval(Tr, ip);
      real_t pval = P->GetValue(Tr, ip);
      pn *=  beta * pval; ;

      // Compute Q1 * n.grad(u)
      U->GetVectorGradient(Tr, gradU);
      gradU.Mult(nor,gradUn);
      real_t alpha = Q1.Eval(Tr, ip);
      gradUn *= alpha;

      // Compute vec = Q1 n.grad(u) + Q2 pn
      add(gradUn, pn, vec);

      vec *= Tr.Weight() * ip.weight;
      el.CalcShape(ip, shape);
      for (int k = 0; k < vdim; k++)
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s);
         }
   }
}

void VectorNeumannLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("VectorNeumannLFIntegrator::AssembleRHSElementVect\n"
              "  is not implemented as face integrator!\n"
              "  Use LinearForm::AddBoundaryIntegrator instead of\n"
              "  LinearForm::AddBdrFaceIntegrator.");
}



}  // namespace mfem