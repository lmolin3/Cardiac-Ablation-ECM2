#include "custom_integrators.hpp"

namespace mfem
{

/**
 * @brief Get the integration rule for the given finite element and transformation.
 *
 * @param fe The finite element for which to get the integration rule.
 * @param T The element transformation.
 * @return const IntegrationRule& The integration rule.
 */
const IntegrationRule& VectorConvectionIntegrator::GetRule(const FiniteElement &fe,
        ElementTransformation &T)
{
    const int order = 2 * fe.GetOrder() + T.OrderGrad(&fe);
    return IntRules.Get(fe.GetGeomType(), order);
}

/**
 * @brief Assemble the element matrix for the VectorConvectionIntegrator.
 *
 * @param el The finite element.
 * @param Trans The element transformation.
 * @param elmat The element matrix to be assembled.
 */
void VectorConvectionIntegrator::AssembleElementMatrix(const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{
    const int dof = el.GetDof();
    dim = el.GetDim();

    Vector vec1;
    elmat.SetSize(dim * dof);
    dshape.SetSize(dof, dim);
    adjJ.SetSize(dim);
    shape.SetSize(dof);
    vec2.SetSize(dim);
    vec3.SetSize(dof);
    pelmat.SetSize(dof);
    DenseMatrix pelmat_T(dof);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &GetRule(el, Trans);
    }

    W->Eval(W_ir, Trans, *ir);

    elmat = 0.0;
    pelmat_T = 0.0;

    // Calculate constant values outside the loop
    const double alpha_weight = alpha * ir->IntPoint(0).weight;
    Trans.SetIntPoint(&ir->IntPoint(0));
    CalcAdjugate(Trans.Jacobian(), adjJ);
    const double q_const = alpha_weight * Trans.Weight();

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcDShape(ip, dshape);
        el.CalcShape(ip, shape);

        Trans.SetIntPoint(&ip);
        CalcAdjugate(Trans.Jacobian(), adjJ);
        W_ir.GetColumnReference(i, vec1);     // tmp = W

        const double q = alpha * ip.weight; // q = alpha*weight   || q = weight
        adjJ.Mult(vec1, vec2);               // element Transformation J^{-1} |J|
        vec2 *= q;

        dshape.Mult(vec2, vec3);           // (w . grad u)           q ( alpha J^{-1} |J| w dPhi )
        MultVWt(shape, vec3, pelmat);      // (w . grad u,v)         q ( alpha J^{-1} |J| w dPhi Phi^T)

        if (SkewSym)
        {
            pelmat_T.Transpose(pelmat);
        }

        for (int k = 0; k < dim; k++)
        {
            if (SkewSym)
            {
                elmat.AddMatrix(.5, pelmat, dof * k, dof * k);
                elmat.AddMatrix(-.5, pelmat_T, dof * k, dof * k);
            }
            else
            {
                elmat.AddMatrix(pelmat, dof * k, dof * k);
            }
        }
    }
}

/**
 * @brief Get the integration rule for the VectorGradCoefficientIntegrator.
 *
 * @param fe The finite element for which to get the integration rule.
 * @param T The element transformation.
 * @return const IntegrationRule& The integration rule.
 */
const IntegrationRule& VectorGradCoefficientIntegrator::GetRule(const FiniteElement &fe,
        ElementTransformation &T)
{
    return VectorConvectionIntegrator::GetRule(fe, T);
}

/**
 * @brief Assemble the element matrix for the VectorGradCoefficientIntegrator.
 *
 * @param el The finite element.
 * @param Trans The element transformation.
 * @param elmat The element matrix to be assembled.
 */
void VectorGradCoefficientIntegrator::AssembleElementMatrix(const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{
    const int dof = el.GetDof();
    dim = el.GetDim();

    shape.SetSize(dof);
    elmat.SetSize(dof * dim);
    pelmat.SetSize(dof);
    gradW.SetSize(dim);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &GetRule(el, Trans);
    }

    elmat = 0.0;
    // compute gradient (with respect to the physical element)
    W_gf = W->GetGridFunction();

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        el.CalcShape(ip, shape);

        W_gf->GetVectorGradient(Trans, gradW);

        MultVVt(shape, pelmat);
        const double q = alpha * ip.weight * Trans.Weight();

        for (int ii = 0; ii < dim; ii++)
        {
            for (int jj = 0; jj < dim; jj++)
            {
                elmat.AddMatrix(q * gradW(ii, jj), pelmat, ii * dof, jj * dof);
            }
        }
    }
}


/**
 * @brief Assemble the element matrix for the VectorGradCoefficientIntegrator.
 *
 * @param el The finite element.
 * @param Trans The element transformation.
 * @param elmat The element matrix to be assembled.
 */
void LumpedVectorMassIntegrator::AssembleElementMatrix (
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
    int spaceDim = Trans.GetSpaceDim();
    const int dof = el.GetDof();
    
   // Retrieve vdim from VectorMassIntegrator. If vdim is not set, set it to the space dimension
   double vdim = GetVDim();
   vdim = (vdim == -1) ? spaceDim : vdim;

    VectorMassIntegrator::AssembleElementMatrix (el, Trans, elmat); // Assemble consistent Mass Matrix
    ones.SetSize( vdim * dof ); 
    ones = 1.0;
    M_tot = elmat.InnerProduct(ones,ones); // Mtot = 1^T M 1
    M_lump   = elmat.Trace();                            // Compute trace
    //M_tot = TotalMass(el, Trans, elmat, vdim);     // Compute total mass

    double s = M_tot / M_lump ;                 // Compute scaling
    
    // Lump matrix
    int h = elmat.Height();
    int w = elmat.Width();
    double elmat_diag = 0.0;
    for (int i = 0; i < h; i++)
    {
        elmat_diag = (elmat)(i, i);
        for (int j = 0; j < w; j++)
        {
            (elmat)(i, j) = 0.0;
        }
        (elmat)(i, i) = s * elmat_diag;
    }
}

/**
 * @brief Integrate coefficient associated to VectorMassIntegrator to get Total Mass Mtot.
 *
 */
double LumpedVectorMassIntegrator::TotalMass (
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat, double vdim)
{
   int nd = el.GetDof();

   if (VQ)
   {
      vec.SetSize(vdim);
   }
   else if (MQ)
   {
      mcoeff.SetSize(vdim);
   }

   const IntegrationRule *ir = IntRule;
   int Q_order = 0;
   if (ir == NULL)
   {
       int order = 2 * el.GetOrder() + Trans.OrderW() + Q_order;

       if (el.Space() == FunctionSpace::rQk)
       {
           ir = &RefinedIntRules.Get(el.GetGeomType(), order);
       }
       else
       {
           ir = &IntRules.Get(el.GetGeomType(), order);
       }
   }

   double Mtot = 0.0;
   vec = 0.0;
   mcoeff = 0.0;
   for (int s = 0; s < ir->GetNPoints(); s++)
   {
        const IntegrationPoint &ip = ir->IntPoint(s);
        Trans.SetIntPoint (&ip);

        if (Q)       // Scalar coefficient
        {
            double tmp = Q->Eval(Trans, ip);
            Mtot += ip.weight * tmp;
        }
        else if (VQ) // Vector coefficient
        {   
            Vector tmp;
            VQ->Eval(tmp, Trans, ip);
            tmp *= ip.weight;
            vec += tmp;
        }
        else        // Matrix coefficient
        {
            DenseMatrix tmp;
            MQ->Eval(tmp, Trans, ip);
            tmp *= ip.weight;
            mcoeff += tmp;
        }
   };

    // Copmpute Mtot from integrated Vector/Matrix coefficient
    if (VQ)
    {
        Mtot = vec.Norml2();
    }
    else if (MQ)
    {
        Mtot = mcoeff.FNorm();
    }

   return Mtot;

}


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