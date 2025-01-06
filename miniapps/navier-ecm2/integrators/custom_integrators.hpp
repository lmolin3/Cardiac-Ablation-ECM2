// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"

namespace mfem
{

/** Class for integrating the trilinear form for the convective operator 
    
    c(w,u,v) := alpha (w . grad u, v) = alpha * w . grad u
    
    for vector field FE spaces, where alpha is a scalar, w is a known vector field (VectorCoefficient),
    u=(u1,...,un) and v=(v1,...,vn); ui and vi are defined by scalar FE through standard transformation.
    The resulting local element matrix is square, of size <tt> dim*dof </tt>,
    where \c dim is the vector dimension space and \c dof is the local degrees
    of freedom. 
*/

class VectorConvectionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient *W;
   double alpha;
   bool SkewSym;
   // PA extension // Not supported yet
   Vector pa_data;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

private:
   DenseMatrix dshape, adjJ, W_ir, pelmat, pelmat_T;
   Vector shape, vec1, vec2, vec3;

public:
   VectorConvectionIntegrator(VectorCoefficient &w, double alpha = 1.0, bool SkewSym_ = false)
      : W(&w), alpha(alpha), SkewSym(SkewSym_) {}

   static const IntegrationRule &GetRule(const FiniteElement &fe,
                                         ElementTransformation &T);
                                         
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

};


/** Class for integrating the trilinear form for the convective operator 
    
    c(u,w,v) := alpha (u . grad w, v) = alpha * u . grad w
    
    for vector field FE spaces, where alpha is a scalar, w is a known vector field (VectorGridFunctionCoefficient),
    u=(u1,...,un) and v=(v1,...,vn); ui and vi are defined by scalar FE through standard transformation.
    The resulting local element matrix is square, of size <tt> dim*dof </tt>,
    where \c dim is the vector dimension space and \c dof is the local degrees
    of freedom. 
*/
class VectorGradCoefficientIntegrator : public BilinearFormIntegrator
{
protected:
   VectorGridFunctionCoefficient *W;
   const GridFunction* W_gf;
   double alpha;
   // PA extension // Not supported yet
   Vector pa_data;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

private:
   DenseMatrix pelmat, gradW;
   Vector shape;

public:
   VectorGradCoefficientIntegrator(VectorGridFunctionCoefficient &w, double alpha = 1.0)
      : W(&w), alpha(alpha) {}


   static const IntegrationRule &GetRule(const FiniteElement &fe,
                                       ElementTransformation &T);

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
};


/** Class for lumping the VectorMassIntegrator using diagonal scaling:
    
   Ml_{ii} = M_{tot}/M_{lump}  M_{ii}

   where M_{tot} = int_vol Q dV    is the total mass (Q is a SCALAR coefficient)
    
   see VectorMassIntegrator for the definition of the consistent Mass Matrix.
*/
class LumpedVectorMassIntegrator : public VectorMassIntegrator
{
private:
   Vector diag;
   double M_lump, M_tot;
   DenseMatrix mcoeff;
   Vector vec, ones;

public:
   /// Construct an integrator with coefficient 1.0
   LumpedVectorMassIntegrator()
      : VectorMassIntegrator(), diag(0.0), M_lump(0.0), M_tot(0.0), mcoeff(0.0), vec(0.0) {};

   /** Construct an integrator with scalar coefficient q.  If possible, save
       memory by using a scalar integrator since the resulting matrix is block
       diagonal with the same diagonal block repeated. */
   LumpedVectorMassIntegrator(Coefficient &q, int qo = 0)
      : VectorMassIntegrator(q, qo),  diag(0.0), M_lump(0.0), M_tot(0.0), mcoeff(0.0), vec(0.0)  {};

   LumpedVectorMassIntegrator(Coefficient &q, const IntegrationRule *ir)
      : VectorMassIntegrator(q, ir), diag(0.0), M_lump(0.0), M_tot(0.0), mcoeff(0.0), vec(0.0)  {};

   /// Construct an integrator with diagonal coefficient q
   LumpedVectorMassIntegrator(VectorCoefficient &q, int qo = 0)
      : VectorMassIntegrator(q, qo), diag(0.0), M_lump(0.0), M_tot(0.0), mcoeff(0.0), vec(0.0)  {};

   /// Construct an integrator with matrix coefficient q
   LumpedVectorMassIntegrator(MatrixCoefficient &q, int qo = 0)
      : VectorMassIntegrator(q, qo), diag(0.0), M_lump(0.0), M_tot(0.0), mcoeff(0.0), vec(0.0)  {};


   virtual double TotalMass(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat,
                                      double vdim);

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

};



/** Class for boundary integration of Neumann BCs with known vector and scalar field
 * 
 * L(v) := (Q1 n.grad(U) + Q2 P.n, v), where
 * 
 * U is a vector field and P is a scalar field. Q1 and Q2 are scalar Coefficients.
 * 
 * (e.g. for Navier Stokes Q1=viscosity, Q2 = -1.0)
 * 
 **/
class VectorNeumannLFIntegrator : public LinearFormIntegrator
{
private:
    const ParGridFunction *U;
    const ParGridFunction *P;
    Coefficient &Q1, &Q2;
    Vector shape, vec, nor, pn, gradUn;
    DenseMatrix gradU;

public:
   /// Constructs a boundary integrator with a given VectorCoefficient QG
   VectorNeumannLFIntegrator(ParGridFunction &U, ParGridFunction &P, Coefficient &Q1, Coefficient &Q2)
    : U(&U), P(&P), Q1(Q1), Q2(Q2) {}

   /** Given a particular boundary Finite Element and a transformation (Tr)
       computes the element boundary vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   // For DG spaces    NYI
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

}