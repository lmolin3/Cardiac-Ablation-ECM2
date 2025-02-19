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