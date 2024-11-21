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

#ifndef MFEM_ELECTROSTATICS_SOLVER
#define MFEM_ELECTROSTATICS_SOLVER

#include "../common/pfem_extras.hpp"
#include "../common/mesh_extras.hpp"
#include "electromagnetics.hpp"
#include "bc/electrostatics_bchandler.hpp"

#ifdef MFEM_USE_MPI

#include <string>
#include <map>

using namespace std;

namespace mfem
{

   using common::H1_ParFESpace;
   using common::ND_ParFESpace;
   using common::ParDiscreteGradOperator;

   namespace electrostatics
   {

      class ElectrostaticsSolver
      {
      public:
         ElectrostaticsSolver(std::shared_ptr<ParMesh> pmesh, int order,
                              BCHandler *bcs,
                              PWMatrixCoefficient *Sigma_,
                              bool verbose_ = false);

         ~ElectrostaticsSolver();

         HYPRE_BigInt GetProblemSize();

         void PrintSizes();

         void EnablePA(bool pa = false);

         void Setup();

         void Update();

         void Solve();

         // Add Volumetric term to rhs
         void AddVolumetricTerm(Coefficient *coeff, Array<int> &attr); // Using scalar coefficient
         void AddVolumetricTerm(ScalarFuncT func, Array<int> &attr);   // Using function
         void AddVolumetricTerm(Coefficient *coeff, int &attr);        // Using coefficient and single attribute
         void AddVolumetricTerm(ScalarFuncT func, int &attr);          // Using function and single attribute

         // Compute E^T M1 E, where M1 is the H1 mass matrix with conductivity
         // coefficient.
         double ElectricLosses(ParGridFunction &E_gf) const;

         // E is the input, w is the output which is L2 heating.
         void GetJouleHeating(ParGridFunction &E_gf, ParGridFunction &w_gf) const;

         void GetErrorEstimates(Vector &errors);

         void RegisterVisItFields(VisItDataCollection &visit_dc_);

         void RegisterParaviewFields(ParaViewDataCollection &paraview_dc_);

         void AddParaviewField(const std::string &field_name, ParGridFunction *gf);

         void AddVisItField(const std::string &field_name, ParGridFunction *gf);

         void WriteFields(int it = 0);

         void InitializeGLVis();

         void DisplayToGLVis();

         void PrintTimingData();

         // Prints the program's logo to the given output stream
         void display_banner(ostream &os);

         // Getters for phi and E
         ParGridFunction &GetPotential() { return *phi; }
         ParGridFunction &GetElectricField() { return *E; }

         // Getters for the FESpaces
         H1_ParFESpace *GetFESpace() { return H1FESpace; }

      private:
         void Assemble();

         void ProjectDirichletBCS( ParGridFunction &gf);

         int order; // Basis function order

         // Shared pointer to Mesh
         std::shared_ptr<ParMesh> pmesh;
         int dim;

         VisItDataCollection *visit_dc;       // To prepare fields for VisIt viewing
         ParaViewDataCollection *paraview_dc; // To prepare fields for ParaView viewing

         H1_ParFESpace *H1FESpace;    // Continuous space for phi
         ND_ParFESpace *HCurlFESpace; // Tangentially continuous space for E

         ParBilinearForm *divEpsGrad; // Laplacian operator
         ParBilinearForm *SigmaMass;  // Mass matrix with conductivity

         ParDiscreteGradOperator *grad; // For Computing E from phi

         ParLinearForm *rhs_form; // Dual of rhs

         OperatorHandle opA, opM;
         Vector Phi, *B;

         CGSolver solver;
         Solver *prec;

         bool pa; // Enable partial assembly

         ParGridFunction *phi; // Electric Scalar Potential
         ParGridFunction *E;   // Electric Field

         PWMatrixCoefficient *Sigma; // Electric conductivity Coefficient

         std::map<std::string, socketstream *> socks; // Visualization sockets

         Array<int> ess_tdof_list;        // (All) Essential Boundary Condition DoFs
         Array<int> ess_bdr_phi_tdofs;    // Essential Boundary Condition DoFs (potential)
         Array<int> ess_bdr_EField_tdofs; // Essential Boundary Condition DoFs (uniform electric field)

         BCHandler *bcs; // Boundary Condition Handler

         // Bookkeeping for voumetric terms.
         std::vector<CoeffContainer> volumetric_terms;
         Array<int> tmp_domain_attr; // Temporary domain attributes

         StopWatch sw_setup, sw_assemble, sw_solve;

         bool verbose;
      };

      // A Coefficient is an object with a function Eval that returns a double. The
      // JouleHeatingCoefficient object will contain a reference to the electric field
      // grid function, and the conductivity sigma, and returns sigma E dot E at a
      // point.
      class JouleHeatingCoefficient : public Coefficient
      {
      private:
         ParGridFunction &E_gf;
         PWMatrixCoefficient Sigma;

      public:
         JouleHeatingCoefficient(const PWMatrixCoefficient &Sigma_,
                                 ParGridFunction &E_gf_)
             : E_gf(E_gf_), Sigma(Sigma_) {}
         virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
         virtual ~JouleHeatingCoefficient() {}
      };

   } // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_VOLTA_SOLVER
