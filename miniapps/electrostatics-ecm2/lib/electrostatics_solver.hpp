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

      class JouleHeatingCoefficient: public Coefficient
      {
      private:
         ParGridFunction *E_gf;
         Coefficient *Q;
         MatrixCoefficient *MQ;
      public:
         JouleHeatingCoefficient(Coefficient *Sigma_,
                                 ParGridFunction *E_gf_)
             : E_gf(E_gf_), Q(Sigma_), MQ(NULL) {};

         JouleHeatingCoefficient(MatrixCoefficient *Sigma_,
                                 ParGridFunction *E_gf_)
            : E_gf(E_gf_), Q(NULL), MQ(Sigma_) {};
         real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
         virtual ~JouleHeatingCoefficient() {}
      };

      class ElectrostaticsSolver
      {
      public:
         ElectrostaticsSolver(std::shared_ptr<ParMesh> pmesh, int order,
                              BCHandler *bcs,
                              Coefficient *Sigma_,
                              bool verbose_ = false);

         ElectrostaticsSolver(std::shared_ptr<ParMesh> pmesh, int order,
                              BCHandler *bcs,
                              MatrixCoefficient *Sigma_,
                              bool verbose_ = false);

         ~ElectrostaticsSolver();

         HYPRE_BigInt GetProblemSize();
         int GetLocalProblemSize();

         void PrintSizes();

         void EnablePA(bool pa = false);

         void Setup(int prec_type = 1, int pl = 0);

         void Update();

         // Update rhs useful for transient simulations (or domain decomposition iteration)
         void Solve(bool updateRhs = false);

         // Add Volumetric term to rhs
         void AddVolumetricTerm(Coefficient *coeff, Array<int> &attr); // Using scalar coefficient
         void AddVolumetricTerm(ScalarFuncT func, Array<int> &attr);   // Using function
         void AddVolumetricTerm(Coefficient *coeff, int &attr);        // Using coefficient and single attribute
         void AddVolumetricTerm(ScalarFuncT func, int &attr);          // Using function and single attribute

         // Compute E^T M1 E, where M1 is the H1 mass matrix with conductivity
         // coefficient.
         real_t ElectricLosses(ParGridFunction &E_gf) const;

         // w is the output which is L2 heating. This just projects the Joule heating coefficient which is already setup internally with the electric field and conductivity.
         void GetJouleHeating(ParGridFunction &w_gf) const;

         Coefficient *GetJouleHeatingCoefficient() { return static_cast<Coefficient*>(w_coeff); }

         void GetErrorEstimates(Vector &errors);

         void RegisterVisItFields(VisItDataCollection &visit_dc_);

         void RegisterParaviewFields(ParaViewDataCollection &paraview_dc_);

         void AddParaviewField(const std::string &field_name, ParGridFunction *gf);

         void AddVisItField(const std::string &field_name, ParGridFunction *gf);

         void WriteFields(int it = 0, const real_t &time = 0.0);

         void InitializeGLVis();

         void DisplayToGLVis();

         void PrintTimingData();

         // Prints the program's logo to the given output stream
         void display_banner(ostream &os);

         // Getters for phi and E
         ParGridFunction *GetPotentialGfPtr() { return phi; }
         ParGridFunction &GetPotential() { return *phi; }
         ParGridFunction &GetElectricField() { return *E; }

         // Getters for the FESpaces
         ParFiniteElementSpace *GetFESpace() { return H1FESpace; }
         ParFiniteElementSpace *GetL2FESpace() { return L2FESpace; }

      private:
         // Check if any essential BCs were applied and fix at least one point since solution is not unique
         void FixEssentialTDofs( Array<int> &ess_tdof_list);

         void Assemble();

         void AssembleRHS();

         void ProjectDirichletBCS( ParGridFunction &gf);

         int order; // Basis function order

         // Shared pointer to Mesh
         std::shared_ptr<ParMesh> pmesh;
         int dim;

         VisItDataCollection *visit_dc;       // To prepare fields for VisIt viewing
         ParaViewDataCollection *paraview_dc; // To prepare fields for ParaView viewing

         ParFiniteElementSpace *H1FESpace;    // Continuous space for phi
         ParFiniteElementSpace *L2FESpace;    // Discontinuous space for w
         ParFiniteElementSpace *HCurlFESpace; // Tangentially continuous space for E

         ParBilinearForm *divEpsGrad; // Laplacian operator
         ParBilinearForm *SigmaMass;  // Mass matrix with conductivity

         ParDiscreteGradOperator *grad; // For Computing E from phi

         ParLinearForm *rhs_form; // Dual of rhs

         OperatorHandle opA, opM;
         Vector Phi, B;

         IterativeSolver *solver;
         Solver *prec;
         int prec_type;
         bool symmetric = true;

         bool pa; // Enable partial assembly

         ParGridFunction *phi; // Electric Scalar Potential
         ParGridFunction *E;   // Electric Field

         JouleHeatingCoefficient *w_coeff; // Joule Heating Coefficient

         Coefficient *SigmaQ; // Electric conductivity Coefficient
         MatrixCoefficient *SigmaMQ; // Electric conductivity Coefficient

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

         int my_id;
         int num_procs;
      };

   } // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_VOLTA_SOLVER

