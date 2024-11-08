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
//
// Class for Cell Death Solver
//
// The CellDeathSolver class is a simple class that solves the three-state cell-death model.
// The model is based on the following system of ODEs:
//
// dN/dt = -k1 N + k2 U
// dU/dt = k1 N - k2 U - k3 D
// dD/dt = k3 U
//
// where N, U, and D are the number of alive, vulnerable, and dead cells, respectively.
//
// The rate coefficients k1, k2, and k3 are given by:
// ki = Ai * exp(-deltaEi / (R * Ti))
//
// The ODE system is solved at each quadrature point using the Eigenvalue method.

#ifndef MFEM_CELLDEATH_SOLVER
#define MFEM_CELLDEATH_SOLVER

#include "mesh_extras.hpp"
#include "pfem_extras.hpp"
#include "utils.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

      using common::H1_ParFESpace;

      namespace celldeath
      {

            class CellDeathSolver
            {
            public:
                  CellDeathSolver(std::shared_ptr<ParMesh> pmesh, int order,
                             ParGridFunction *T_,
                             real_t A1_, real_t A2_, real_t A3_,
                             real_t deltaE1_, real_t deltaE2_, real_t deltaE3_,
                             bool verbose = false);

                  ~CellDeathSolver();

                  // Setup the projection
                  void SetupProjection();

                  // Project the temperature field
                  void ProjectTemperature(Vector &Tin, Vector &Tout);

                  // Solve the system
                  void Solve(real_t t, real_t dt);

                  // Visualization and Postprocessing
                  void RegisterVisItFields(VisItDataCollection &visit_dc_);

                  void RegisterParaviewFields(ParaViewDataCollection &paraview_dc_);

                  void AddParaviewField(const std::string &field_name, ParGridFunction *gf);

                  void AddVisItField(const std::string &field_name, ParGridFunction *gf);

                  void WriteFields(const int &it = 0, const double &time = 0);

                  ParaViewDataCollection &GetParaViewDc() { return *paraview_dc; }
                  VisItDataCollection &GetVisItDc() { return *visit_dc; }

                  HYPRE_BigInt GetProblemSize();

                  void display_banner(std::ostream &os);

                  void SetVerbose(bool verbose_) { verbose = verbose_; }

                  // Getters
                  ParGridFunction &GetAliveCellsGf() { return N_gf; }
                  ParGridFunction &GetVulnerableCellsGf() { return U_gf; }
                  ParGridFunction &GetDeadCellsGf() { return D_gf; }


            private:
                  // Private methods
                  // Compute eigenvalues/eigenvectors given the coefficients ki (handle cases in which ki = 0)
                  inline void EigenSystem(real_t k1, real_t k2, real_t k3, Vector &lambda, DenseMatrix &P);

                  // Shared pointer to Mesh
                  std::shared_ptr<ParMesh> pmesh;
                  int dim;

                  // FE spaces
                  FiniteElementCollection *fec;
                  ParFiniteElementSpace *fes;
                  ParFiniteElementSpace *fesT;
                  int fes_truevsize;
                  int fesT_truevsize;
                  int order;
                  int orderT;
                  TrueTransferOperator *transferOp;

                  // Grid functions and Vectors
                  ParGridFunction N_gf;   // Alive cells grid function
                  ParGridFunction U_gf;   // Vulnerable cells grid function
                  ParGridFunction D_gf;   // Dead cells grid function

                  Vector N, U, D, T, Tsrc;

                  // Coefficients
                  ParGridFunction *T_gf;
                  real_t A1, A2, A3;
                  real_t deltaE1, deltaE2, deltaE3;
                  static constexpr real_t R = 8.31446261815324; // J/(mol*K)
                  const real_t invR = 1.0 / R;
                  real_t k1, k2, k3;

                  // Eigenvalue problem
                  Vector Xn;       // initial conditions
                  Vector X;        // solution

                  // Postprocessing
                  VisItDataCollection *visit_dc;       // To prepare fields for VisIt viewing
                  ParaViewDataCollection *paraview_dc; // To prepare fields for ParaView viewing

                  bool verbose;
            };

      } // namespace celldeath

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_CELLDEATH_SOLVER
