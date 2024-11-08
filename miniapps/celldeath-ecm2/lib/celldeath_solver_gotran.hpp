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
// The CellDeathSolverGotran class is a simple class that solves the three-state cell-death model.
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

#ifndef MFEM_CELLDEATH_SOLVER_GOTRAN
#define MFEM_CELLDEATH_SOLVER_GOTRAN

#include "mesh_extras.hpp"
#include "pfem_extras.hpp"
#include "utils.hpp"
#include "ThreeStateCellDeath.h"

#ifdef MFEM_USE_MPI

namespace mfem
{

      using common::H1_ParFESpace;

      namespace celldeathgotran
      {

            class CellDeathSolverGotran
            {
            public:
                  CellDeathSolverGotran(std::shared_ptr<ParMesh> pmesh_,
                                       int order_, ParGridFunction *T_,
                                       real_t A1_, real_t A2_, real_t A3_,
                                       real_t deltaE1_, real_t deltaE2_, real_t deltaE3_,
                                       bool verbose = false);

                  ~CellDeathSolverGotran();

                  // Setup the projection
                  void SetupProjection();

                  // Project the temperature field
                  void ProjectTemperature(Vector &Tin, Vector &Tout);

                  // Solve the system
                  void Solve(real_t t, real_t dt, int method, int substeps);

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

                  // ODE model
                  static const int num_param = 3;
                  static const int num_states = 3;

                  real_t parameters[num_param];
                  real_t (*parameters_nodes)[num_param];
                  real_t init_states[num_states];
                  real_t (*states)[num_states];

                  // Coefficients
                  ParGridFunction *T_gf;
                  real_t A1, A2, A3;
                  real_t deltaE1, deltaE2, deltaE3;
                  static constexpr real_t R = 8.31446261815324; // J/(mol*K)

                  // Postprocessing
                  VisItDataCollection *visit_dc;       // To prepare fields for VisIt viewing
                  ParaViewDataCollection *paraview_dc; // To prepare fields for ParaView viewing

                  bool verbose;
            };

      } // namespace celldeathgotran

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_CELLDEATH_SOLVER_GOTRAN
