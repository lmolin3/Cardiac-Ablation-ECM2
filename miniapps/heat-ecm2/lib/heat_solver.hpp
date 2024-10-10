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

#ifndef MFEM_HEAT_SOLVER
#define MFEM_HEAT_SOLVER

#include "mesh_extras.hpp"
#include "pfem_extras.hpp"
#include "bc/heat_bchandler.hpp"
#include "heat_operator.hpp"

#ifdef MFEM_USE_MPI

#include <map>
#include <string>

namespace mfem
{

  using common::H1_ParFESpace;

  namespace heat
  {
    // Converters from Celsius to Kelvin
    double CelsiusToKelvin(double Tc);
    double KelvinToCelsius(double Tk);

    class HeatSolver
    {
    public:
      HeatSolver(std::shared_ptr<ParMesh> pmesh, int order, BCHandler *bcs,
                 MatrixCoefficient *Kappa = nullptr,
                 Coefficient *c_ = nullptr, Coefficient *rho_ = nullptr,
                 real_t advection = 0.0,
                 VectorCoefficient *u = nullptr,
                 real_t reaction = 0.0,
                 int ode_solver_type = 1, bool verbose = true);

      // overload for the case of purely conductive heat transfer
      HeatSolver(std::shared_ptr<ParMesh> pmesh, int order, BCHandler *bcs,
                 MatrixCoefficient *Kappa,
                 Coefficient *c_, Coefficient *rho_,
                 int ode_solver_type, bool verbose = true);

      ~HeatSolver();

      HYPRE_BigInt GetProblemSize();

      H1_ParFESpace *GetFESpace() { return H1FESpace; };
      H1_ParFESpace *GetVectorFESpace() { return VectorH1FESpace; };

      void PrintSizes();

      /// Enable partial assembly for every operator.
      // (to be effective, must be set before Setup() is called)
      void EnablePA(bool pa) { op->EnablePA(pa); }

      /// Set the solver and AdvectionReactionDiffusionOperator
      void Setup();

      void SetInitialTemperature(ParGridFunction &T0);

      void Update();

      void Step(double &time, double dt, int step, bool UpdateHistory = true);

      // Explicit update of the time step history: useful to avoid automatic one in Step method (for multiple solutions at same time step)
      void UpdateTimeStepHistory();

      // Add Volumetric heat term (to AdvectionReactionDiffusionOperator)
      void AddVolumetricTerm(Coefficient *coeff,
                             Array<int> &attr);                   // Using scalar coefficient
      void AddVolumetricTerm(ScalarFuncT func, Array<int> &attr); // Using function
      void AddVolumetricTerm(Coefficient *coeff,
                             int &attr); // Using coefficient and single attribute
      void AddVolumetricTerm(ScalarFuncT func,
                             int &attr); // Using function and single attribute

      // Visualization and Postprocessing
      void RegisterVisItFields(VisItDataCollection &visit_dc_);

      void RegisterParaviewFields(ParaViewDataCollection &paraview_dc_);

      void AddParaviewField(const std::string &field_name, ParGridFunction *gf);

      void AddVisItField(const std::string &field_name, ParGridFunction *gf);

      void WriteFields(const int &it = 0, const double &time = 0);

      ParaViewDataCollection &GetParaViewDc() { return *paraview_dc; }
      VisItDataCollection &GetVisItDc() { return *visit_dc; }

      void InitializeGLVis();

      void DisplayToGLVis();

      std::vector<double> GetTimingData();

      void PrintTimingData();

      void display_banner(std::ostream &os);

      void SetVerbose(bool verbose_) { verbose = verbose_; }

      // Getters for T
      ParGridFunction &GetTemperatureGf() { return *T_gf; }
      ParGridFunction *GetTemperatureGfPtr() { return T_gf; }

      Vector &GetTemperature() { return *T; }
      Vector *GetTemperaturePtr() { return T; }

    private:
      ODESolver *CreateODESolver(int ode_solver_type, TimeDependentOperator &op);

      /* Compute approximation of first derivative on essential tdofs*/
      void ComputeDerivativeApproximation(const Vector &T, double dt) const;

      /* Rotate the solution history */
      void UpdateTimeStepHistory(const Vector &u);

      /* Set time integration coefficients for derivative approximation */
      void SetTimeIntegrationCoefficients(int step);

      int order; // Basis function order

      // Shared pointer to Mesh
      std::shared_ptr<ParMesh> pmesh;
      int dim;

      H1_ParFESpace *H1FESpace;
      H1_ParFESpace *VectorH1FESpace;

      int fes_truevsize;
      Array<int> ess_tdof_list;

      ParGridFunction *T_gf; // Temperature field grid function
      Vector *T;             // Temperature field vector

      BCHandler *bcs; // Boundary Condition Handler

      AdvectionReactionDiffusionOperator *op; // Conduction Operator

      ODESolver *ode_solver; // ODE Solver

      Array<int> tmp_domain_attr; // Temporary domain attributes

      VisItDataCollection *visit_dc;       // To prepare fields for VisIt viewing
      ParaViewDataCollection *paraview_dc; // To prepare fields for ParaView viewing

      std::map<std::string, socketstream *> socks; // Visualization sockets

      StopWatch sw_init, sw_setup, sw_solve;

      bool verbose;

      // Previous solutions (to compute the time derivative for dirichlet bcs)
      int time_scheme_order;
      double alpha;
      Vector beta;
      std::deque<Vector> T_prev;
      mutable Vector T_bcs; // auxiliary vector for bcs
    };

  } // namespace heat

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_HEAT_SOLVER
