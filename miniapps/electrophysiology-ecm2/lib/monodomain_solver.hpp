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

// Class for MonodomainDiffusionSolver
//
// This class provides the operator used to solve the diffusion step of the Monodomain model,
// after applying operator splitting to the full Monodomain equations.
// The operator discretizes a time-dependent diffusion equation of the form:
//
//        chi Cm * du/dt = ∇•(sigma∇u) + f
//
// into the form:
//
//        M * du_dt = -K(u) + f
//
// where:
// - M is the mass matrix, scaled by chi * Cm.
// - K is the stiffness matrix, defined as K = ∇•(sigma∇).
// - f is the right-hand side, representing including Neumann and Robin boundary conditions.
//

#pragma once

#include <mfem.hpp>
#include "../bc/ep_bchandler.hpp"
#include "implicit_solver.hpp"
#include "../../common-ecm2/utils.hpp"

namespace mfem
{

  namespace electrophysiology
  {

    // Include functions from ecm2_utils namespace
    using namespace mfem::ecm2_utils;

    // Forward declaration of ImplicitSolverFA
    class ImplicitSolverBase;
    class ImplicitSolverFA;
    class ImplicitSolverPA;
    using SolverOptions = mfem::ecm2_utils::SolverOptions;

    class MonodomainDiffusionSolver : public TimeDependentOperator
    {
    public:
      MonodomainDiffusionSolver(ParFiniteElementSpace *fes,
                                  BCHandler *bcs_,
                                  MatrixCoefficient *sigma_coeff_,
                                  Coefficient *chi_coeff_, Coefficient *Cm_coeff_,
                                  int ode_solver_type = 21, 
                                  bool verbose = false);

      virtual ~MonodomainDiffusionSolver();

      // Enable partial assembly
      void EnablePA(bool pa_ = false);

      /** @brief Enable reassembling the implicit solver at every time step
       * @note: this needs to be called before Setup()
       * Rebuild of the implicit solver is required under the following conditions:
       * - The parameters are time-dependent (chi, Cm, sigma)
       * - It has time-dependent Robin BCs
       *
       * The rhs is assembled at every time step anyway, so Neumann and Dirichlet BCs are not a problem.
       * The implicit solver T is automatically reassembled, regardless of this flag if:
       * - the time step changes,
       * - the parameters are explicitly changed by calling SetParameters() */
      void EnableRebuild() { enable_rebuild = true; }

      /** Set up the MonodomainDiffusionSolver.
       * This involves adding all the necessary integrators to the linear form for
       * the rhs (neumann, robin contribution)
       */
      virtual void Setup( real_t dt = 0.0, int prec_type = 1);

      /** Update the MonodomainDiffusionSolver in case of changes in Mesh. */
      void Update();

      /** Perform one time-step of the simulation.
          If provisional is true, the time step is not counted (used for
          predictor-corrector methods). */
      void Step(Vector &x, real_t &t, real_t &dt, bool provisional = false);

      /** Compute action of the MonodomainDiffusionSolver: du_dt = M^{-1}*(-K(u)). */
      virtual void Mult(const Vector &u, Vector &du_dt) const;

      /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
          This is the only requirement for high-order SDIRK implicit integration.*/
      virtual void ImplicitSolve(const real_t dt, const Vector &u, Vector &k);

      /** Update time for bcs and rhs*/
      virtual void SetTime(const real_t time);

      /** Set the time step 
       * @note If different from the cached time step, the implicit solver is deleted and rebuilt at the next call to ImplicitSolve.
      */
      inline void SetTimeStep(const real_t dt);

      /**  Update timestep history
       * Not used for now, but could be useful to store*/
      void UpdateTimeStepHistory(Vector &x);

      /** Set the starting potential for the current step */
      void SetStartingPotential(const Vector *un);

      // Get ess_tdof_list
      Array<int> &GetEssTDofList() { return ess_tdof_list; }

      // Get the current solution gf
      ParGridFunction *GetPotentialGf() { return &u_gf; }

      protected:
      // Mesh and finite element space
      ParMesh *pmesh;             ///< NOT OWNED
      ParFiniteElementSpace *fes; ///< NOT OWNED
      int dim;
      int fes_truevsize;

      // BCHandler
      BCHandler *bcs; ///< OWNED
      Array<int> ess_tdof_list;
      
      // Enable partial assembly
      bool pa; 

      // Bilinear/Linear Forms
      std::unique_ptr<ParLinearForm> fform;
      std::unique_ptr<ParBilinearForm> M_form;
      std::unique_ptr<ParBilinearForm> K_form;
      std::unique_ptr<ParBilinearForm> RobinMass_form;

      // ParGridFunctions
      mutable ParGridFunction u_gf;      // Current solution
      mutable ParGridFunction du_dt_gf;  // Current time derivative

      // Vectors
      mutable Vector z, b;

      // Operators
      OperatorHandle opM;
      OperatorHandle opMe;
      OperatorHandle opK;
      OperatorHandle opRobinMass; // Mass matrix with Robin BCs
      HypreParMatrix *Mfull = nullptr;

      real_t cached_dt = 0.0;
      int current_step = 0;

      // ODESolver
      std::unique_ptr<ODESolver> ode_solver;

      // Linear Solvers
      SolverOptions solver_opts;
      std::unique_ptr<CGSolver> M_solver;    ///< Solver for the mass matrix
      std::unique_ptr<Solver> M_prec; ///< Preconditioner for the mass matrix

      bool implicit_time_integration = false; // Implicit time integration flag
      bool enable_rebuild = false;            // Trigger rebuild of the implicit solver at each time step
      std::unique_ptr<ImplicitSolverBase> T_solver; // Implicit solver for T = M + dt K

      // Coefficients
      MatrixCoefficient *sigma_coeff;                   ///< NOT OWNED
      std::unique_ptr<ProductCoefficient> chi_Cm_coeff; ///< OWNED

      // State variables
      ODEStateDataVector *u_prev;    // Previous solution

      // Verbosity
      bool verbose;

      // Set time for coefficients
      void SetCoefficientsTime(const real_t &time);

      // Build the implicit solver
      void BuildImplicitSolver();
    };

  } // namespace electrophysiology

} // namespace mfem
