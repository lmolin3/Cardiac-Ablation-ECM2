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
// - f is the right-hand side, representing Neumann boundary conditions.
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
      /// Optional common mass/diffusion rule; must outlive this solver.
      void SetIntegrationRule(const IntegrationRule *ir) { integration_rule = ir; }

      /** @brief Optional separate rule for the mass form only; must outlive this solver.
       *
       * The point of a separate rule is mass lumping. With an H1 Gauss-Lobatto
       * basis and a *collocated* Gauss-Lobatto rule (one quadrature point per
       * node) the mass matrix is diagonal, so M^{-1} costs a single scaling
       * instead of a CG solve. That matters only on the explicit path, where
       * Mult() applies M^{-1} every stage; the implicit path solves M + dt K and
       * gains nothing. Collocation under-integrates the mass form, which on a
       * curved mesh is a real approximation, so it is opt-in and reported. */
      void SetMassIntegrationRule(const IntegrationRule *ir) { mass_integration_rule = ir; }

      /// True when Setup() selected an explicit ODE solver.
      bool UsesImplicitTimeIntegration() const { return implicit_time_integration; }

      /** @brief Enable reassembling the implicit solver at every time step
       * @note: this needs to be called before Setup()
       * Rebuild of the implicit solver is required under the following conditions:
       * - The parameters are time-dependent (chi, Cm, sigma)
       *
       * The rhs is assembled at every time step anyway, so Neumann and Dirichlet BCs are not a problem.
       * The implicit solver T is automatically reassembled, regardless of this flag if:
       * - the time step changes,
       * - the parameters are explicitly changed by calling SetParameters() */
      void EnableRebuild() { enable_rebuild = true; }

      /** Set up the MonodomainDiffusionSolver.
       * This involves adding all the necessary integrators to the linear form for
       * the rhs (neumann contribution)
       */
      /** @param prec_type Preconditioner for the PA implicit solve:
       *    0 - Jacobi (default), 1 - LOR + BoomerAMG.
       *
       * Jacobi is the default because the monodomain implicit operator
       * T = chi*Cm*M + dt*sigma*K is strongly mass dominated: with cardiac values
       * chi*Cm ~ 1.4 while dt*sigma ~ 1e-4, so T is close to a (well conditioned)
       * mass matrix. */
      virtual void Setup( real_t dt = 0.0, int prec_type = 0, real_t rel_tol = 1e-6,
                          bool warm_start = true);

      /** Update the MonodomainDiffusionSolver in case of changes in Mesh or FiniteElementSpace */
      void Update();

      /** Perform one time-step of the simulation.
          If provisional is true, the time step is not counted (used for
          predictor-corrector methods). */
      void Step(Vector &x, real_t &t, real_t &dt, bool provisional = false);

      /** @brief Memory space this operator wants its input/output vectors in.
       *
       * Operator::GetMemoryClass() defaults to MemoryClass::HOST. ODESolver sizes
       * its work vectors with GetMemoryType(f.GetMemoryClass()), so without this
       * override the time integrator's work vectors (e.g. BackwardEulerSolver::k)
       * are allocated in host memory even under a device backend, and every step
       * pays to migrate them. Reporting the device memory class keeps them
       * resident where the PA operators and the CG solver actually work.
       *
       * Device::GetDeviceMemoryClass() is MemoryClass::HOST unless a device
       * backend was configured, so this is a no-op for CPU runs. */
      MemoryClass GetMemoryClass() const override
      { return Device::GetDeviceMemoryClass(); }

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
      /// Iterations of the linear solve actually used by the last step: the
      /// implicit M + dt K solve, or the mass solve on the explicit path.
      int GetNumIterations() const
      {
         if (implicit_time_integration) { return T_solver ? T_solver->GetNumIterations() : 0; }
         return M_solver ? M_solver->GetNumIterations() : 0;
      }
      bool GetConverged() const
      {
         if (implicit_time_integration) { return T_solver && T_solver->GetConverged(); }
         return M_solver && M_solver->GetConverged();
      }

      // Get the current solution gf
      ParGridFunction *GetPotentialGf() { return &u_gf; }

      protected:
      // Mesh and finite element space
      ParMesh *pmesh;             ///< NOT OWNED
      ParFiniteElementSpace *fes; ///< NOT OWNED
      int dim;
      int fes_truevsize;

      bool assembled = false;

      // BCHandler
      BCHandler *bcs; ///< OWNED
      Array<int> ess_tdof_list;
      
      // Enable partial assembly
      bool pa; 
      const IntegrationRule *integration_rule = nullptr;
      const IntegrationRule *mass_integration_rule = nullptr;
      /// The mass rule falls back to the common rule when none was set.
      const IntegrationRule *MassIntegrationRule() const
      { return mass_integration_rule ? mass_integration_rule : integration_rule; }

      // Bilinear/Linear Forms
      std::unique_ptr<ParLinearForm> fform;
      std::unique_ptr<ParBilinearForm> M_form;
      std::unique_ptr<ParBilinearForm> K_form;

      // ParGridFunctions
      mutable ParGridFunction u_gf;      // Current solution
      mutable ParGridFunction du_dt_gf;  // Current time derivative

      // Vectors
      mutable Vector z, b;

      // Operators
      OperatorHandle opM;
      OperatorHandle opMe;
      OperatorHandle opK;
      HypreParMatrix *Mfull = nullptr;

      real_t cached_dt = 0.0;
      int current_step = 0;

      // Preconditioner type for the PA implicit solver (0: Jacobi, 1: LOR+AMG).
      // Stored from Setup() so that BuildImplicitSolver() can honor it.
      int prec_type = 0;
      real_t lin_rel_tol = 1e-6;  // CG relative tolerance for the implicit solve
      // Warm-start the implicit CG solve from the previous step's du/dt. ~1.9x fewer
      // iterations at unchanged accuracy; see ImplicitSolverBase::SetWarmStart.
      bool warm_start = true;
      Vector du_dt_prev;

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

      // Verbosity
      bool verbose;

      // Set time for coefficients
      void SetCoefficientsTime(const real_t &time);

      // Build the implicit solver
      void BuildImplicitSolver();

      // Reassemble and setup the solver
      inline void AssembleAndSetupSolver();

      // Assemble the operators
      inline void AssembleOperators();
    };

  } // namespace electrophysiology

} // namespace mfem
