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

// Class for (Fourier) Heat Operator
//
// This class provides the operator used to solve the heat equation in HeatSolver with either explicit or implicit time integration.
// The operator discretizes the heat equation:
//
//        rho * C * du/dt = ∇•(k∇u) - α • u ∇u - β u + f
//
// into the form:
//
//        M * du_dt = -K(u) + f
//
// where:
// - M is the mass matrix, scaled by rho * C.
// - K is the stiffness matrix, defined as K = ∇•(k∇) - α • u ∇ - β = -(D + A - R).
// - f is the right-hand side, representing volumetric heat sources.
//
// The operator also handles various boundary conditions, including:
// - Dirichlet boundary conditions
// - Neumann boundary conditions (scalar, vector • norm)
// - Robin boundary conditions
//
// Key Assumptions:
// - Only the field in the advection term is time-dependent (i.e., u = u(t)).
// - The stiffness matrix K is reassembled at each time step if the advection term is provided.
//   (Note: matrix could be defined separately, but this would increase memory requirements).
//
// This class is designed to be used within the HeatSolver framework to facilitate the numerical solution of the heat equation.

#ifndef MFEM_HEAT_OPERATOR
#define MFEM_HEAT_OPERATOR

#include "../../common/mesh_extras.hpp"
#include "../../common/pfem_extras.hpp"
#include "../bc/heat_bchandler.hpp"
#include "implicit_solver.hpp"
#include "../../common-ecm2/utils.hpp"
#include <deque>

#ifdef MFEM_USE_MPI

#include <map>
#include <string>

namespace mfem
{

  using common::H1_ParFESpace;

  // Include functions from ecm2_utils namespace
  using namespace ecm2_utils;

  namespace heat
  {
    // Forward declaration of ImplicitSolverFA
    class ImplicitSolverBase;
    class ImplicitSolverFA;
    class ImplicitSolverPA;

    class AdvectionReactionDiffusionOperator : public TimeDependentOperator
    {
    protected:
      // Shared pointer to Mesh
      std::shared_ptr<ParMesh> pmesh;
      int dim;

      // BCHandler (operator takes ownership)
      BCHandler *bcs;

      ParFiniteElementSpace &H1FESpace;
      Array<int> ess_tdof_list;
      int fes_truevsize;
      bool pa; // Enable partial assembly

      ParBilinearForm *M;
      ParBilinearForm *K;
      ParBilinearForm *RobinMass;

      ParLinearForm *fform;
      Vector *fvec;

      OperatorHandle opM;
      OperatorHandle opMe;
      OperatorHandle opK;
      OperatorHandle opRobinMass; // Mass matrix with Robin BCs
      HypreParMatrix *Mfull = nullptr;

      real_t current_dt = 0.0;
      int current_step = 0;

      CGSolver M_solver; // Krylov solver for inverting the mass matrix M
      Solver *M_prec;    // Preconditioner for the mass matrix M

      bool implicit_time_integration = false; // Implicit time integration flag
      // TODO: it might make sense to have flags for time-dependent bcs and coefficients
      bool enable_reassembling = true;        // Reassemble the operator T at every iteration (can be set to false in case of constant coefficients and/or bcs)
      ImplicitSolverBase *T_solver;           // Implicit solver for T = M + dt K

      ProductCoefficient *rhoC;
      MatrixCoefficient *Kappa; // Diffusion term
      real_t alpha;             // Convection term
      VectorCoefficient *u;
      Coefficient *beta; // Reaction term

      bool has_diffusion = false;
      bool has_advection = false;
      bool has_reaction = false;

      // Bookkeeping for voumetric heat terms.
      std::vector<CoeffContainer> volumetric_terms;

      mutable Vector z; // auxiliary vector

      // Approximation of first derivative
      mutable Vector *dT_approx; // auxiliary vectors for bcs

      // verbosity
      bool verbose;

      // Set time for coefficients
      void SetCoefficientsTime(const real_t &time);   

    public:
      AdvectionReactionDiffusionOperator(std::shared_ptr<ParMesh> pmesh_, ParFiniteElementSpace &f,
                                         BCHandler *bcs,
                                         MatrixCoefficient *Kappa = nullptr,
                                         Coefficient *c_ = nullptr, Coefficient *rho_ = nullptr,
                                         real_t alpha = 0.0, VectorCoefficient *u = nullptr,
                                         real_t beta = 0.0,
                                         bool verbose = false);

      // Enable partial assembly
      void EnablePA(bool pa_ = false);

      // Disable re-assembly of bilinear form at each iteration
      void DisableReassembly() { enable_reassembling = false; }

      /** Set up the AdvectionReactionDiffusionOperator.
       * This involves adding all the necessary integrators to the linear form for
       * the rhs (volumetric terms, neumann, robin contribution)
       *
       * @note Must be called AFTER adding the volumetric terms with
       * AddVolumetricTerm()
       */
      virtual void Setup(real_t dt = 0.0, bool implicit_time_integration = false, int prec_type = 1);

      /** Update the AdvectionReactionDiffusionOperator in case of changes in Mesh. */
      void Update();

      /** Compute action of the AdvectionReactionDiffusionOperator: du_dt = M^{-1}*(-K(u)). */
      virtual void Mult(const Vector &u, Vector &du_dt) const;

      /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
          This is the only requirement for high-order SDIRK implicit integration.*/
      virtual void ImplicitSolve(const real_t dt, const Vector &u, Vector &k);

      /** Update bcs and rhs*/
      virtual void SetTime(const real_t time);

      /** Set the starting temperature for the current step */
      void SetStartingTemperature(const Vector *Tn);

      // Add Volumetric heat term
      void AddVolumetricTerm(Coefficient *coeff,
                             Array<int> &attr,
                             bool own = true);                   // Using scalar coefficient
      void AddVolumetricTerm(ScalarFuncT func, Array<int> &attr, bool own = true); // Using function

      void ProjectDirichletBCS(const real_t &time, ParGridFunction &gf);

      /// Getter
      // Get derivative approximation vector
      Vector &GetDerivativeApproximation() { return *dT_approx; }

      // Get ess_tdof_list
      Array<int> &GetEssTDofList() { return ess_tdof_list; }

      virtual ~AdvectionReactionDiffusionOperator();
    };

  } // namespace heat

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_HEAT_OPERATOR
