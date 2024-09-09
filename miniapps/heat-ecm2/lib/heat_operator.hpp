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

#ifndef MFEM_HEAT_OPERATOR
#define MFEM_HEAT_OPERATOR

#include "mesh_extras.hpp"
#include "pfem_extras.hpp"
#include "bc/heat_bchandler.hpp"
#include "utils.hpp"
#include <deque>

#ifdef MFEM_USE_MPI

#include <map>
#include <string>

namespace mfem {

using common::H1_ParFESpace;

// Include functions from ecm2_utils namespace
using namespace ecm2_utils;

namespace heat {
// Forward declaration of ImplicitSolver
class ImplicitSolver;
class ConductionOperator : public TimeDependentOperator {
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
  HypreParMatrix *Mfull;

  double current_dt = 0.0;
  int current_step = 0;

  CGSolver M_solver; // Krylov solver for inverting the mass matrix M
  Solver *M_prec;    // Preconditioner for the mass matrix M

  ImplicitSolver *T_solver; // Implicit solver for T = M + dt K

  PWMatrixCoefficient *Kappa;
  ProductCoefficient *rhoC;

  // Bookkeeping for voumetric heat terms.
  std::vector<CoeffContainer> volumetric_terms;

  mutable Vector z; // auxiliary vector

  // Approximation of first derivative
  mutable Vector *dT_approx; // auxiliary vectors for bcs

public:
  ConductionOperator(std::shared_ptr<ParMesh> pmesh_, ParFiniteElementSpace &f,
                     BCHandler *bcs, PWMatrixCoefficient *Kappa_,
                     Coefficient *c_, Coefficient *rho_);

  // Enable partial assembly
  void EnablePA(bool pa_ = false);

  /** Set up the ConductionOperator.
   * This involves adding all the necessary integrators to the linear form for
   * the rhs (volumetric terms, neumann, robin contribution)
   *
   * @note Must be called AFTER adding the volumetric terms with
   * AddVolumetricTerm()
   */
  virtual void Setup();

  /** Update the ConductionOperator in case of changes in Mesh. */
  void Update();

  /** Compute action of the ConductionOperator: du_dt = M^{-1}*(-K(u)). */
  virtual void Mult(const Vector &u, Vector &du_dt) const;

  /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
      This is the only requirement for high-order SDIRK implicit integration.*/
  virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k);

  /** Update bcs and rhs*/
  virtual void SetTime(const double time);

  /** Set the time step */
  void SetTimeStep(double dt);

  /** Set the starting temperature for the current step */
  void SetStartingTemperature(const Vector *Tn);

  // Add Volumetric heat term
  void AddVolumetricTerm(Coefficient *coeff,
                         Array<int> &attr); // Using scalar coefficient
  void AddVolumetricTerm(ScalarFuncT func, Array<int> &attr); // Using function

  void ProjectDirichletBCS(const double &time, ParGridFunction &gf);

  /// Getter
  // Get derivative approximation vector
  Vector &GetDerivativeApproximation() { return *dT_approx; }

  // Get ess_tdof_list
  Array<int> &GetEssTDofList() { return ess_tdof_list; }

  virtual ~ConductionOperator();
};

class ImplicitSolver : public Solver {
private:
  HypreParMatrix *M, *K, *RobinMassMat;
  HypreParMatrix *T, *Te;
  CGSolver *linear_solver;
  HypreSmoother *prec;
  double current_dt;
  bool finalized;
  Array<int> ess_tdof_list;

public:
  ImplicitSolver(HypreParMatrix *M_, HypreParMatrix *K_,
                 Array<int> &ess_tdof_list_);

  void SetOperator(const Operator &op);

  void SetTimeStep(double dt_);

  void Reset();

  void BuildOperator(HypreParMatrix *M_, HypreParMatrix *K_,
                     HypreParMatrix *RobinMass_ = nullptr);

  void EliminateBC(const Vector &x, Vector &b) const;

  virtual void Mult(const Vector &x, Vector &y) const;

  bool IsFinalized() const;

  ~ImplicitSolver();
};

} // namespace heat

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_HEAT_OPERATOR
