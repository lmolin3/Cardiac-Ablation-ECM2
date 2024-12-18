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

#ifndef MFEM_NAVIER_UNSTEADY_HPP
#define MFEM_NAVIER_UNSTEADY_HPP

#define MFEM_NAVIER_UNSTEADY_VERSION 0.1

#include "mfem.hpp"
#include "custom_bilinteg.hpp"
#include "bc/navier_bchandler.hpp"
#include "utils.hpp"
#include "mesh_extras.hpp"
#include "pfem_extras.hpp"

namespace mfem
{
// Include functions from ecm2_utils namespace
using namespace ecm2_utils;
using common::H1_ParFESpace;

namespace navier
{

/**
 * \class NavierUnsteadySolver
 * \brief Unsteady Incompressible Navier Stokes solver with (Picard) Algebraic Chorin-Temam splitting formulation with Pressure Correction (CTPC).
 *
 * This class solves the unsteady incompressible Navier-Stokes equations using an Algebraic Chorin-Temam splitting method with Pressure Correction (CTPC). The Navier-Stokes problem corresponds to the saddle point system:
 *
 *   du/dt - nu \nabla^2 u + u_{ext} \cdot \nabla u + \nabla p = f
 *   \div u = 0
 *
 * 
 * The nonlinear term is linearized using an extrapolated Picard iteration (u_{ext} \nabla u):
 * 
 *   u_{ext} = b1 u_{n} + b_2 u_{n-1} + b3 u_{n-2}
 * 
 * A variable order BDF scheme is used for time integration:
 * 
 *   du/dt = 1/dt (alpha u - u_{BDF})
 * 
 * 
 * The algebraic form of the linearized system is:
 *
 *   [  A   G  ] [v] = [ fv + u_bdf ]
 *   [  D   0  ] [p]   [     fp     ]
 *
 * Where:
 * - A = alpha/dt M + K + C
 * - D = (negative) divergence, Dt = (negative) gradient
 * - fp = - D_elim u_bc
 * - u_{BDF} = 1/dt (a1 u_{n} + a_2 u_{n-1} + a3 u_{n-2})
 *
 * The algebraic system using Chorin-Temam with pressure correction (CTPC) is:
 * 
 * [  A      0   ][  I   H G Q ]
 * [  D    D H G ][        Q    ]
 *
 * Where:
 * - Q = (D H A H Dt)^-1 (D H Dt)
 * - H = dt/alpha M^-1
 *
 * 
 * This system leads to the following segregated steps:
 *
 * 1) Velocity prediction: A u_pred = fv + 1/dt M u_bdf
 * 2) Pressure prediction: DHG p_pred = alpha/dt (D u_pred - fp)
 * 3) Pressure correction: DHG p = dt/alpha (DHAHG p_pred)
 * 4) Velocity correction: M u = M u_pred - G p_pred
 *
 * 
 * The numerical solvers for each step of the segregated scheme are as follows:
 *
 * 1) GMRES preconditioned with BoomerAMG 
 * 2) CG on operator DHGc (constrained TripleProductOperator). Inner CG for inverting M.
 *    Preconditioned with BoomerAMG of Laplacian of pressure
 * 3) ""
 * 4) CG preconditioned with BoomerAMG
 *
 * 
 * For a detailed description of this method, refer to the following references:
 *
 * [1] Saleri, F., & Veneziani, A. (2005). Pressure correction algebraic splitting methods for the incompressible Navier--Stokes equations.
 *     SIAM journal on numerical analysis, 43(1), 174-194.
 *
 * [2] Quarteroni, Alfio, Fausto Saleri, and Alessandro Veneziani. "Factorization methods for the numerical approximation of Navier–Stokes equations."
 *     Computer methods in applied mechanics and engineering 188.1-3 (2000): 505-526.
 *
 * [3] Gauthier, Alain, Fausto Saleri, and Alessandro Veneziani. "A fast preconditioner for the incompressible Navier Stokes Equations."
 *     Computing and Visualization in science 6 (2004): 105-112.
 *
 * [4] Rebholz, Leo G., Alex Viguerie, and Mengying Xiao. "Analysis of Algebraic Chorin Temam splitting for incompressible NSE and comparison to Yosida methods."
 *     Journal of Computational and Applied Mathematics 365 (2020): 112366.
 */


class NavierUnsteadySolver
{
public:

    NavierUnsteadySolver(std::shared_ptr<ParMesh> pmesh_, BCHandler *bcs, real_t kin_vis_ = -1, int uorder_=2, int porder_=1,  bool verbose_=true);

    ~NavierUnsteadySolver();

    // Add volumetric term to the rhs
    void AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr);
    void AddAccelTerm(VecFuncT *func, Array<int> &attr);

    /// Solver setup and Solution

    void SetBCHandler(BCHandler *bcs_) { bcs = bcs_; }

    /**
    * \brief Set the Solvers and Linearization parameters
    *
    * Set parameters ( @a rtol, @a atol, @a maxiter, @a print level) for solvers in the segregatd scheme.
    *
    * \param params1 struct containing parameters for velocity prediction step
    * \param params2 struct containing parameters for pressure prediction step
    * \param params3 struct containing parameters for pressure correction step
    * \param params4 struct containing parameters for velocity correction step
    *
    */
    void SetSolvers( SolverParams params1, SolverParams params2, SolverParams params3, SolverParams params4 );

    /**
    * \brief Set the maximum order to use for the BDF method.
    *
    * \param order maximum bdf order to use for time integration 1 <= order <= 3
    *
    */
    void SetMaxBDFOrder(int order_) { max_bdf_order = order_; };

    /**
    * \brief Set gamma parameter for relaxation step.
    *
    * \param gamma_ parameter for relaxation parameter [0,1] (under-relaxation)
    *
    */
    void SetGamma(real_t &gamma_){ gamma = gamma_; };

    /// Enable partial assembly for every operator.
    // (to be effective, must be set before Setup() is called)
    void EnablePA(bool pa_) { pa = pa_; };

    /**
    * \brief Finalizes setup.
    *
    *   Initialize forms, solvers and preconditioners.
    *
    * \note This method should be called only after:
    * - Setting the boundary conditions and forcing terms (AddVelDirichletBC/AddTractionBC/AddAccelTerm).
    * - Setting the Linear solvers.
    * - Setting PA
    */
    void Setup( real_t dt);


    /// Initial condition for velocity
    void SetInitialConditionVel(VectorCoefficient &u_in);
    /// Initial condition for previous velocity
    void SetInitialConditionPrevVel(VectorCoefficient &u_in);
    /// Initial condition for pressure
    void SetInitialConditionPres(Coefficient &p_in);


    /**
     * @brief Compute the solution at the next time step t+dt.
     *
     * This function computes the solution of unsteady Navier-Stokes for the next time step. 
     * The solver uses a segregated scheme (Chorin-Temam with Pressure Correction) and solves
     * the problem in he following 4 steps:
     * 
     * 1. Velocity prediction
     * 2. Pressure prediction
     * 3. Pressure correction
     * 4. Velocity correction
     *
     * Time adaptivity should be implemented outside this class, and user should run the solver
     * with the updated timestep dt.
     * 
     * @param[in, out] time The current time, which will be updated to t+dt.
     * @param[in] dt The time step size to be applied for the update.
     * @param[in] current_step The current time step number or index (used to switch BDF order).
     *
     */
    void Step(real_t &time, real_t dt, int current_step);


    /// Return a pointer to the velocity ParGridFunction.
    ParFiniteElementSpace* GetFESpaceVelocity() { return ufes; }
    /// Return a pointer to the pressure ParGridFunction.
    ParFiniteElementSpace* GetFESpacePressure() { return pfes; }

    /// Return velocity tdof vector
    Vector& GetVelocityVector() { return *u; }
    /// Return predicted velocity tdof vector
    Vector& GetPredictedVelocityVector() { return *u_pred; }
    /// Return pressure tdof vector
    Vector& GetPressureVector() { return *p; }
    /// Return predicted pressure tdof vector
    Vector& GetPredictedPressureVector() { return *p_pred; }

    /// Return the velocity GridFunction
    ParGridFunction *GetVelocity() { return u_gf; }
    /// Return the predicted velocity GridFunction
    ParGridFunction* GetPredictedVelocity() { return u_pred_gf; }
    /// Return the pressure GridFunction
    ParGridFunction* GetPressure() { return p_gf; }
    /// Return the predicted pressure GridFunction
    ParGridFunction* GetPredictedPressure() { return p_pred_gf; }

   /// Print timing summary of the solving routine.
   /**
    * The summary shows the timing in seconds in the first row of
    *
    * 1. SETUP: Time spent for the setup of all forms, solvers and preconditioners.
    * 2. ASSEMBLY CONVECTIVE MATRIX: Time spent assembling the coinvective term
    * 3. VELOCITY PREDICTION: Time spent in the velocity prediction solve.
    * 4. PRESSURE PREDICTION: Time spent in the pressure prediction solve.
    * 5. PRESSURE CORRECTION: Time spent in the pressure correction solve.
    * 6. VELOCITY CORRECTION: Time spent in the velocity correction solve.
    *
    * The second row shows a proportion of a column relative to the whole
    * time step.
    */
   void PrintTimingData();

   // Visualization and Postprocessing
   void RegisterVisItFields(VisItDataCollection &visit_dc_);

   void RegisterParaviewFields(ParaViewDataCollection &paraview_dc_);

   void AddParaviewField(const std::string &field_name, ParGridFunction *gf);

   void AddVisItField(const std::string &field_name, ParGridFunction *gf);

   void WriteFields(const int &it = 0, const real_t &time = 0);

   ParaViewDataCollection &GetParaViewDc() { return *paraview_dc; }
   VisItDataCollection &GetVisItDc() { return *visit_dc; }

   private:
   /// mesh
   std::shared_ptr<ParMesh> pmesh;
   int sdim;

   /// Velocity and Pressure FE spaces
   ParFiniteElementSpace *ufes = nullptr;
   ParFiniteElementSpace *pfes = nullptr;
   int uorder;
   int porder;
   int udim;
   int pdim;

   bool pa = false;

   /// Grid functions
   ParGridFunction *u_gf = nullptr;      // corrected velocity
   ParGridFunction *p_gf = nullptr;      // corrected pressure
   ParGridFunction *u_pred_gf = nullptr; // predicted velocity
   ParGridFunction *p_pred_gf = nullptr; // predicted pressure
   ParGridFunction *u_bdf_gf = nullptr;  // bdf velocity
   ParGridFunction *u_ext_gf = nullptr;  // extrapolated velocity

   /// Boundary conditions
   BCHandler *bcs; // Boundary Condition Handler

   Array<int> vel_ess_tdof;      // All essential velocity true dofs.
   Array<int> vel_ess_tdof_full; // All essential true dofs from VectorCoefficient.
   Array<int> vel_ess_tdof_x;    // All essential true dofs x component.
   Array<int> vel_ess_tdof_y;    // All essential true dofs y component.
   Array<int> vel_ess_tdof_z;    // All essential true dofs z component.
   Array<int> pres_ess_tdof;     // All essential pressure true dofs.

   // Bookkeeping for acceleration (forcing) terms.
   std::vector<VecCoeffContainer> accel_terms;

   /// Bilinear/linear forms
   ParBilinearForm *K_form = nullptr;
   ParBilinearForm *M_form = nullptr;
   ParMixedBilinearForm *D_form = nullptr;
   ParMixedBilinearForm *G_form = nullptr;
   ParBilinearForm *NL_form = nullptr;
   ParBilinearForm *S_form = nullptr;
   ParLinearForm *f_form = nullptr;

   /// Vectors
   Vector *u = nullptr;      // Corrected velocity
   Vector *p = nullptr;      // Corrected pressure
   Vector *u_pred = nullptr; // Predicted velocity
   Vector *p_pred = nullptr; // Predicted pressure

   Vector *u_bdf = nullptr; // BDF velocity
   Vector *u_ext = nullptr; // Extrapolated velocity (Picard linearization)

   Vector *un = nullptr;  // Corrected velocity (timestep n,   for BDF 1,2,3)
   Vector *un1 = nullptr; // Corrected velocity (timestep n-1, for BDF 2,3)
   Vector *un2 = nullptr; // Corrected velocity (timestep n-2, for BDF 3)

   Vector *fv = nullptr;     // load vector for velocity (original)
   Vector *fp = nullptr;     // load vector for pressure (modification with ess bcs)
   Vector *rhs_v1 = nullptr; // load vector for velocity prediction
   Vector *rhs_v2 = nullptr; // load vector for velocity correction
   Vector *rhs_p1 = nullptr; // load vector for pressure prediction
   Vector *rhs_p2 = nullptr; // load vector for pressure correction
   Vector *Gp = nullptr;     // product G p_pred (used in both pressure and velocity correction)

   Vector *tmp1 = nullptr; // auxiliary vectors
   Vector *tmp2 = nullptr;

   /// Wrapper for extrapolated velocity coefficient
   VectorGridFunctionCoefficient *u_ext_vc = nullptr;

   /// Matrices/operators
   OperatorHandle opK;      // velocity laplacian
   OperatorHandle opM;      // velocity mass (unmodified)
   OperatorHandle opD;      // divergence
   OperatorHandle opG;      // gradient
   OperatorHandle opNL;     // convective term   w . grad u (Picard convection)
   OperatorHandle opS; // pressure laplacian
   OperatorHandle opDe;

   HypreParMatrix *C = nullptr;      // C = alpha/dt + A
   HypreParMatrix *Ce = nullptr;       // matrix for dirichlet bc modification
   HypreParMatrix *sigmaM = nullptr; // velocity mass (modified with bdf coeff and bcs)    alpha/dt M



   TripleProductOperator *DHG = nullptr;
   ConstrainedOperator *DHGc = nullptr;

   /// Linear form to compute the mass matrix to set pressure mean to zero.
   ParLinearForm *mass_lf = nullptr;
   real_t volume = 0.0;

   /// Kinematic viscosity.
   ConstantCoefficient* kin_vis;

   // Coefficents
   ConstantCoefficient C_bdfcoeff;
   ConstantCoefficient C_visccoeff;
   ConstantCoefficient S_bdfcoeff;
   ConstantCoefficient S_visccoeff;

   // BDF coefficients
   int max_bdf_order = 3;
   real_t alpha = 0.0;
   real_t a1 = 0.0;
   real_t a2 = 0.0;
   real_t a3 = 0.0;
   real_t b1 = 0.0;
   real_t b2 = 0.0;
   real_t b3 = 0.0;

   /// Coefficient for relaxation step
   real_t gamma;

   /// Linear solvers parameters
   SolverParams s1Params;
   SolverParams s2Params;
   SolverParams s3Params;
   SolverParams s4Params;

   /// Solvers and Preconditioners
   GMRESSolver *invC = nullptr;    // solver for velocity prediction (non-symmetric system)
   GMRESSolver *invDHG1 = nullptr; // solver for pressure prediction
   GMRESSolver *invDHG2 = nullptr; // solver for pressure correction
   CGSolver *H1 = nullptr;         // solver for approximated momentum matrix in L-step (Schur Complement)
   CGSolver *H2 = nullptr;         // solver for approximated momentum matrix in U-step

   Solver *invC_pc = nullptr;           // preconditioner for velocity block
   HypreBoomerAMG *invDHG_pc = nullptr; // preconditioner for pressure schur complement
   HypreBoomerAMG *H1_pc = nullptr;     // preconditioner for velocity mass matrix

   /// Variables for iterations/norm solvers
   int iter_v1solve = 0;
   int iter_v2solve = 0;
   int iter_p1solve = 0;
   int iter_p2solve = 0;

   real_t res_v1solve = 0.0;
   real_t res_v2solve = 0.0;
   real_t res_p1solve = 0.0;
   real_t res_p2solve = 0.0;

   /// Boolean checking status of bcs/rhs (avoid unnecessary projection for time update)
   bool updatedRHS = false;
   bool updatedBCS = false;

   /// Timers
   StopWatch sw_setup;
   StopWatch sw_conv_assembly;
   StopWatch sw_vel_pred;
   StopWatch sw_pres_pred;
   StopWatch sw_pres_corr;
   StopWatch sw_vel_corr;

   /// Data Collections
   ParaViewDataCollection *paraview_dc = nullptr;
   VisItDataCollection *visit_dc = nullptr;

   /// Enable/disable verbose output.
   bool verbose;

   /// @brief Update the BDF/Ext coefficient.
   /**
    * @param step current step (to check what BDF order to use)
    *
    * The BDF scheme reads du/dt ~ (alpha u - u_{BDF})/dt
    * The Extrapolated velocity is u_ext = a1 un + a2 u_{n-1} + a3 u_{n-2}
    *
    * where:
    *
    *           { 1       bdf1                { un                                  bdf1
    * alpha   = { 3/2     bdf2   ,  u_{BDF} = { 2 un - 1/2 u_{n-1}                  bdf2
    *           { 11/6    bdf3                { 3 un - 3/2 u_{n-1} + 1/2 u_{n-2}    bdf3
    *
    *           { un                              bdf1
    * u_{ext} = { 2 un -  u_{n-1}                 bdf2
    *           { 3 un - 3 u_{n-1} + 1 u_{n-2}    bdf3
    */ 
   void SetTimeIntegrationCoefficients(int step);

    /// Update solution at previous timesteps for BDF
    void UpdateSolution();

    /**
     * @brief Update time in acceleration and traction coefficients, and re-assemble rhs vector fv.
     *
     * This function updates the time used in the calculations for acceleration and traction coefficients.
     * It also re-assembles the right-hand side (rhs) vector fv, which may depend on the updated time.
     *
     * @param new_time The new time value to be used in the coefficients.
     *
     * @note This function is responsible for updating time-related parameters in the system and
     *       re-evaluating the right-hand side vector when the time changes.
     *       Make sure to call this function whenever the time is updated in your simulation.
     */
    void UpdateTimeRHS(real_t new_time);


    /**
     * @brief Update time in Boundary Condition (Dirichlet) coefficients and project on the solution vector.
     *
     * This function updates the time used in the calculations for Boundary Condition (Dirichlet) coefficients
     * and then projects these updated coefficients onto the solution vector.
     *
     * @param new_time The new time value to be used in the calculations.
     *
     * @note This function is responsible for updating time-dependent Boundary Condition coefficients
     *       and ensuring they are correctly applied to the solution vector when the time changes.
     *       Be sure to call this function whenever the time is updated in your simulation.
     */
    void UpdateTimeBCS(real_t new_time);

    /// Remove mean from a Vector.
    /**
     * Modify the Vector @a v by subtracting its mean using
     * \f$v = v - \frac{\sum_i^N v_i}{N} \f$
     */
    void Orthogonalize(Vector &v);

    /// Remove the mean from a ParGridFunction.
    /**
     * Modify the ParGridFunction @a v by subtracting its mean using
     * \f$ v = v - \int_\Omega \frac{v}{vol(\Omega)} dx \f$.
     */
    void MeanZero(ParGridFunction &v);

    /// Print logo of the Navier solver.
    void PrintLogo();

    /// Print information about the Navier solver.
    void PrintInfo();

    /// Print matrices
#ifdef MFEM_DEBUG
    void PrintMatricesVectors( const char* id, int num );
#endif

};

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

class QuantitiesOfInterest
{
public:
   QuantitiesOfInterest(ParMesh *pmesh)
   {
      H1_FECollection h1fec(1);
      ParFiniteElementSpace h1fes(pmesh, &h1fec);

      onecoeff.constant = 1.0;
      mass_lf = new ParLinearForm(&h1fes);
      mass_lf->AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
      mass_lf->Assemble();

      ParGridFunction one_gf(&h1fes);
      one_gf.ProjectCoefficient(onecoeff);

      volume = mass_lf->operator()(one_gf);
   };

   ~QuantitiesOfInterest() { delete mass_lf; };

real_t ComputeCFL(ParGridFunction &u, real_t dt)
{
   ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
   FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim();

   Vector ux, uy, uz;
   Vector ur, us, ut;
   real_t cflx = 0.0;
   real_t cfly = 0.0;
   real_t cflz = 0.0;
   real_t cflm = 0.0;
   real_t cflmax = 0.0;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               fe->GetOrder());
      ElementTransformation *tr = fes->GetElementTransformation(e);

      u.GetValues(e, ir, ux, 1);
      ur.SetSize(ux.Size());
      u.GetValues(e, ir, uy, 2);
      us.SetSize(uy.Size());
      if (vdim == 3)
      {
         u.GetValues(e, ir, uz, 3);
         ut.SetSize(uz.Size());
      }

      real_t hmin = pmesh_u->GetElementSize(e, 1) /
                    (real_t) fes->GetElementOrder(0);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         tr->SetIntPoint(&ip);
         const DenseMatrix &invJ = tr->InverseJacobian();
         const real_t detJinv = 1.0 / tr->Jacobian().Det();

         if (vdim == 2)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)) * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)) * detJinv;
         }
         else if (vdim == 3)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)
                     + uz(i) * invJ(2, 0))
                    * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)
                     + uz(i) * invJ(2, 1))
                    * detJinv;
            ut(i) = (ux(i) * invJ(0, 2) + uy(i) * invJ(1, 2)
                     + uz(i) * invJ(2, 2))
                    * detJinv;
         }

         cflx = fabs(dt * ux(i) / hmin);
         cfly = fabs(dt * uy(i) / hmin);
         if (vdim == 3)
         {
            cflz = fabs(dt * uz(i) / hmin);
         }
         cflm = cflx + cfly + cflz;
         cflmax = fmax(cflmax, cflm);
      }
   }

   real_t cflmax_global = 0.0;
   MPI_Allreduce(&cflmax,
                 &cflmax_global,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_MAX,
                 pmesh_u->GetComm());

   return cflmax_global;
}


   real_t ComputeKineticEnergy(ParGridFunction &v)
   {
      Vector velx, vely, velz;
      real_t integ = 0.0;
      const FiniteElement *fe;
      ElementTransformation *T;
      FiniteElementSpace *fes = v.FESpace();

      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         int intorder = 2 * fe->GetOrder();
         const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

         v.GetValues(i, *ir, velx, 1);
         v.GetValues(i, *ir, vely, 2);
         v.GetValues(i, *ir, velz, 3);

         T = fes->GetElementTransformation(i);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);

            real_t vel2 = velx(j) * velx(j) + vely(j) * vely(j)
                          + velz(j) * velz(j);

            integ += ip.weight * T->Weight() * vel2;
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&integ,
                    &global_integral,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      return 0.5 * global_integral / volume;
   };

// Computes Q = 0.5*(tr(\nabla u)^2 - tr(\nabla u \cdot \nabla u))
void ComputeQCriterion(ParGridFunction &u, ParGridFunction &q)
{
   FiniteElementSpace *v_fes = u.FESpace();
   FiniteElementSpace *fes = q.FESpace();

   // AccumulateAndCountZones
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   q = 0.0;

   // Local interpolation
   int elndofs;
   Array<int> v_dofs, dofs;
   Vector vals;
   Vector loc_data;
   int vdim = v_fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, dofs);
      v_fes->GetElementVDofs(e, v_dofs);
      u.GetSubVector(v_dofs, loc_data);
      vals.SetSize(dofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval
         // GetVectorGradientHat
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         real_t q_val = 0.5 * (sq(grad(0, 0)) + sq(grad(1, 1)) + sq(grad(2, 2)))
                        + grad(0, 1) * grad(1, 0) + grad(0, 2) * grad(2, 0)
                        + grad(1, 2) * grad(2, 1);

         vals(dof) = q_val;
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < dofs.Size(); j++)
      {
         int ldof = dofs[j];
         q(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication

   // Count the zones globally.
   GroupCommunicator &gcomm = q.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(q.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(q.GetData());

   // Compute means
   for (int i = 0; i < q.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         q(i) /= nz;
      }
   }
}

private:
   ConstantCoefficient onecoeff;
   ParLinearForm *mass_lf;
   real_t volume;

    template<typename T>
    T sq(T x)
    {
    return x * x;
    }

};






#endif // MFEM_NAVIER_UNSTEADY_HPP



} // namespace navier

} // namespace mfem
