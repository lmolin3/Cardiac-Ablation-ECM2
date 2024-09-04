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
#include "integrators/custom_bilinteg.hpp"
#include "utils.hpp"
#include <sys/stat.h>  // Include for mkdir

namespace mfem
{
// Include functions from ecm2_utils namespace
using namespace ecm2_utils;

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
 *   [  C   G  ] [v] = [ fv +  M u_bdf ]
 *   [  D   0  ] [p]   [       fp      ]
 *
 * Where:
 * - C = alpha/dt M + K + NL
 * - D = (negative) divergence, G = gradient
 * - fp = - D_elim u_bc
 * - u_{BDF} = 1/dt (a1 u_{n} + a_2 u_{n-1} + a3 u_{n-2})
 *
 * 
 * The problem is solved with GMRES acting on the BlockOperator, preconditioned with the following block diagonal preconditioner:
 *
 *   P = [  Chat                          ]     where Mp = pressure mass matrix
 *       [        (1/kin_vis Mp + dt Sp)  ]           Sp = pressure laplacian
 * 
 */


class NavierUnsteadySolver
{
public:

    NavierUnsteadySolver(ParMesh *mesh,int uorder=2, int porder=1, double kin_vis_=0.0, bool verbose=true);

    ~NavierUnsteadySolver();


    /// Boundary conditions/Forcing terms  
    /**
    * \brief Add Dirichlet velocity BC using VectorCoefficient and list of essential mesh attributes.
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing VectorCoefficient
    * and list of essential mesh attributes (they will be applied at setup time).
    *
    * \param coeff Pointer to VectorCoefficient
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr);

    /**
    * \brief Add Dirichlet velocity BC using Vector function and list of essential mesh attributes.
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing Vector function
    * and list of essential mesh attributes (they will be applied at setup time).
    *
    * \param func Pointer to VecFuncT
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddVelDirichletBC(VecFuncT *func, Array<int> &attr);

    /**
    * \brief Add Dirichlet velocity BC componentwise using Coefficient and list of active mesh boundaries.
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing
    * Coefficient, list of essential mesh attributes, and constrained component (they will be applied at setup time).
    *
    * \param coeff Pointer to Coefficient
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    * \param dir Component of bc constrained (0=x, 1=y, 2=z)
    *
    */
    void AddVelDirichletBC(Coefficient *coeff, Array<int> &attr, int &dir);

    /**
    * \brief Add Dirichlet velocity BC using VectorCoefficient and specific mesh attribute.
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing VectorCoefficient,
    * and integer for specific mesh attribute (they will be applied at setup time).
    *
    * \param coeff Pointer to VectorCoefficient
    * \param attr Boundary attribute
    *
    */
    void AddVelDirichletBC(VectorCoefficient *coeff, int &attr);

    /**
    * \brief Add Dirichlet velocity BC passing VecFuncT and specific mesh attribute.
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing VecFuncT
    * and integer for specific mesh attribute (they will be applied at setup time).
    *
    * \param func Pointer to VecFuncT
    * \param attr Boundary attribute
    *
    */
    void AddVelDirichletBC(VecFuncT *func, int &attr);

    /**
    * \brief Add Dirichlet velocity BC componentwise passing coefficient and specific mesh attribute.
    *
    * Add a Dirichlet velocity boundary condition to internal list of essential bcs, passing
    * Coefficient, specific mesh attribute, and constrained component (they will be applied at setup time).
    *
    * \param coeff Pointer to Coefficient
    * \param attr Boundary attribute
    * \param dir Component of bc constrained (0=x, 1=y, 2=z)
    *
    * \note dir=2 only if mesh is three dimensional.
    *
    */
    void AddVelDirichletBC(Coefficient *coeff, int &attr, int &dir);

    /**
    * \brief Add Traction (Neumann) BC using VectorCoefficient and list of essential boundaries.
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
    * using VectorCoefficient, and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
    *
    * \param coeff Pointer to VectorCoefficient
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddTractionBC(VectorCoefficient *coeff, Array<int> &attr);

    /**
    * \brief Add Traction (Neumann) BC using VecFuncT and list of essential boundaries.
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
    * using VecFuncT and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
    *
    * \param coeff Pointer to VectorCoefficient
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddTractionBC(VecFuncT *coeff, Array<int> &attr);

    /**
    * \brief Add Traction (Neumann) BC using VectorCoefficient and specific mesh attribute.
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
    * using VectorCoefficient, and specific mesh attribute (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
    *
    * \param coeff Pointer to VectorCoefficient
    * \param attr Boundary attribute
    *
    */
    void AddTractionBC(VectorCoefficient *coeff, int &attr);

    /**
    * \brief Add Traction (Neumann) BC using VecFuncT and specific mesh attribute.
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
    * using VecFuncT and specific mesh attribute(they will be applied at setup time by adding BoundaryIntegrators to the rhs).
    *
    * \param func Pointer to VecFuncT
    * \param attr Boundary attribute
    *
    */
    void AddTractionBC(VecFuncT *func, int &attr);

    /**
    * \brief Add Traction (Neumann) BC computed from vector field u and scalar field p, and list of essential boundaries.
    *        Neumann: (alpha n.grad(u) + beta p.n, v)
    * 
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
    * using list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
    * The boundary contribution is computed with a VectorBoundaryLFIntegrator (psi,v), with vector computed as
    * 
    *     psi = (alpha n.grad(u) + beta p.n, v)
    *
    * \param alpha Coefficient multiplying vector field term
    * \param u     ParGridFunction for vector field
    * \param beta  Coefficient multiplying scalard field term
    * \param p     ParGridFunction for scalar field
    * \param attr  Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddCustomTractionBC(Coefficient *alpha, ParGridFunction *u, Coefficient *beta, ParGridFunction *p, Array<int> &attr);

    /**
    * \brief Add Traction (Neumann) BC computed from vector field u and scalar field p, and specific mesh attribute.
    *        Neumann: (alpha n.grad(u) + beta p.n, v)
    * 
    *
    * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
    * using list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
    * The boundary contribution is computed with a VectorBoundaryLFIntegrator (psi,v), with vector computed as
    * 
    *     psi = (alpha n.grad(u) + beta p.n, v)
    *
    * \param alpha Coefficient multiplying vector field term
    * \param u     ParGridFunction for vector field
    * \param beta  Coefficient multiplying scalard field term
    * \param p     ParGridFunction for scalar field
    * \param attr  Mesh attribute
    *
    */
    void AddCustomTractionBC(Coefficient *alpha, ParGridFunction *u, Coefficient *beta, ParGridFunction *p, int &attr);

    /**
    * \brief Add forcing term to the rhs.
    *
    * Add a forcing term (acceleration) to internal list of acceleration terms (they will be applied at setup time by adding DomainIntegrators to the rhs).
    *
    * \param coeff Pointer to VectorCoefficient
    * \param attr Domain attributes
    *
    */
    void AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr);

    /**
    * \brief Add forcing term to the rhs passing VecFuncT.
    *
    * Add a forcing term (acceleration) to internal list of acceleration terms, passing
    * VecFuncT and list of domain attributes (they will be applied at setup time by adding DomainIntegrators to the rhs).
    *
    * \param coeff Pointer to VectorCoefficient
    * \param attr Domain attributes
    *
    */
    void AddAccelTerm(VecFuncT *func, Array<int> &attr);

    /// Solver setup and Solution

    /**
    * \brief Set the Solvers and Linearization parameters
    *
    * Set parameters ( @a rtol, @a atol, @a maxiter, @a print level) for solvers in the segregatd scheme.
    *
    * \param params struct containing parameters for velocity prediction step
    *
    */
    void SetSolver( SolverParams params);

    /**
    * \brief Set the maximum order to use for the BDF method.
    *
    * \param order maximum bdf order to use for time integration 1 <= order <= 3
    *
    */
    void SetMaxBDFOrder(int order) { max_bdf_order = order; };

    /**
    * \brief Set gamma parameter for relaxation step.
    *
    * \param gamma_ parameter for relaxation parameter [0,1] (under-relaxation)
    *
    */
    void SetGamma(double &gamma){ gamma = gamma; };

    /**
    * \brief Set flag to enable saving vectors/matrices at each iteration (only in debug).
    *
    * \note this can be very time/memory consuming!
    *
    */
    void SetExportData(bool ExportData_ ) { ExportData = ExportData_; };

    /**
    * \brief Finalizes setup.
    *
    * Finalizes setup of NavierStokes solver: initialize (forms, linear solvers, and preconditioners), performs assembly.
    *
    * @param[in] dt The time step size to be applied for the update.
    * 
    * \note This method should be called only after:
    * - Setting the boundary conditions and forcing terms (AddVelDirichletBC/AddTractionBC/AddAccelTerm).
    * - Setting the Linear solvers.
    */
    void Setup(double dt);


    /**
    * \brief Set the initial condition for Velocity.
    *
    */
    void SetInitialConditionVel(VectorCoefficient &u_in);

    /**
    * \brief Set the initial condition for Velocity at previous step.
    *
    */
    void SetInitialConditionPrevVel(VectorCoefficient &u_in);

    /**
    * \brief Set the initial condition for Pressure.
    *
    */
    void SetInitialConditionPres(Coefficient &p_in);
    

    /**
     * @brief Compute the solution at the next time step t+dt.
     *
     * This function computes the solution of unsteady Navier-Stokes for the next time step. 
     * The solver uses a monolithic scheme.
     *
     * Time adaptivity should be implemented outside this class, and user should run the solver
     * with the updated timestep dt.
     * 
     * @param[in, out] time The current time, which will be updated to t+dt.
     * @param[in] dt The time step size to be applied for the update.
     * @param[in] current_step The current time step number or index (used to switch BDF order).
     *
     */
    void Step(double &time, double dt, int current_step);


    /// Getter methods

    /**
    * \brief Returns pointer to the velocity FE space.
    */
    ParFiniteElementSpace* GetUFes()
    {
        return ufes;
    }

    /**
    * \brief Returns pointer to the pressure FE space.
    */
    ParFiniteElementSpace* GetPFes()
    {
        return pfes;
    }

    /**
    * \brief Returns the corrected velocity solution vector.
    */
    Vector& GetVelocityVector()
    {
        return x->GetBlock(0);
    }

    /**
    * \brief Returns the corrected pressure solution vector.
    */
    Vector& GetPressureVector()
    {
        return x->GetBlock(1);
    }


    /**
    * \brief Returns the velocity solution (GridFunction).
    */
    ParGridFunction& GetVelocity()
    {
        u_gf->SetFromTrueDofs(x->GetBlock(0));
        return *u_gf;
    }


    /**
    * \brief Returns the pressure solution (GridFunction).
    */
    ParGridFunction& GetPressure()
    {
        p_gf->SetFromTrueDofs(x->GetBlock(1));
        return *p_gf;
    }


    // Other methods

   /// Print timing summary of the solving routine.
   /**
    * The summary shows the timing in seconds in the first row of
    *
    * 1. SETUP: Time spent for the setup of all forms, solvers and preconditioners.
    * 2. ASSEMBLY CONVECTIVE MATRIX: Time spent assembling the coinvective term
    * 3. SOLVE: Time spent in solving the linear system.
    *
    * The second row shows a proportion of a column relative to the whole
    * time step.
    */
   void PrintTimingData();


   void SetOutputFolder(const char* folderPath){outfolder = folderPath;};

private:
    /// mesh
    ParMesh* pmesh = nullptr;
    int dim;

    /// Velocity and Pressure FE spaces
    FiniteElementCollection* ufec = nullptr;
    FiniteElementCollection* pfec = nullptr;
    ParFiniteElementSpace*   ufes = nullptr;
    ParFiniteElementSpace*   pfes = nullptr;
    int uorder;
    int porder;
    int udim;
    int pdim;
    Array<int> block_offsets; // number of variables + 1

    /// Grid functions
    ParGridFunction      *u_gf = nullptr;     // corrected velocity
    ParGridFunction      *p_gf = nullptr;     // corrected pressure
    ParGridFunction  *u_bdf_gf = nullptr;     // bdf velocity
    ParGridFunction  *u_ext_gf = nullptr;     // extrapolated velocity

    /// Dirichlet conditions
    Array<int> vel_ess_attr;          // Essential mesh attributes (full velocity applied).
    Array<int> vel_ess_attr_x;        // Essential mesh attributes (x component applied).
    Array<int> vel_ess_attr_y;        // Essential mesh attributes (y component applied).
    Array<int> vel_ess_attr_z;        // Essential mesh attributes (z component applied).
    Array<int> traction_attr;         // Traction mesh attributes.
    Array<int> custom_traction_attr;  // Custom traction mesh attributes.
    Array<int> ess_attr_tmp;          // Temporary variable for essential mesh attributes.
    Array<int> trac_attr_tmp;         // Temporary variable for traction mesh attributes.

    Array<int> vel_ess_tdof;          // All essential velocity true dofs.
    Array<int> vel_ess_tdof_full;     // All essential true dofs from VectorCoefficient.
    Array<int> vel_ess_tdof_x;        // All essential true dofs x component.
    Array<int> vel_ess_tdof_y;        // All essential true dofs y component.
    Array<int> vel_ess_tdof_z;        // All essential true dofs z component.
    Array<int> pres_ess_tdof;         // All essential pressure true dofs.

    // Bookkeeping for velocity dirichlet bcs (full Vector coefficient).
    std::vector<VecCoeffContainer> vel_dbcs;

    // Bookkeeping for velocity dirichlet bcs (componentwise).
    std::string dir_string;    // string for direction name for printing output
    std::vector<CompCoeffContainer> vel_dbcs_xyz;

    // Bookkeeping for pressure dirichlet bcs (scalar coefficient).
    std::vector<CoeffContainer> pres_dbcs;

    // Bookkeeping for traction (neumann) bcs.
    std::vector<VecCoeffContainer> traction_bcs;

    // Bookkeeping for custom traction (neumann) bcs.
    std::vector<CustomNeumannContainer> custom_traction_bcs;

    // Bookkeeping for acceleration (forcing) terms.
    std::vector<VecCoeffContainer> accel_terms;

    /// Bilinear/linear forms
    ParBilinearForm      *K_form  = nullptr;
    ParBilinearForm      *M_form  = nullptr;
    ParBilinearForm     *NL_form  = nullptr;
    ParMixedBilinearForm *D_form  = nullptr;
    ParBilinearForm      *S_form  = nullptr;
    ParLinearForm        *f_form  = nullptr;

    /// Vectors
    BlockVector*   x = nullptr;
    BlockVector* rhs = nullptr;

    Vector   *u_bdf  = nullptr;   // BDF velocity
    Vector   *u_ext  = nullptr;   // Extrapolated velocity (Picard linearization)

    Vector   *un  = nullptr;      // Corrected velocity (timestep n,   for BDF 1,2,3)
    Vector   *un1 = nullptr;      // Corrected velocity (timestep n-1, for BDF 2,3)
    Vector   *un2 = nullptr;      // Corrected velocity (timestep n-2, for BDF 3)

    Vector   *fv     = nullptr;   // load vector for velocity (original)
    Vector   *fp     = nullptr;   // load vector for pressure (modification with ess bcs)

    /// Wrapper for extrapolated velocity coefficient 
    VectorGridFunctionCoefficient *u_ext_vc = nullptr;

    /// Matrices/operators
    HypreParMatrix           *M = nullptr;         // velocity mass
    HypreParMatrix           *K = nullptr;         // velocity laplacian
    HypreParMatrix           *D = nullptr;         // divergence
    HypreParMatrix           *G = nullptr;         // gradient
    HypreParMatrix           *NL = nullptr;        // convective term (picard extrapolated)
    HypreParMatrix           *C = nullptr;         // C = alpha/dt + kin_vis K + NL
    HypreParMatrix           *A = nullptr;         // A = kin_vis K + NL

    HypreParMatrix          *S = nullptr;          // preconditioner for pressure schur complement (1/kin_vis Mp + dt Sp)

    HypreParMatrix         *Ce  = nullptr;         // matrices for dirichlet bc modification 
    HypreParMatrix         *De  = nullptr;

    /// Linear form to compute the mass matrix to set pressure mean to zero.
    ParLinearForm *mass_lf = nullptr;
    double volume = 0.0;

    /// Kinematic viscosity.
    double kin_vis;

    // Coefficents
    ConstantCoefficient C_bdfcoeff;
    ConstantCoefficient C_visccoeff;
    ConstantCoefficient S_bdfcoeff;
    ConstantCoefficient S_visccoeff;

    // BDF coefficients
    int max_bdf_order = 3;
    double alpha = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double a3 = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double b3 = 0.0;

    /// Coefficient for relaxation step
    double gamma;

    /// Linear solvers parameters
    SolverParams sParams;

    /// Solvers and Preconditioners
    GMRESSolver  *nsSolver = nullptr;               // solver for Navier Stokes system
    BlockOperator    *nsOp = nullptr;               // Navier-Stokes block operator
    BlockDiagonalPreconditioner* nsPrec = nullptr;  // diagonal block preconditioner

    HypreBoomerAMG    *invC = nullptr;      // preconditioner for velocity block
    HypreBoomerAMG    *invS = nullptr;      // approximation for Schur Complement

    /// Variables for iterations/norm solvers
    int iter_solve = 0;
    double res_solve = 0.0;

    /// Boolean checking status of bcs/rhs (avoid unnecessary projection for time update)
    bool updatedRHS = false;
    bool updatedBCS = false;

    /// Timers 
    StopWatch sw_setup;
    StopWatch sw_conv_assembly;
    StopWatch sw_solve;


    /// Enable/disable verbose output.
    bool verbose;

    /// Enable/disable vectors/matrices output.
    bool ExportData;

    /// Exit flag   TODO: modify flag depending on state and make output of FSolve integer
    int flag;

    /// outfolder for debug
    const char* outfolder = nullptr;

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
     *           { 11/6    bdf3                { 3 un - 3/2 u_{n-1} + 1/3 u_{n-2}    bdf3
     * 
     *           { un                              bdf1
     * u_{ext} = { 2 un -  u_{n-1}                 bdf2
     *           { 3 un - 3 u_{n-1} + u_{n-2}    bdf3
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
    void UpdateTimeRHS(double new_time);


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
    void UpdateTimeBCS(double new_time);


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

#endif

}

