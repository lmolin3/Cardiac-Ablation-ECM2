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

    NavierUnsteadySolver(ParMesh* mesh,int uorder=2, int porder=1, double kin_vis_=0.0, bool verbose=true);

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
    * \brief Add Dirichlet pressure BC using Coefficient and list of essential mesh attributes.
    *
    * Add a Dirichlet pressure boundary condition to internal list of essential bcs passing Coefficient
    * and list of essential mesh attributes (they will be applied at setup time).
    *
    * \param coeff Pointer to Coefficient
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddPresDirichletBC(Coefficient *coeff, Array<int> &attr);

    /**
    * \brief Add Dirichlet pressure BC using Scalar function and list of essential mesh attributes.
    *
    * Add a Dirichlet pressure boundary condition to internal list of essential bcs passing Vector function
    * and list of essential mesh attributes (they will be applied at setup time).
    *
    * \param func Pointer to ScalarFuncT
    * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
    *
    */
    void AddPresDirichletBC(ScalarFuncT *func, Array<int> &attr);

    /**
    * \brief Add Dirichlet pressure BC using Coefficient and specific mesh attribute.
    *
    * Add a Dirichlet pressure boundary condition to internal list of essential bcs passing Coefficient,
    * and integer for specific mesh attribute (they will be applied at setup time).
    *
    * \param coeff Pointer to Coefficient
    * \param attr Boundary attribute
    *
    */
    void AddPresDirichletBC(Coefficient *coeff, int &attr);

    /**
    * \brief Add Dirichlet pressure BC passing ScalarFuncT and specific mesh attribute.
    *
    * Add a Dirichlet pressure boundary condition to internal list of essential bcs passing ScalarFuncT
    * and integer for specific mesh attribute (they will be applied at setup time).
    *
    * \param func Pointer to ScalarFuncT
    * \param attr Boundary attribute
    *
    */
    void AddPresDirichletBC(ScalarFuncT *func, int &attr);

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
    * \note This method should be called only after:
    * - Setting the boundary conditions and forcing terms (AddVelDirichletBC/AddTractionBC/AddAccelTerm).
    * - Setting the Linear solvers.
    */
    void Setup( double dt);


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
        return *u;
    }

    /**
    * \brief Returns the predicted velocity solution vector.
    */
    Vector& GetPredictedVelocityVector()
    {
        return *u_pred;
    }

    /**
    * \brief Returns the corrected pressure solution vector.
    */
    Vector& GetPressureVector()
    {
        return *p;
    }

    /**
    * \brief Returns the predicted pressure solution vector.
    */
    Vector& GetPredictedPressureVector()
    {
        return *p_pred;
    }

    /**
    * \brief Returns the velocity solution (GridFunction).
    */
    ParGridFunction& GetVelocity()
    {
        u_gf->SetFromTrueDofs(*u);
        return *u_gf;
    }

    /**
    * \brief Returns the predicted velocity solution (GridFunction).
    */
    ParGridFunction& GetPredictedVelocity()
    {
        u_pred_gf->SetFromTrueDofs(*u_pred);
        return *u_pred_gf;
    }

    /**
    * \brief Returns the pressure solution (GridFunction).
    */
    ParGridFunction& GetPressure()
    {
        p_gf->SetFromTrueDofs(*p);
        return *p_gf;
    }

    /**
    * \brief Returns the predicted pressure solution (GridFunction).
    */
    ParGridFunction& GetPredictedPressure()
    {
        p_pred_gf->SetFromTrueDofs(*p_pred);
        return *p_pred_gf;
    }


    // Other methods

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

    /// Grid functions
    ParGridFunction      *u_gf = nullptr;     // corrected velocity
    ParGridFunction      *p_gf = nullptr;     // corrected pressure
    ParGridFunction *u_pred_gf = nullptr;     // predicted velocity
    ParGridFunction *p_pred_gf = nullptr;     // predicted pressure
    ParGridFunction  *u_bdf_gf = nullptr;     // bdf velocity
    ParGridFunction  *u_ext_gf = nullptr;     // extrapolated velocity

    /// Dirichlet conditions
    Array<int> vel_ess_attr;          // Essential mesh attributes (full velocity applied).
    Array<int> vel_ess_attr_x;        // Essential mesh attributes (x component applied).
    Array<int> vel_ess_attr_y;        // Essential mesh attributes (y component applied).
    Array<int> vel_ess_attr_z;        // Essential mesh attributes (z component applied).
    Array<int> ess_attr_tmp;          // Temporary variable for essential mesh attributes.
    Array<int> trac_attr_tmp;         // Temporary variable for traction mesh attributes.
    Array<int> traction_attr;         // Traction mesh attributes.   
    Array<int> custom_traction_attr;  // Custom traction mesh attributes.
    Array<int> pres_ess_attr;         // Essential mesh attributes for pressure.

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
    ParMixedBilinearForm *D_form  = nullptr;
    ParMixedBilinearForm *G_form  = nullptr;
    ParBilinearForm      *NL_form  = nullptr;
    ParBilinearForm      *S_form  = nullptr;
    ParLinearForm        *f_form  = nullptr;

    /// Vectors
    Vector        *u = nullptr;   // Corrected velocity
    Vector        *p = nullptr;   // Corrected pressure
    Vector   *u_pred = nullptr;   // Predicted velocity
    Vector   *p_pred = nullptr;   // Predicted pressure
    Vector       *z1 = nullptr;   // Pressure update

    Vector   *u_bdf  = nullptr;   // BDF velocity
    Vector   *u_ext  = nullptr;   // Extrapolated velocity (Picard linearization)

    Vector   *un  = nullptr;      // Corrected velocity (timestep n,   for BDF 1,2,3)
    Vector   *un1 = nullptr;      // Corrected velocity (timestep n-1, for BDF 2,3)
    Vector   *un2 = nullptr;      // Corrected velocity (timestep n-2, for BDF 3)

    Vector   *fv     = nullptr;   // load vector for velocity (original)
    Vector   *fp     = nullptr;   // load vector for pressure (modification with ess bcs)
    Vector   *rhs_v1 = nullptr;   // load vector for velocity prediction 
    Vector   *rhs_v2 = nullptr;   // load vector for velocity correction
    Vector   *rhs_p1 = nullptr;   // load vector for pressure prediction
    Vector   *rhs_p2 = nullptr;   // load vector for pressure correction
    Vector       *Gp = nullptr;   // product G p_pred (used in both pressure and velocity correction)

    Vector     *tmp1 = nullptr;   // auxiliary vectors 
    Vector     *tmp2 = nullptr;   

    /// Wrapper for extrapolated velocity coefficient 
    VectorGridFunctionCoefficient *u_ext_vc = nullptr;

    /// Matrices/operators
    HypreParMatrix           *K = nullptr;         // velocity laplacian 
    HypreParMatrix           *M = nullptr;         // velocity mass (unmodified)
    HypreParMatrix      *sigmaM = nullptr;         // velocity mass (modified with bdf coeff and bcs)    alpha/dt M
    HypreParMatrix           *D = nullptr;         // divergence
    HypreParMatrix           *G = nullptr;         // gradient
    HypreParMatrix           *A = nullptr;         // A = kin_vis K + NL
    HypreParMatrix           *C = nullptr;         // C = alpha/dt + A
    HypreParMatrix           *NL = nullptr;        // convective term   w . grad u (Picard convection)

    HypreParMatrix          *S = nullptr;          // pressure laplacian

    HypreParMatrix         *Ce  = nullptr;         // matrices for dirichlet bc modification 
    HypreParMatrix         *De  = nullptr;

    TripleProductOperator *DHG = nullptr;
    ConstrainedOperator  *DHGc = nullptr;

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
    SolverParams s1Params;
    SolverParams s2Params;
    SolverParams s3Params;
    SolverParams s4Params;

    /// Solvers and Preconditioners
    GMRESSolver   *invC     = nullptr;      // solver for velocity prediction (non-symmetric system)
    GMRESSolver   *invDHG1  = nullptr;      // solver for pressure prediction
    GMRESSolver   *invDHG2  = nullptr;      // solver for pressure correction
    CGSolver         *H1    = nullptr;      // solver for approximated momentum matrix in L-step (Schur Complement)
    CGSolver         *H2    = nullptr;      // solver for approximated momentum matrix in U-step 

    Solver           *invC_pc     = nullptr;   // preconditioner for velocity block
    HypreBoomerAMG   *invDHG_pc   = nullptr;   // preconditioner for pressure schur complement 
    HypreBoomerAMG   *H1_pc       = nullptr;   // preconditioner for velocity mass matrix

    /// Variables for iterations/norm solvers
    int iter_v1solve = 0;
    int iter_v2solve = 0;
    int iter_p1solve = 0;
    int iter_p2solve = 0;

    double res_v1solve = 0.0;
    double res_v2solve = 0.0;
    double res_p1solve = 0.0;
    double res_p2solve = 0.0;

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

