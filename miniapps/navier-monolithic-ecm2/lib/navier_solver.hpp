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

// MFEM headers
#include "mfem.hpp"

// From commons-ecm2
#include "utils.hpp"

// From navier-monolithic-ecm2
#include "navier_preconditioners.hpp"
#include "schur_preconditioners.hpp"
#include "navier_qoi.hpp"
#include "integrators/custom_bilinear_integrators.hpp"
#include "integrators/custom_linear_integrators.hpp"
#include "bc/navier_bchandler.hpp"
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
         * \class MonolithicNavierSolver
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
         * [  A      0   ][  I   H2 G R ]
         * [  D    D H1 G ][        Q   ]
         *
         * Where:
         * - H1 = dt/alpha M^-1 for Chorin-Temam and Yosida
         * - H2 = dt/alpha M^-1 for Chorin-Temam and C^-1 for Yosida
         * - Q = (D H A H Dt)^-1 (D H Dt) ( A = C for Chorin-Temam and A = kin_vis K + NL for Yosida)
         * - R = Q for Chorin-Temam and I for Yosida
         *
         * This system leads to the following segregated steps:
         *
         * 1) Velocity prediction: A u_pred = fv + 1/dt M u_bdf
         * 2) Pressure prediction: DHG p_pred = alpha/dt (D u_pred - fp)
         * 3) Pressure correction: DHG z = dt/alpha (DHAHG p_pred)
         *          for Yosida: p = p_pred + z
         *          for Chorin-Temam: p = z
         * 4) Velocity correction: u = u_pred - H2 G p
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
         */

        class MonolithicNavierSolver
        {
        public:
            MonolithicNavierSolver(std::shared_ptr<ParMesh> pmesh_, BCHandler *bcs, real_t kin_vis_ = -1, int uorder_ = 2, int porder_ = 1, bool verbose_ = true);

            ~MonolithicNavierSolver();

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
             * \param params struct containing parameters for system solution
             *
             */
            void SetSolver(SolverParams params);

            /**
             * \brief Set the maximum order to use for the BDF method.
             *
             * \param order maximum bdf order to use for time integration 1 <= order <= 3
             *
             */
            void SetMaxBDFOrder(int order_) { max_bdf_order = order_; };

            /// Enable partial assembly for every operator.
            // (to be effective, must be set before Setup() is called)
            void EnablePA(bool pa_) { pa = pa_; };

            // Set verbosity
            void SetVerbose(bool verbose_) { verbose = verbose_; }
            void SetPreconditionerVerbose(bool verbose_) { pc_verbose = verbose_; }

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
            virtual void Setup(real_t dt, int pc_type_ = 0, int schur_pc_type_ = 5, bool mass_lumping = false, bool stiff_strain = false);

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
            virtual void Step(real_t &time, real_t dt, int current_step);

            /// Return a pointer to the velocity ParGridFunction.
            ParFiniteElementSpace *GetFESpaceVelocity() { return ufes; }
            /// Return a pointer to the pressure ParGridFunction.
            ParFiniteElementSpace *GetFESpacePressure() { return pfes; }

            /// Return velocity tdof vector
            Vector &GetVelocityVector() { return x->GetBlock(0); }
            /// Return pressure tdof vector
            Vector &GetPressureVector() { return x->GetBlock(1); }

            /// Return the velocity GridFunction
            ParGridFunction *GetVelocity() { return u_gf; }
            /// Return the pressure GridFunction
            ParGridFunction *GetPressure() { return p_gf; }

            // Problem size
            HYPRE_BigInt GetVelocitySize()
            {
               return ufes->GlobalTrueVSize();
            }

            HYPRE_BigInt GetPressureSize()
            {
               return pfes->GlobalTrueVSize();
            }

            // Visualization and Postprocessing
            void RegisterVisItFields(VisItDataCollection &visit_dc_);

            void RegisterParaviewFields(ParaViewDataCollection &paraview_dc_);

            void AddParaviewField(const std::string &field_name, ParGridFunction *gf);

            void AddVisItField(const std::string &field_name, ParGridFunction *gf);

            void WriteFields(const int &it = 0, const real_t &time = 0);

            ParaViewDataCollection &GetParaViewDc() { return *paraview_dc; }
            VisItDataCollection &GetVisItDc() { return *visit_dc; }

        protected:
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

            bool pa = false; // not supported yet

            /// Grid functions
            ParGridFunction *u_gf = nullptr;     // corrected velocity
            ParGridFunction *p_gf = nullptr;     // corrected pressure
            ParGridFunction *u_ext_gf = nullptr; // extrapolated velocity

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
            ParLinearForm *f_form = nullptr;

            /// Vectors
            BlockVector *x = nullptr;
            BlockVector *rhs = nullptr;

            Vector *u_bdf = nullptr; // BDF velocity
            Vector *u_ext = nullptr; // Extrapolated velocity (Picard linearization)

            Vector *un = nullptr;  // Corrected velocity (timestep n,   for BDF 1,2,3)
            Vector *un1 = nullptr; // Corrected velocity (timestep n-1, for BDF 2,3)
            Vector *un2 = nullptr; // Corrected velocity (timestep n-2, for BDF 3)

            /// Wrapper for extrapolated velocity coefficient
            VectorGridFunctionCoefficient *u_ext_vc = nullptr;

            /// Matrices/operators
            OperatorHandle opK;  // velocity laplacian
            OperatorHandle opM;  // velocity mass (unmodified)
            OperatorHandle opD;  // divergence
            OperatorHandle opG;  // gradient
            OperatorHandle opNL; // convective term   w . grad u (Picard convection)
            OperatorHandle opDe;
            OperatorHandle opSigmaM; // alpha/dt M (for Chorin-Temam and Pressure Corrected Preconditioners)

            OperatorHandle opC;           // C = alpha/dt + A
            HypreParMatrix *Ce = nullptr; // matrix for dirichlet bc modification

            /// Linear form to compute the mass matrix to set pressure mean to zero.
            ParLinearForm *mass_lf = nullptr;
            real_t volume = 0.0;

            /// Kinematic viscosity.
            ConstantCoefficient *kin_vis;

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

            /// Linear solvers parameters
            SolverParams sParams;

            /// Solvers and Preconditioners
            FGMRESSolver *nsSolver = nullptr; // Solver for Navier Stokes system
            Array<int> block_offsets;        // Block offsets for velocity and pressure
            BlockOperator *nsOp = nullptr;   // Navier-Stokes block operator

            int pc_type = 0;                      // PC type for Schur Complement: 0 Pressure Mass, 1 Pressure Laplacian, 2 PCD, 3 Cahouet-Chabard, 4 Approximate inverse
            NavierBlockPreconditioner *nsPrec = nullptr; // Navier-Stokes block preconditioner

            bool mass_lumping;      // Enable Mass Lumping 
            bool stiff_strain;      // Enable Stiff Strain
            
            int schur_pc_type = 0;  // PC type for Schur Complement: 0 Pressure Mass, 1 Pressure Laplacian, 2 PCD, 3 Cahouet-Chabard, 4 Approximate inverse
            Solver *invS = nullptr; // Schur Complement Preconditioner

            /// Variables for iterations/norm solvers
            int iter_solve = 0;
            real_t res_solve = 0.0;

            /// Timers
            StopWatch sw_setup;
            StopWatch sw_conv_assembly;
            StopWatch sw_solve;

            /// Data Collections
            ParaViewDataCollection *paraview_dc = nullptr;
            VisItDataCollection *visit_dc = nullptr;

            /// Enable/disable verbose output.
            bool verbose;
            bool pc_verbose = false;

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
            virtual void UpdateTimeBCS(real_t new_time);

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
        };

#endif // MFEM_NAVIER_UNSTEADY_HPP

    } // namespace navier

} // namespace mfem
