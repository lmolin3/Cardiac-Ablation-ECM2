#pragma once

#include <mfem.hpp>
#include "bc/navierstokes_bchandler.hpp"
#include "navierstokes_preconditioners.hpp"
#include "navierstokes_residual.hpp"

namespace mfem
{
   using namespace navier;

   // Forward declaration
   class DiscretePressureLaplacian;
   class QuantitiesOfInterest;

   enum SplittingType
   {            // Splitting type for the Navier-Stokes operator
      IMPLICIT, // Fully implicit
      IMEX,     // IMEX
      EXPLICIT  // Fully explicit
   };

   /// Base abstract class for the segregated incompressible Navier-Stokes operator semidiscretized in space:
   ///
   /// [ M u_dot ]   =  f(t,U) + g(t,U)
   /// [    0    ]
   ///
   /// - D(M^-1)G p =  D(M^-1) ( Ku + C(u)u - F ) + H_dot
   ///
   /// Where f is the part of the residual treated implicitly, and g is treated explicitly.
   /// Derived class define different choices of f and g, which modifies the methods
   /// ImplicitMult, ExplicitMult, ImplicitSolve
   ///
   class NavierStokesOperator : public TimeDependentOperator
   {

      friend class NavierStokesResidual;
      friend class NavierStokesResidualImplicit;
      friend class NavierStokesResidualIMEX;

   public:
      /// @brief NavierStokesOperator
      /// @param vel_fes
      /// @param pres_fes
      /// @param BCHandler
      NavierStokesOperator(std::shared_ptr<ParMesh> mesh,
                           ParFiniteElementSpace *vel_fes,
                           ParFiniteElementSpace *pres_fes,
                           double kin_vis,
                           std::shared_ptr<BCHandler> bcs,
                           bool verbose);

      virtual ~NavierStokesOperator()
      {
         delete K_form;
         K_form = nullptr;
         delete M_form;
         M_form = nullptr;
         delete D_form;
         D_form = nullptr;
         delete forcing_form;
         forcing_form = nullptr;
         delete NL_form;
         NL_form = nullptr;
         delete DHGc;
         DHGc = nullptr; // Do not delete DHG, it is owned by DHGc
         delete invM;
         invM = nullptr;
         delete invM_pc;
         invM_pc = nullptr;
         delete invS;
         invS = nullptr;
         delete pc_builder;
         pc_builder = nullptr; // Do not delete invS_pc, it is owned by pc_builder
      }

      // Setups the Navier-Stokes operator
      // Must be called AFTER the BCHandler is populated with bcs
      void Setup(double initial_dt);

      // Set the RK order
      void SetOrder(int order);

      // Action of Navier-Stokes operator
      void Mult(const Vector &x, Vector &y) const override;

      // Multiply by velocity mass matrix
      void MassMult(const Vector &x, Vector &y);

      // Compute the explicit part of the Navier-Stokes residual
      virtual void ExplicitMult(const Vector &x, Vector &y) const override = 0;

      // Compute the implicit part of the Navier-Stokes residual
      virtual void ImplicitMult(const Vector &x, Vector &y) const = 0;

      // Solve the implicit step
      virtual void ImplicitSolve(Vector &b, Vector &x);

      // Solve for the segregated pressure
      void SolvePressure(Vector &y);

      // Compute term Hdot = D Udot for the inflow data
      void ComputeHdot(const Vector &xu, Vector &Hdot, int order);

      // Setup the Navier-Stokes operator
      void SetTimeStep(const double new_dt);

      // Set the time
      // Assumes EvalMode::ADDITIVE_TERM_1 by default (enables assembly of forcing term)
      void SetTime(double t, TimeDependentOperator::EvalMode eval_mode);

      // Set the time
      void SetTime(double t) override;

      // Set implicit RK coefficient (and propagate to residual)
      void SetImplicitCoefficient(const double coeff);

      // Set the Solvers for Pressure solve and Matrix inversion
      void SetSolvers(SolverParams params_p, SolverParams params_m, int pc_type);

      // Add acceleration term to the rhs (from VectorCoefficient)
      void AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr);

      // Add acceleration term to the rhs (from VectorFunction template)
      void AddAccelTerm(VecFuncT *func, Array<int> &attr);

      // Getters
      ParFiniteElementSpace &GetVelocityFES() { return *ufes; }
      ParFiniteElementSpace &GetPressureFES() { return *pfes; }
      ParGridFunction &GetVelocityGF() { return *u_gf; }
      ParGridFunction &GetPressureGF() { return *p_gf; }
      int GetUdim() { return udim; }
      int GetPdim() { return pdim; }
      Array<int> &GetOffsets() { return offsets; }

   protected:
      // Assemble the bilinear/linear forms
      void Assemble();

      // Update and Project the velocity Dirichlet BCs
      void ProjectVelocityDirichletBC(Vector &u);

      // Update and Project the pressure Dirichlet BCs
      void ProjectPressureDirichletBC(Vector &p);

      // Shared pointer to Mesh
      std::shared_ptr<ParMesh> pmesh;
      int dim;

      /// Velocity and Pressure FE spaces
      ParFiniteElementSpace *ufes = nullptr;
      ParFiniteElementSpace *pfes = nullptr;
      int uorder;
      int porder;
      int udim;
      int pdim;

      /// Grid functions
      std::unique_ptr<ParGridFunction> u_gf, p_gf, un_gf;

      /// BC Handler for NavierStokes
      std::shared_ptr<BCHandler> bcs;

      /// Dofs
      Array<int> vel_ess_tdof;      // All essential velocity true dofs.
      Array<int> vel_ess_tdof_full; // All essential true dofs from VectorCoefficient.
      Array<int> vel_ess_tdof_x;    // All essential true dofs x component.
      Array<int> vel_ess_tdof_y;    // All essential true dofs y component.
      Array<int> vel_ess_tdof_z;    // All essential true dofs z component.
      Array<int> pres_ess_tdof;     // All essential pressure true dofs.

      /// Bilinear/linear forms
      ParBilinearForm *K_form = nullptr;      // Bilinear form for velocity laplacian
      ParBilinearForm *M_form = nullptr;      //  Bilinear form for velocity mass
      ParMixedBilinearForm *D_form = nullptr; // Mixed form for divergence
      ParMixedBilinearForm *G_form = nullptr; // Mixed form for grAdient
      ParLinearForm *forcing_form = nullptr;  // Linear form for forcing term
      ParNonlinearForm *NL_form = nullptr;    // Nonlinear form to evaluate action of convective term

      /// Vectors and Operators
      Vector fu_rhs;       // Vector for forcing term
      Vector rhs_p;        // Vector for pressure rhs
      Vector Hdot;         // Vector for derivative of inflow data Hdot = D Udot
      mutable Vector z, w; // Mutable Vectors for intermediate computations

      OperatorHandle M; // velocity mass
      OperatorHandle K; // velocity laplacian
      OperatorHandle D; // divergence
      OperatorHandle G; // gradient

      TripleProductOperator *DHG = nullptr; // D(M^-1)G
      ConstrainedOperator *DHGc = nullptr;  // Constrained D(M^-1)G for pressure BCs

      /// Solvers and Preconditioners (pressure solve)
      SolverParams params_p; // Solver parameters for pressure solve
      SolverParams params_m; // Solver parameters for Mass matrix inversion

      CGSolver *invS = nullptr;        // solver for pressure solve
      int pc_type = 0;                 // PC type for Schur Complement: 0 Pressure Mass, 1 Pressure Laplacian, 2 PCD, 3 Cahouet-Chabard, 4 Approximate inverse
      PCBuilder *pc_builder = nullptr; // Preconditioner builder for Schur complement
      Solver *invS_pc;                 // Preconditioner for Schur complement

      CGSolver *invM = nullptr;  // solver for mass matrix inversion
      Solver *invM_pc = nullptr; // preconditioner for mass matrix inversion

      /// Coefficients
      std::unique_ptr<ConstantCoefficient> kin_vis;
      ConstantCoefficient zero_coeff;

      /// Residual object for Implicit solver
      std::unique_ptr<NavierStokesResidual> ns_residual; // Residual object for implicit solve
      std::unique_ptr<Solver> newton_pc;                 // Preconditioner for newton solver (inversion of Jacobian)

      /// Time
      double time = 0.0;
      mutable double dt = -1.0; // dt for intermediate steps

      /// Splitting type
      SplittingType splitting_type;

      /// RK order
      int rk_order = 1; // Order of RK timestepping (needed to compute Hdot consistently) --> Retrieved from NavierStokesSRKSolver::GetOrder()

      // Offsets for block vector
      Array<int> offsets;

      // Integration rules
      IntegrationRules intrules;
      IntegrationRule ir, ir_nl, ir_face;

      // Bookkeeping for acceleration (forcing) terms.
      std::vector<VecCoeffContainer> accel_terms;

      // Enable/disable verbose output.
      bool verbose;
   };

   // Class derived from NavierStokesOperator using the IMEX scheme
   class NavierStokesOperatorIMEX : public NavierStokesOperator
   {
   public:
      NavierStokesOperatorIMEX(std::shared_ptr<ParMesh> mesh,
                               ParFiniteElementSpace *vel_fes,
                               ParFiniteElementSpace *pres_fes,
                               double kin_vis,
                               std::shared_ptr<BCHandler> bcs,
                               bool verbose);

      virtual void ExplicitMult(const Vector &xb, Vector &yb) const;
      virtual void ImplicitMult(const Vector &xb, Vector &yb) const;

   private:
   };

   // Class derived from NavierStokesOperator using the Implicit splitting type
   class NavierStokesOperatorImplicit : public NavierStokesOperator
   {
   public:
      NavierStokesOperatorImplicit(std::shared_ptr<ParMesh> mesh,
                                   ParFiniteElementSpace *vel_fes,
                                   ParFiniteElementSpace *pres_fes,
                                   double kin_vis,
                                   std::shared_ptr<BCHandler> bcs,
                                   bool verbose);

      virtual void ExplicitMult(const Vector &xb, Vector &yb) const;
      virtual void ImplicitMult(const Vector &xb, Vector &yb) const;
   };

   // Class DiscretePressureLaplacian compute the action x -> - D(M^-1)G x
   class DiscretePressureLaplacian : public TripleProductOperator
   {
   public:
      // Constructor
      DiscretePressureLaplacian(const Operator *D, const Operator *invM,
                                const Operator *G, bool ownD, bool own_invM, bool ownG)
          : TripleProductOperator(D, invM, G, ownD, own_invM, ownG)
      {
      }

      // Override the Mult method
      void Mult(const Vector &x, Vector &y) const override
      {
         TripleProductOperator::Mult(x, y); // Compute y = D M^-1 G x
         y.Neg();                           // Compute y = - D M^-1 G x
      }
   };

   // Class containing computation of potential quantities of interest
   class QuantitiesOfInterest
   {
   public:
      QuantitiesOfInterest(ParMesh *pmesh);

      ~QuantitiesOfInterest() { delete mass_lf; };

      // Computes K = 0.5 * int_{Omega} |u|^2
      double ComputeKineticEnergy(ParGridFunction &v);

      // Compute CFL = dt u / h
      double ComputeCFL(ParGridFunction &u, double dt);

   private:
      ConstantCoefficient onecoeff;
      ParLinearForm *mass_lf;
      double volume;
   };

}