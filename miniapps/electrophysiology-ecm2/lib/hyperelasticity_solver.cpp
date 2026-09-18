#include "hyperelasticity_solver.hpp"
#include "../../../general/forall.hpp"

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;
using namespace mfem::electrophysiology;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif

// NOTE: everything below is deliberately at namespace scope rather than in an
// anonymous namespace. Enzyme differentiates the q-function through the
// __enzyme_autodiff template, whose template arguments include the q-function's
// type; a type with internal linkage cannot appear there, and the build fails
// with "used but not defined in this translation unit".
namespace mfem
{
namespace electrophysiology
{
namespace mechanics_qf
{

constexpr int dim = 3;

// Field ids of the differentiable operator.
constexpr int Displacement = 0;
constexpr int Coords = 1;
constexpr int ActiveField = 2;
constexpr int Fibers = 3;
constexpr int Energy = 4;

#ifdef MFEM_USE_ENZYME

/**
 * @brief Renormalise an interpolated fiber direction.
 *
 * f0 is a unit field at the nodes, but H1 interpolation of a ROTATING unit
 * field does not preserve the norm in between: averaging two unit vectors
 * separated by a transmural rotation of dtheta gives cos(dtheta/2) at the
 * midpoint. With a 120 degree sweep over 3 elements that is |f0| = 0.94, hence
 * I4 - 1 = -0.117 at F = I -- the Holzapfel fiber term is then nonzero in the
 * undeformed configuration and the reference state carries residual stress.
 * Renormalising at the quadrature point makes the reference state stress-free
 * for any mesh resolution and fiber field.
 */
MFEM_HOST_DEVICE inline
tensor<real_t, dim> UnitFiber(const tensor<real_t, dim> &f0)
{
   const real_t n2 = sqnorm(f0);
   // A zero fiber vector means the field was never set; leave it alone rather
   // than dividing by zero, the fiber terms then simply contribute nothing.
   const real_t inv = (n2 > 1.0e-24_r) ? (1.0_r / sqrt(n2)) : 0.0_r;
   return inv * f0;
}

/**
 * @brief Transversely isotropic Holzapfel-type passive strain-energy density.
 *
 *   Psi(F) = kappa/2 log(J)^2 + c/2 (I1_bar - dim)
 *          + k1/(2 k2) (exp(k2 (I4 - 1)^2) - 1)
 *
 * with J = det F, I1_bar = J^(-2/3) tr(C), I4 = f0 . C f0. This matches the
 * Holzapfel material of miniapps/dfem/dfem-hyperelasticity_energy.cpp, with the
 * fiber direction taken from a field instead of being hard-coded.
 *
 * As in the reference, the full I4 is used rather than max(I4, 1): switching
 * the fiber term off in compression is the physically correct choice but
 * introduces a non-smoothness that the second derivative would have to handle.
 */
template <typename T>
MFEM_HOST_DEVICE inline
T PsiHolzapfel(const tensor<T, dim, dim> &F,
               const tensor<real_t, dim> &f0,
               real_t c, real_t kappa, real_t k1, real_t k2)
{
   const auto C = transpose(F) * F;
   const auto Jd = det(F);
   const auto Jm23 = pow(Jd, -2.0_r / 3.0_r);

   const auto I1_bar = Jm23 * tr(C);
   // I4 = f0 . C f0 = ||F f0||^2, computed without forming f0 (x) f0.
   const auto Ff0 = F * f0;
   const auto I4 = sqnorm(Ff0);
   const auto fiber_strain = I4 - 1.0_r;
   const auto log_J = log(Jd);

   const auto psi_vol = 0.5_r * kappa * log_J * log_J;
   const auto psi_iso = 0.5_r * c * (I1_bar - real_t(dim));
   const auto psi_aniso =
      (k1 / (2.0_r * k2)) * (exp(k2 * fiber_strain * fiber_strain) - 1.0_r);

   return psi_vol + psi_iso + psi_aniso;
}


/**
 * @brief Active stress: an extra fiber-directed term in the energy.
 *
 *   Psi(F, Ta) = Psi_Holzapfel(F) + 1/2 Ta (I4f - 1),  I4f = ||F f0||^2
 *
 * The active term is deliberately *not* switched off in compression: unlike the
 * passive fiber term it represents crossbridge force, which the cell generates
 * whether the fiber is stretched or shortened, so a max(I4f, 1) guard here would
 * suppress exactly the shortening the model is meant to produce.
 */
struct ActiveStressEnergy
{
   real_t c, kappa, k1, k2;

   MFEM_HOST_DEVICE inline
   void operator()(const tensor<dscalar_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &Ta,
                   const tensor<real_t, dim> &f0,
                   const real_t &w,
                   dscalar_t &energy) const
   {
      // The q-function adapter supplies the finite element measure: it maps
      // reference gradients to physical gradients and multiplies psi by
      // det(J) * w, i.e. the quadrature form of dx in Pi_int = int psi dx.
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto F = IdentityMatrix<dim>() + dudx;

      const auto f = UnitFiber(f0);
      const auto Ff0 = F * f;
      const auto I4f = sqnorm(Ff0);

      const auto psi = PsiHolzapfel(F, f, c, kappa, k1, k2)
                       + 0.5_r * Ta * (I4f - 1.0_r);

      energy = psi * det(J) * w;
   }
};


/**
 * @brief Active strain: multiplicative split F = F_e F_a.
 *
 *   F_a      = (1 - gamma) f0 (x) f0 + 1/sqrt(1 - gamma) (I - f0 (x) f0)
 *   F_a^(-1) = 1/(1 - gamma) f0 (x) f0 + sqrt(1 - gamma) (I - f0 (x) f0)
 *   F_e      = F F_a^(-1)
 *   Psi(F, gamma) = Psi_Holzapfel(F_e)
 *
 * F_a shortens the fiber by (1 - gamma) and expands the cross-fiber plane
 * isochorically, so det(F_a) = 1 and the activation itself does no volume work.
 * Only the passive law is evaluated, on F_e -- there is no separate active
 * stress term.
 */
struct ActiveStrainEnergy
{
   real_t c, kappa, k1, k2;

   MFEM_HOST_DEVICE inline
   void operator()(const tensor<dscalar_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &gamma_in,
                   const tensor<real_t, dim> &f0,
                   const real_t &w,
                   dscalar_t &energy) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto F = IdentityMatrix<dim>() + dudx;

      // gamma -> 1 makes F_a^(-1) singular, so clamp to a physically
      // meaningful shortening range. gamma < 0 (lengthening) stays allowed.
      const real_t gamma = gamma_in > 0.95_r ? 0.95_r : gamma_in;
      const real_t one_minus_g = 1.0_r - gamma;
      const real_t along = 1.0_r / one_minus_g;
      const real_t across = sqrt(one_minus_g);

      // F_a^(-1) = along * (f0 x f0) + across * (I - f0 x f0)
      const auto f = UnitFiber(f0);
      const auto M = make_tensor<dim, dim>(
      [&](int i, int j) { return f(i) * f(j); });
      const auto Fa_inv = along * M + across * (IdentityMatrix<dim>() - M);

      const auto Fe = F * Fa_inv;

      energy = PsiHolzapfel(Fe, f, c, kappa, k1, k2) * det(J) * w;
   }
};

#endif // MFEM_USE_ENZYME

} // namespace mechanics_qf
} // namespace electrophysiology
} // namespace mfem


namespace mfem
{
namespace electrophysiology
{

using namespace mechanics_qf;

#ifdef MFEM_USE_ENZYME

/**
 * @brief Holds the differentiable operator and the Newton-facing wrapper.
 *
 * Kept out of the header so the dFEM template machinery is instantiated in one
 * translation unit only.
 */
class HyperelasticitySolver::Impl : public Operator
{
   /// Matrix-free Hessian-vector product used by Newton's method.
   ///
   /// The wrapped DerivativeOperator returned by GetSecondDerivative computes
   /// the unconstrained second variation. This class adapts it to the Newton
   /// solve by zeroing constrained directions and restoring identity rows on
   /// essential true dofs.
   class HessianOperator : public Operator
   {
   public:
      HessianOperator(const Impl &oper, const Vector &state) :
         Operator(oper.Height()),
         oper(oper),
         z(oper.Height())
      {
         MultiVector X{state, oper.mesh_nodes_tdofs, oper.active, oper.fibers};
         hessian = oper.energy_dop->GetSecondDerivative(Displacement, X);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         z = x;
         z.SetSubVector(oper.ess_tdofs, 0.0);

         MultiVector Y{y};
         hessian->Mult(z, Y);

         auto d_y = y.ReadWrite();
         const auto d_x = x.Read();
         const auto d_dofs = oper.ess_tdofs.Read();
         mfem::forall(oper.ess_tdofs.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_y[d_dofs[i]] = d_x[d_dofs[i]];
         });
      }

      /**
       * @brief Assemble the constrained Hessian into a HypreParMatrix.
       *
       * dFEM hands back the unconstrained second variation, so the essential
       * dofs are eliminated here to match what HessianOperator::Mult applies:
       * identity rows and columns on the constrained dofs. Returns nullptr if
       * this integrator was not registered with a HypreParMatrix assembly path.
       */
      HypreParMatrix *AssembleMatrix() const
      {
         HypreParMatrix *A = nullptr;
         hessian->Assemble(A);
         if (A == nullptr) { return nullptr; }
         // EliminateRowsCols zeroes the constrained rows/columns and puts 1 on
         // the diagonal, which is exactly the operator Mult() represents.
         delete A->EliminateRowsCols(oper.ess_tdofs);
         return A;
      }

      void AssembleDiagonal(Vector &diag) const override
      {
         hessian->AssembleDiagonal(diag);

         auto d_diag = diag.ReadWrite();
         const auto d_dofs = oper.ess_tdofs.Read();
         mfem::forall(oper.ess_tdofs.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_diag[d_dofs[i]] = 1.0;
         });
      }

   private:
      friend class HessianAMG;
      const Impl &oper;
      mutable Vector z;
      std::shared_ptr<DerivativeOperator> hessian;
   };

public:
   /**
    * @brief BoomerAMG on the assembled Hessian.
    *
    * NewtonSolver hands the current Jacobian to the linear solver each
    * iteration, which forwards it here through SetOperator. That is the hook
    * used to re-assemble and rebuild the hierarchy at the new linearisation
    * point -- an AMG setup built once at u = 0 would degrade as the material
    * stiffens under load.
    */
   class HessianAMG : public Solver
   {
   public:
      explicit HessianAMG(ParFiniteElementSpace &fes) : Solver(), fes(fes) {}

      void SetOperator(const Operator &op) override
      {
         const auto *hess = dynamic_cast<const HessianOperator *>(&op);
         MFEM_VERIFY(hess, "HessianAMG expects a HessianOperator");

         height = width = op.Height();

         A.reset(hess->AssembleMatrix());
         MFEM_VERIFY(A != nullptr,
                     "HessianAMG: the Hessian could not be assembled into a "
                     "HypreParMatrix; use MechanicsPreconditioner::Jacobi.");

         amg = std::make_unique<HypreBoomerAMG>(*A);
         // Tell AMG this is a vector system so it coarsens the displacement
         // components together instead of treating the matrix as a scalar
         // problem. SetSystemsOptions, not SetElasticityOptions: the latter
         // wants rigid body modes, which hypre does not implement in a GPU
         // build, and it additionally assumes Ordering::byVDIM while the
         // displacement space here is byNODES.
         amg->SetSystemsOptions(dim, fes.GetOrdering() == Ordering::byNODES);
         amg->SetPrintLevel(0);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         MFEM_ASSERT(amg, "HessianAMG::SetOperator has not been called");
         amg->Mult(x, y);
      }

   private:
      ParFiniteElementSpace &fes;
      std::unique_ptr<HypreParMatrix> A;
      std::unique_ptr<HypreBoomerAMG> amg;
   };

public:
   Impl(ParFiniteElementSpace &fes,
        ParFiniteElementSpace &active_fes,
        ParFiniteElementSpace &fiber_fes,
        const IntegrationRule &ir,
        ActiveFormulation formulation,
        const HolzapfelParameters &p) :
      Operator(fes.GetTrueVSize()),
      fes(fes),
      qspace(*fes.GetParMesh(), ir),
      qspace_vec(qspace, 1)
   {
      auto &mesh_nodes =
         *static_cast<ParGridFunction *>(fes.GetParMesh()->GetNodes());
      mesh_nodes.GetTrueDofs(mesh_nodes_tdofs);

      const std::vector<FieldDescriptor> inputs =
      {
         {Displacement, &fes},
         {Coords, mesh_nodes.ParFESpace()},
         {ActiveField, &active_fes},
         {Fibers, &fiber_fes}
      };
      const std::vector<FieldDescriptor> outputs =
      {
         {Energy, &qspace_vec}
      };

      energy_dop = std::make_shared<DifferentiableOperator>(
                      inputs, outputs, *fes.GetParMesh());

      Array<int> all_domain_attr;
      if (fes.GetMesh()->attributes.Size() > 0)
      {
         all_domain_attr.SetSize(fes.GetMesh()->attributes.Max());
         all_domain_attr = 1;
      }

      // Only the displacement is differentiated: the active field, the fibers
      // and the mesh coordinates are frozen data for this solve. Requesting all
      // second-derivative blocks that can be formed from that single first
      // derivative gives exactly the Hessian d2 Pi / du2.
      auto derivatives = std::integer_sequence<size_t, Displacement> {};
      auto second_derivatives = SecondDerivatives<Pairs::All> {};

      if (formulation == ActiveFormulation::ActiveStress)
      {
         ActiveStressEnergy qf{p.c, p.kappa, p.k1, p.k2};
         energy_dop->AddDomainIntegrator<LocalQFBackend>(
            qf,
            Inputs<Gradient<Displacement>, Gradient<Coords>,
            Value<ActiveField>, Value<Fibers>, Weight> {},
            Outputs<FunctionalValue<Energy>> {},
            ir, all_domain_attr, derivatives, second_derivatives);
      }
      else
      {
         ActiveStrainEnergy qf{p.c, p.kappa, p.k1, p.k2};
         energy_dop->AddDomainIntegrator<LocalQFBackend>(
            qf,
            Inputs<Gradient<Displacement>, Gradient<Coords>,
            Value<ActiveField>, Value<Fibers>, Weight> {},
            Outputs<FunctionalValue<Energy>> {},
            ir, all_domain_attr, derivatives, second_derivatives);
      }

      // The first variation of a functional is exposed as a stateless
      // derivative operator, so this wrapper is built once and reused for every
      // residual evaluation.
      gradient = energy_dop->GetDerivative(Displacement);
   }

   void SetEssentialTrueDofs(const Array<int> &tdofs) { ess_tdofs = tdofs; }

   void SetActiveField(const Vector &a) { active = a; }
   void SetFiberField(const Vector &f) { fibers = f; }

   void Mult(const Vector &x, Vector &y) const override
   {
      // Residual R(u) = dPi/du. For functional integrators GetDerivative
      // returns the gradient action directly.
      MultiVector X{x, mesh_nodes_tdofs, active, fibers};
      MultiVector Y{y};
      gradient->Mult(X, Y);
      y.SetSubVector(ess_tdofs, 0.0);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      // Newton asks for the gradient of the residual; since the residual is the
      // energy gradient, this is the Hessian of the energy.
      hessian = std::make_shared<HessianOperator>(*this, x);
      return *hessian;
   }

   const Array<int> &GetEssentialTrueDofs() const { return ess_tdofs; }

   ParFiniteElementSpace &GetFESpace() const { return fes; }

private:
   friend class HessianOperator;

   ParFiniteElementSpace &fes;
   QuadratureSpace qspace;
   VectorQuadratureSpace qspace_vec;

   Vector mesh_nodes_tdofs;
   Vector active;
   Vector fibers;
   Array<int> ess_tdofs;

   std::shared_ptr<DifferentiableOperator> energy_dop;
   std::shared_ptr<DerivativeOperator> gradient;
   mutable std::shared_ptr<HessianOperator> hessian;
};

#else // !MFEM_USE_ENZYME

class HyperelasticitySolver::Impl { };

#endif // MFEM_USE_ENZYME


HyperelasticitySolver::HyperelasticitySolver(ParFiniteElementSpace &fes,
                                             ParFiniteElementSpace &active_fes,
                                             ParFiniteElementSpace &fiber_fes,
                                             const IntegrationRule &ir,
                                             ActiveFormulation formulation,
                                             const HolzapfelParameters &params)
   : fes_(fes), active_fes_(active_fes), fiber_fes_(fiber_fes), ir_(ir),
     formulation_(formulation), params_(params)
{
   MFEM_VERIFY(fes.GetVDim() == dim,
               "HyperelasticitySolver: displacement space must have vdim == 3.");
   MFEM_VERIFY(fiber_fes.GetVDim() == dim,
               "HyperelasticitySolver: fiber space must have vdim == 3.");
   MFEM_VERIFY(active_fes.GetVDim() == 1,
               "HyperelasticitySolver: active field space must be scalar.");

   u_gf_.SetSpace(&fes_);
   u_gf_ = 0.0;
   u_gf_.GetTrueDofs(u_);
   u_ = 0.0;

   active_tvector_.SetSize(active_fes_.GetTrueVSize());
   active_tvector_ = 0.0;
   active_tvector_.UseDevice(true);

   fiber_tvector_.SetSize(fiber_fes_.GetTrueVSize());
   fiber_tvector_ = 0.0;
   fiber_tvector_.UseDevice(true);

   BuildOperator();
}

HyperelasticitySolver::~HyperelasticitySolver() = default;

void HyperelasticitySolver::BuildOperator()
{
#ifdef MFEM_USE_ENZYME
   impl_ = std::make_unique<Impl>(fes_, active_fes_, fiber_fes_, ir_,
                                  formulation_, params_);
   if (ess_bdr_.Size() > 0)
   {
      Array<int> ess_tdofs;
      fes_.GetEssentialTrueDofs(ess_bdr_, ess_tdofs);
      impl_->SetEssentialTrueDofs(ess_tdofs);
   }
   impl_->SetActiveField(active_tvector_);
   impl_->SetFiberField(fiber_tvector_);
#else
   MFEM_ABORT("HyperelasticitySolver requires MFEM_USE_ENZYME=YES: the active "
              "strain-energy formulation uses dFEM functional second derivatives.");
#endif
}

void HyperelasticitySolver::SetFormulation(ActiveFormulation formulation)
{
   if (formulation == formulation_ && impl_) { return; }
   formulation_ = formulation;
   BuildOperator();
}

void HyperelasticitySolver::SetMaterialParameters(const HolzapfelParameters &params)
{
   params_ = params;
   BuildOperator();
}

void HyperelasticitySolver::SetEssentialAttributes(const Array<int> &ess_bdr)
{
   ess_bdr_ = ess_bdr;
#ifdef MFEM_USE_ENZYME
   Array<int> ess_tdofs;
   fes_.GetEssentialTrueDofs(ess_bdr_, ess_tdofs);
   impl_->SetEssentialTrueDofs(ess_tdofs);
#endif
}

void HyperelasticitySolver::SetFiberField(const Vector &f0_tvector)
{
   MFEM_VERIFY(f0_tvector.Size() == fiber_tvector_.Size(),
               "HyperelasticitySolver::SetFiberField(): size mismatch, expected "
               << fiber_tvector_.Size() << " got " << f0_tvector.Size());
   fiber_tvector_ = f0_tvector;
#ifdef MFEM_USE_ENZYME
   impl_->SetFiberField(fiber_tvector_);
#endif
}

void HyperelasticitySolver::SetFiberField(VectorCoefficient &f0_coeff)
{
   ParGridFunction f0_gf(&fiber_fes_);
   f0_gf.ProjectCoefficient(f0_coeff);
   f0_gf.GetTrueDofs(fiber_tvector_);
#ifdef MFEM_USE_ENZYME
   impl_->SetFiberField(fiber_tvector_);
#endif
}

void HyperelasticitySolver::UpdateActiveField(const Vector &active_tvector)
{
   MFEM_VERIFY(active_tvector.Size() == active_tvector_.Size(),
               "HyperelasticitySolver::UpdateActiveField(): size mismatch, expected "
               << active_tvector_.Size() << " got " << active_tvector.Size());
   active_tvector_ = active_tvector;
#ifdef MFEM_USE_ENZYME
   impl_->SetActiveField(active_tvector_);
#endif
}

void HyperelasticitySolver::SetNewtonOptions(int max_iter, real_t rel_tol,
                                             int print_level, real_t abs_tol)
{
   newton_max_iter_ = max_iter;
   newton_rel_tol_ = rel_tol;
   newton_print_level_ = print_level;
   newton_abs_tol_ = abs_tol;
}

void HyperelasticitySolver::SetLinearSolverOptions(int max_iter, real_t rel_tol,
                                                   int print_level, real_t abs_tol)
{
   linear_max_iter_ = max_iter;
   linear_rel_tol_ = rel_tol;
   linear_print_level_ = print_level;
   linear_abs_tol_ = abs_tol;
}

void HyperelasticitySolver::SolveQuasiStaticStep()
{
#ifdef MFEM_USE_ENZYME
   CGSolver cg(fes_.GetComm());
   cg.SetRelTol(linear_rel_tol_);
   cg.SetAbsTol(linear_abs_tol_);
   cg.SetMaxIter(linear_max_iter_);
   cg.SetPrintLevel(linear_print_level_);

   OperatorJacobiSmoother jacobi;
   std::unique_ptr<Impl::HessianAMG> amg;
   switch (prec_)
   {
      case MechanicsPreconditioner::None:
         break;
      case MechanicsPreconditioner::Jacobi:
         cg.SetPreconditioner(jacobi);
         break;
      case MechanicsPreconditioner::AMG:
         amg = std::make_unique<Impl::HessianAMG>(impl_->GetFESpace());
         cg.SetPreconditioner(*amg);
         break;
   }

   NewtonSolver newton(fes_.GetComm());
   newton.SetSolver(cg);
   newton.SetOperator(*impl_);
   newton.SetRelTol(newton_rel_tol_);
   newton.SetAbsTol(newton_abs_tol_);
   newton.SetMaxIter(newton_max_iter_);
   newton.SetPrintLevel(newton_print_level_);

   // Essential dofs stay at their current (prescribed) values: the residual and
   // the Hessian both carry identity rows there, so Newton never moves them.
   Vector zero;
   newton.Mult(zero, u_);

   last_newton_iterations_ = newton.GetNumIterations();
   last_linear_iterations_ = cg.GetNumIterations();
   last_converged_ = newton.GetConverged();

   u_gf_.Distribute(u_);
#else
   MFEM_ABORT("HyperelasticitySolver requires MFEM_USE_ENZYME=YES.");
#endif
}

void HyperelasticitySolver::ShorteningFromTension(const Vector &Ta, Vector &gamma,
                                                  real_t Ta_half, real_t gamma_max)
{
   const int n = Ta.Size();
   gamma.SetSize(n);
   gamma.UseDevice(true);

   const real_t *d_Ta = Ta.Read();
   real_t *d_g = gamma.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      const real_t t = d_Ta[i] > 0.0 ? d_Ta[i] : 0.0;
      d_g[i] = gamma_max * t / (t + Ta_half);
   });
}

} // namespace electrophysiology
} // namespace mfem
