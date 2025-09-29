#include "quasi_static_elasticity_solver.hpp" 

using namespace mfem;
using namespace mfem::elasticity_ecm2;

template <int dim>
ElasticitySolver<dim>::ElasticitySolver(ParMesh *pmesh_, int order, bool verbose_)
{

   // Create the finite element space and set height/width of Operator
   fec = new H1_FECollection(order, dim);
   fes = new ParFiniteElementSpace(pmesh_, fec, dim);

   // Create the operator
   op = std::make_unique<ElasticityOperator<dim>>(fes, verbose_);

   // Initialize gfs
   u_gf.SetSpace(fes);
   u_gf = 0.0;

   //zero = std::make_unique<Vector>(fes->GetTrueVSize());
   U = std::make_unique<Vector>(fes->GetTrueVSize());
   //*zero = 0.0;
   *U = 0.0;
}


template <int dim>ElasticitySolver<dim>::~ElasticitySolver()
{
   delete fec;
   fec = nullptr;
   delete fes;
   fes = nullptr;
}

template <int dim>
HYPRE_BigInt
ElasticitySolver<dim>::GetProblemSize()
{
   return fes->GlobalTrueVSize();
}

template <int dim>
void ElasticitySolver<dim>::Setup(int k_grad_update, PreconditionerType prec_type)
{
   // Setup the operator
   op->Setup();

   // Setup the nonlinear solver and jacobian solver
   switch (prec_type)
   {
      case PreconditionerType::AMG:
            j_prec = std::make_unique<AMGElasticityPreconditioner<dim>>();
         break;
      default:
         MFEM_ABORT("Unknown PreconditionerType");
   }
   //j_prec->SetOperator(*op); // Should be called already inside the NewtonSolver-->GMRESolver

   j_solver = std::make_unique<GMRESSolver>(MPI_COMM_WORLD);
   j_solver->iterative_mode = false;
   j_solver->SetAbsTol(0.0);
   j_solver->SetRelTol(1e-4);
   // j_solver->SetKDim(500);
   j_solver->SetMaxIter(500);
   j_solver->SetPrintLevel(2);
   j_solver->SetPreconditioner(*j_prec);

   nonlinear_solver = std::make_unique<FrozenNewtonSolver>(MPI_COMM_WORLD, k_grad_update);
   nonlinear_solver->iterative_mode = true;
   nonlinear_solver->SetOperator(*op);
   nonlinear_solver->SetAbsTol(0.0);
   nonlinear_solver->SetRelTol(1e-6);
   nonlinear_solver->SetMaxIter(500);
   nonlinear_solver->SetSolver(*j_solver);
   nonlinear_solver->SetPrintLevel(1);
   //nonlinear_solver->SetAdaptiveLinRtol(2, 0.5, 0.9);
}


template <int dim>
void ElasticitySolver<dim>::Solve()
{
   // Apply bcs to displacement vector, and update the operator (re-assemble the rhs)
   op->Update();

   for (auto &prescribed_disp : op->prescribed_displacements)
   {
      u_gf.ProjectBdrCoefficient(*prescribed_disp.coeff, prescribed_disp.attr);
   }
   u_gf.GetTrueDofs(*U);
   U->SetSubVector(op->fixed_tdof_list, 0.0);
      
   // Solve F(x) = 0 --> F(x) = H(x) - b = 0
   //nonlinear_solver->Mult(*zero, *U);
   Vector zero;
   nonlinear_solver->Mult(zero, *U);
   MFEM_ASSERT(nonlinear_solver->GetConverged(), "Nonlinear solver did not converge");

   // Update the displacement grid function
   u_gf.SetFromTrueDofs(*U);
}

// Expose the interface for adding bcs to the operator
/**
 * @brief Set the boundaries to be fixed.
 * @param fixed_bdr Array marking the boundary attributes for fixed constraints.
 * @param component The component to apply the fixed constraint to (-1 for all components).
 */
template <int dim>
void ElasticitySolver<dim>::AddFixedConstraint(const Array<int> &fixed_bdr, int component)
{
   op->AddFixedConstraint(fixed_bdr, component);
}

template <int dim>
void ElasticitySolver<dim>::AddFixedConstraint(int attr, int component)
{
   op->AddFixedConstraint(attr, component);
}

template <int dim>
void ElasticitySolver<dim>::AddPrescribedDisplacement(VectorCoefficient *disp, Array<int> &attr, real_t scaling, bool own)
{
   op->AddPrescribedDisplacement(disp, attr, scaling, own);
}

template <int dim>
void ElasticitySolver<dim>::AddPrescribedDisplacement(VecFuncT disp, Array<int> &attr, real_t scaling)
{
   op->AddPrescribedDisplacement(disp, attr, scaling);
}

template <int dim>
void ElasticitySolver<dim>::AddPrescribedDisplacement(VectorCoefficient *disp, int attr, real_t scaling, bool own)
{
   op->AddPrescribedDisplacement(disp, attr, scaling, own);
}

template <int dim>
void ElasticitySolver<dim>::AddPrescribedDisplacement(VecFuncT disp, int attr, real_t scaling)
{
   op->AddPrescribedDisplacement(disp, attr, scaling);
}

template <int dim>
void ElasticitySolver<dim>::AddBoundaryLoad(VectorCoefficient *coeff, Array<int> &attr, real_t scaling, bool own)
{
   op->AddBoundaryLoad(coeff, attr, scaling, own);
}

template <int dim>
void ElasticitySolver<dim>::AddBoundaryLoad(VecFuncT func, Array<int> &attr, real_t scaling)
{
   op->AddBoundaryLoad(func, attr, scaling);
}

template <int dim>
void ElasticitySolver<dim>::AddBoundaryLoad(VectorCoefficient *coeff, int attr, real_t scaling, bool own)
{
   op->AddBoundaryLoad(coeff, attr, scaling, own);
}

template <int dim>
void ElasticitySolver<dim>::AddBoundaryLoad(VecFuncT func, int attr, real_t scaling)
{
   op->AddBoundaryLoad(func, attr, scaling);
}

/**
 * @brief Set the body force (volumetric load) on specified domain attributes.
 * API allows setting the BodyForce:
 * 1. Using either VectorCoefficient/VecFuncT (alias for void(const Vector &x, double t, Vector &u))
 * 2. Specifying single domain attribute or an array of attributes.
 */
template <int dim>
void ElasticitySolver<dim>::AddBodyForce(VectorCoefficient *coeff, Array<int> &attrs, bool own)
{
   op->AddBodyForce(coeff, attrs, own);
}

template <int dim>
void ElasticitySolver<dim>::AddBodyForce(VecFuncT func, Array<int> &attrs)
{
   op->AddBodyForce(func, attrs);
}

template <int dim>
void ElasticitySolver<dim>::AddBodyForce(VectorCoefficient *coeff, int attr, bool own)
{
   op->AddBodyForce(coeff, attr, own);
}

template <int dim>
void ElasticitySolver<dim>::AddBodyForce(VecFuncT func, int attr)
{
   op->AddBodyForce(func, attr);
}


// Explicit template instantiation
namespace mfem {
namespace elasticity_ecm2 {
template class ElasticitySolver<2>;
template class ElasticitySolver<3>;
} // namespace elasticity_ecm2
} // namespace mfem