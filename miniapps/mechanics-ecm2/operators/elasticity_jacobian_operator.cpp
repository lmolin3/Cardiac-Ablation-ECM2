#include "elasticity_jacobian_operator.hpp"

using namespace mfem;
using namespace mfem::elasticity_ecm2;

template <int dim>
ElasticityJacobianOperator<dim>::ElasticityJacobianOperator(const ElasticityOperator<dim> *elasticity, const Vector &x)
   : Operator(elasticity->Height()),
     elasticity(elasticity),
     z(elasticity->Height())
{
   // Create ParGridFunction from the input vector
   ParGridFunction u(elasticity->fes);
   u.SetFromTrueDofs(x);

   // Get mesh nodes
   auto mesh_nodes = static_cast<ParGridFunction*>
                     (elasticity->fes->GetParMesh()->GetNodes());

   // Create the Jacobian derivative operator
   jacobian = elasticity->residual->GetDerivative(ElasticityOperator<dim>::Displacement, {&u}, {mesh_nodes});
}

template <int dim>
void ElasticityJacobianOperator<dim>::Mult(const Vector &x, Vector &y) const
{
   // Copy input to working T-vector
   z = x;
   
   // Apply essential boundary conditions (set to zero)
   z.SetSubVector(elasticity->ess_tdof_list, 0.0);

   // Apply the Jacobian operator
   jacobian->Mult(z, y);

   // Restore essential DOFs in output
   auto d_y = y.ReadWrite();
   const auto d_x = x.Read();
   const auto d_dofs = elasticity->ess_tdof_list.Read();
   mfem::forall(elasticity->ess_tdof_list.Size(), [=] MFEM_HOST_DEVICE(int i)
                { d_y[d_dofs[i]] = d_x[d_dofs[i]]; });
}


// Explicit template instantiation
namespace mfem {
namespace elasticity_ecm2 {
template class ElasticityJacobianOperator<2>;
template class ElasticityJacobianOperator<3>;
} // namespace elasticity_ecm2
} // namespace mfem