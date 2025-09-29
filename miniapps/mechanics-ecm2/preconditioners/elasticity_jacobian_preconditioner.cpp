#include "elasticity_jacobian_preconditioner.hpp"
#include "../operators/elasticity_jacobian_operator.hpp"

using namespace mfem;
using namespace mfem::elasticity_ecm2;

template <int dim>
AMGElasticityPreconditioner<dim>::AMGElasticityPreconditioner() 
   : ElasticityJacobianPreconditioner<dim>() 
{
}

// destructor
template <int dim>
AMGElasticityPreconditioner<dim>::~AMGElasticityPreconditioner()
{
   delete A; A = nullptr;
}

template <int dim>
void AMGElasticityPreconditioner<dim>::SetOperator(const Operator &op)
{
   this->height = op.Height();
   this->width = op.Width();

   auto elasticity_jacobian = dynamic_cast<const ElasticityJacobianOperator<dim>*>(&op);
   MFEM_VERIFY(elasticity_jacobian != nullptr, "invalid operator");

   const Array<int>& ess_tdof_list = elasticity_jacobian->elasticity->ess_tdof_list; 

   elasticity_jacobian->jacobian->Assemble(A); 
   auto Ae = A->EliminateRowsCols(ess_tdof_list);
   delete Ae;

   auto mesh_nodes = static_cast<ParGridFunction*>
                     (elasticity_jacobian->elasticity->fes->GetParMesh()->GetNodes());
                     
   amg = std::make_unique<HypreBoomerAMG>();
   amg->SetOperator(*A);
   amg->SetPrintLevel(0);

   bool amg_elast = false;
   if (amg_elast)
   {
      amg->SetElasticityOptions(elasticity_jacobian->elasticity->fes);
   }
   else
   {
      amg->SetSystemsOptions(mesh_nodes->ParFESpace()->GetMesh()->Dimension(), true);
   }
}

template <int dim>
void AMGElasticityPreconditioner<dim>::Mult(const Vector &x, Vector &y) const
{
   amg->Mult(x, y);
}


// Explicit template instantiation
namespace mfem {
namespace elasticity_ecm2 {
//template class ElasticityJacobianPreconditioner<2>;
//template class ElasticityJacobianPreconditioner<3>;
template class AMGElasticityPreconditioner<2>;
template class AMGElasticityPreconditioner<3>;   
} // namespace elasticity_ecm2
} // namespace mfem