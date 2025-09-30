#include "elasticity_jacobian_preconditioner.hpp"
#include "../operators/elasticity_jacobian_operator.hpp"

using namespace mfem;
using namespace mfem::elasticity_ecm2;

template <int dim>
AMGElasticityPreconditioner<dim>::AMGElasticityPreconditioner(bool amg_elast) 
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

   if (Mpi::Root())
   {
      mfem::out << "Assembling Jacobian" << std::endl;
   }
   delete A; A = nullptr;
   elasticity_jacobian->jacobian->Assemble(A); 
   auto Ae = A->EliminateRowsCols(ess_tdof_list);
   delete Ae;

   auto mesh_nodes = static_cast<ParGridFunction*>
                     (elasticity_jacobian->elasticity->fes->GetParMesh()->GetNodes());
                     
   amg = std::make_unique<HypreBoomerAMG>();
   amg->SetOperator(*A);
   amg->SetPrintLevel(0);

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



template <int dim>
NestedElasticityPreconditioner<dim>::NestedElasticityPreconditioner(int inner_iter_max_, real_t inner_tol_)
   : ElasticityJacobianPreconditioner<dim>(), 
     inner_iter_max(inner_iter_max_), 
     inner_tol(inner_tol_)
{
   inner_solver = std::make_unique<GMRESSolver>();
   inner_solver->SetAbsTol(0.0);
   inner_solver->SetRelTol(inner_tol_);
   inner_solver->SetMaxIter(inner_iter_max);
   inner_solver->SetPrintLevel(-1);
}

#ifdef MFEM_USE_MPI
   template <int dim>
   NestedElasticityPreconditioner<dim>::NestedElasticityPreconditioner(MPI_Comm comm_, int inner_iter_max_, real_t inner_tol_)
      : ElasticityJacobianPreconditioner<dim>(), 
        inner_iter_max(inner_iter_max_), 
        inner_tol(inner_tol_)
   {
      inner_solver = std::make_unique<GMRESSolver>(comm_);
      inner_solver->SetAbsTol(0.0);
      inner_solver->SetRelTol(inner_tol_);
      inner_solver->SetMaxIter(inner_iter_max);
      inner_solver->SetPrintLevel(-1);
   }
#endif

template <int dim>
NestedElasticityPreconditioner<dim>::~NestedElasticityPreconditioner()
{
   // Cleanup handled by unique_ptr
}

template <int dim>
void NestedElasticityPreconditioner<dim>::SetOperator(const Operator &op)
{
   auto elasticity_jacobian = dynamic_cast<const ElasticityJacobianOperator<dim>*>(&op);
   MFEM_VERIFY(elasticity_jacobian != nullptr, "invalid operator");

   inner_solver->SetOperator(op);
}

template <int dim>
void NestedElasticityPreconditioner<dim>::Mult(const Vector &x, Vector &y) const
{
   inner_solver->Mult(x, y);
}



// Explicit template instantiation
namespace mfem {
namespace elasticity_ecm2 {
//template class ElasticityJacobianPreconditioner<2>;
//template class ElasticityJacobianPreconditioner<3>;
template class AMGElasticityPreconditioner<2>;
template class AMGElasticityPreconditioner<3>;   
template class NestedElasticityPreconditioner<2>;
template class NestedElasticityPreconditioner<3>;
} // namespace elasticity_ecm2
} // namespace mfem