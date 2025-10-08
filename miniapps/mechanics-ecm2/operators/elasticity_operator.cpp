#include "elasticity_operator.hpp"

using namespace mfem;
using namespace mfem::elasticity_ecm2;

template <int dim>
ElasticityOperator<dim>::ElasticityOperator(ParFiniteElementSpace *fes_, bool verbose_)
    : pmesh(fes_->GetParMesh()),
      sdim(pmesh->Dimension()),
      fes(fes_),
      verbose(verbose_)
{
   fes_truevsize = fes->GetTrueVSize();
   this->height = fes_truevsize;
   this->width = fes_truevsize;

   if (Mpi::Root() && verbose)
   {
      printf("Elasticity #tdofs: %d\n", fes_truevsize);
   }

   // Initialize bcs
   max_bdr_attributes = pmesh->bdr_attributes.Max();
   max_domain_attributes = pmesh->attributes.Max();
   fixed_attrs.SetSize(max_bdr_attributes);
   fixed_attrs = 0;
   fixed_attrs_xyz.SetSize(max_bdr_attributes);
   fixed_attrs_xyz = 0;
   prescribed_disp_attr.SetSize(max_bdr_attributes);
   prescribed_disp_attr = 0;
   boundary_load_attr.SetSize(max_bdr_attributes);
   boundary_load_attr = 0;
   body_force_attr.SetSize(max_domain_attributes);
   body_force_attr = 0;
   tmp_bdr_attrs.SetSize(max_bdr_attributes);
   tmp_bdr_attrs = 0;
   tmp_domain_attrs.SetSize(max_domain_attributes);
   tmp_domain_attrs = 0;

   // Initialize vectors
   B.SetSize(fes_truevsize);
   B = 0.0;
}


template <int dim>
ElasticityOperator<dim>::~ElasticityOperator()
{
   // Clean up allocated resources
   for (auto &ramped_load : ramped_boundary_loads)
   {
      delete ramped_load;
   }

   for (auto &ramped_force : ramped_body_forces)
   {
      delete ramped_force;
   }
}

////////////////////////////////////////////////////////////////////////
///----------------------/ BC interface /-----------------------------//
////////////////////////////////////////////////////////////////////////

template <int dim>
void ElasticityOperator<dim>::AddFixedConstraint(const Array<int> &attr, int component)
{
   MFEM_ASSERT(attr.Size() >= max_bdr_attributes, "Size of attributes array does not match mesh attributes.");

   // Append to the list of fixed constraints
   fixed_constraints.emplace_back(attr, component);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((prescribed_disp_attr[i] && fixed_attrs[i] && fixed_attrs_xyz[i] && boundary_load_attr[i] && attr[i]) == 0, "Duplicate boundary definition detected.");
      if (attr[i] == 1 && component < 0)
      {
         fixed_attrs[i] = 1; // Mark the attribute as fixed
      }
      else if (attr[i] == 1 && component >= 0)
      {
         fixed_attrs_xyz[i] = 1; // Mark the attribute as fixed for a specific component
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding FixedConstraint to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      if (component >= 0)
      {
         mfem::out << " for component " << component;
      }
      mfem::out << std::endl;
   }
}

template <int dim>
void ElasticityOperator<dim>::AddFixedConstraint(int attr, int component)
{
   tmp_bdr_attrs = 0;
   tmp_bdr_attrs[attr - 1] = 1; // Set the specific attribute
   AddFixedConstraint(tmp_bdr_attrs, component);
}

template <int dim>
void ElasticityOperator<dim>::AddPrescribedDisplacement(VectorCoefficient *coeff, Array<int> &attr, real_t scaling, bool own)
{
   MFEM_ASSERT(attr.Size() >= max_bdr_attributes, "Size of attributes array does not match mesh attributes.");

   // Append to the list of prescribed displacements
   prescribed_displacements.emplace_back(attr, coeff, scaling, own);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((prescribed_disp_attr[i] && fixed_attrs[i] && fixed_attrs_xyz[i] && boundary_load_attr[i] && attr[i]) == 0, "Duplicate boundary definition detected.");
      if (attr[i] == 1)
      {
         prescribed_disp_attr[i] = 1; // Mark the attribute as prescribed displacement
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding PrescribedDisplacement to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

template <int dim>
void ElasticityOperator<dim>::AddPrescribedDisplacement(VecFuncT disp, Array<int> &attr, real_t scaling)
{
   AddPrescribedDisplacement(new VectorFunctionCoefficient(pmesh->Dimension(), disp), attr, scaling, true);
}

template <int dim>
void ElasticityOperator<dim>::AddPrescribedDisplacement(VectorCoefficient *disp, int attr, real_t scaling, bool own)
{
   tmp_bdr_attrs = 0;
   tmp_bdr_attrs[attr - 1] = 1; // Set the specific attribute
   AddPrescribedDisplacement(disp, tmp_bdr_attrs, scaling, own);
}

template <int dim>
void ElasticityOperator<dim>::AddPrescribedDisplacement(VecFuncT disp, int attr, real_t scaling)
{
   AddPrescribedDisplacement(new VectorFunctionCoefficient(pmesh->Dimension(), disp), attr, scaling, true);
}


template <int dim>
void ElasticityOperator<dim>::AddBoundaryLoad(VectorCoefficient *coeff, Array<int> &attr, real_t scaling, bool own)
{
   MFEM_ASSERT(attr.Size() >= max_bdr_attributes, "Size of attributes array does not match mesh attributes.");

   // Append to the list of boundary loads
   boundary_loads.emplace_back(attr, coeff, scaling, own);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((prescribed_disp_attr[i] && fixed_attrs[i] && boundary_load_attr[i] && attr[i]) == 0, "Duplicate boundary definition detected.");
      if (attr[i] == 1)
      {
         boundary_load_attr[i] = 1;
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding BoundaryLoad to boundary attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}


template <int dim>
void ElasticityOperator<dim>::AddBoundaryLoad(VecFuncT func, Array<int> &attr, real_t scaling)
{
   AddBoundaryLoad(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr, scaling, true);
}


template <int dim>
void ElasticityOperator<dim>::AddBoundaryLoad(VectorCoefficient *coeff, int attr, real_t scaling, bool own)
{
   tmp_bdr_attrs = 0;
   tmp_bdr_attrs[attr - 1] = 1; // Set the specific attribute
   AddBoundaryLoad(coeff, tmp_bdr_attrs, scaling, own);
}


template <int dim>
void ElasticityOperator<dim>::AddBoundaryLoad(VecFuncT func, int attr, real_t scaling)
{
   AddBoundaryLoad(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr, scaling, true);
}


template <int dim>
void ElasticityOperator<dim>::AddBodyForce(VectorCoefficient *coeff, Array<int> &attr, real_t scaling, bool own)
{
   MFEM_ASSERT(attr.Size() >= max_domain_attributes, "Size of attributes array does not match mesh attributes.");

   // Append to the list of body forces
   body_forces.emplace_back(attr, coeff, scaling, own);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((body_force_attr[i] && attr[i]) == 0, "Duplicate domain definition detected.");
      if (attr[i] == 1)
      {
         body_force_attr[i] = 1;
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding BodyForce to domain attributes: ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}


template <int dim>
void ElasticityOperator<dim>::AddBodyForce(VecFuncT func, Array<int> &attr, real_t scaling)
{
   AddBodyForce(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr, scaling, true);
}


template <int dim>
void ElasticityOperator<dim>::AddBodyForce(VectorCoefficient *coeff, int attr, real_t scaling, bool own)
{
   tmp_domain_attrs = 0;
   tmp_domain_attrs[attr - 1] = 1; // Set the specific attribute
   AddBodyForce(coeff, tmp_domain_attrs, scaling, own);
}


template <int dim>
void ElasticityOperator<dim>::AddBodyForce(VecFuncT func, int attr, real_t scaling)
{
   AddBodyForce(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr, scaling, true);
}

//////////////////////////////////////////////////////////////////////////////
///----------------------/ Operator interface /-----------------------------//
//////////////////////////////////////////////////////////////////////////////

template <int dim>
void ElasticityOperator<dim>::SetMaterial(const MaterialVariant<dim> &material)
{
   material_type_name = std::visit([](const auto &mat)
                                   {
    using MaterialType = std::decay_t<decltype(mat)>;
    return std::string(MaterialType::name); }, material);

   material_set = true;

   qfunction = std::make_unique<StressQFunction<dim>>(material);
}

template <int dim>
void ElasticityOperator<dim>::Setup(int ir_order)
{
   MFEM_ASSERT(material_set, "Material not set. Call SetMaterial<> before Setup().");

   Array<int> empty;

   ///----- 1. Extract the essential dof list
   // For now we include only fixed constraints, but we can extend to prescribed displacements, velocity and acceleration
   for (auto &fixed_constraint : fixed_constraints)
   {
      Array<int> tmp_tdof_list;
      fes->GetEssentialTrueDofs(fixed_constraint.attr, tmp_tdof_list, fixed_constraint.component);
      fixed_tdof_list.Append(tmp_tdof_list);
   }

   for (auto &prescribed_disp : prescribed_displacements)
   {
      Array<int> tmp_tdof_list;
      fes->GetEssentialTrueDofs(prescribed_disp.attr, tmp_tdof_list); // -1 for all components
      prescribed_disp_tdof_list.Append(tmp_tdof_list);
   }

   ess_tdof_list.Append(fixed_tdof_list);
   ess_tdof_list.Append(prescribed_disp_tdof_list);
   ess_tdof_list.Sort();
   ess_tdof_list.Unique();

   ///----- 2. Create DifferentiableOperator for stiffness matrix
   auto pmesh = fes->GetParMesh();
   auto mesh_nodes = static_cast<ParGridFunction *>(pmesh->GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   // Field descriptors for the solution and parameter fields
   std::vector<FieldDescriptor> solutions, parameters;
   solutions.push_back(FieldDescriptor{Displacement, fes});
   parameters.push_back(FieldDescriptor{Coordinates, &mesh_fes});

   // Create the DifferentiableOperator
   residual = std::make_unique<DifferentiableOperator>(solutions, parameters, *pmesh);
   // residual->DisableTensorProductStructure();

   // Input, Output operators, and what derivatives to form
   auto input_operators = tuple
   {
      Gradient<Displacement>{},
      Gradient<Coordinates>{},
      Weight{}
   };
   auto output_operators = tuple
   {
      Gradient<Displacement>{}
   };

   auto derivatives = std::integer_sequence<size_t, Displacement>{};

   Array<int> solid_domain_attr(pmesh->attributes.Max());  solid_domain_attr = 1;
   const IntegrationRule &displacement_ir =
       IntRules.Get(fes->GetTypicalFE()->GetGeomType(),
                    2 * ir_order + fes->GetTypicalFE()->GetOrder());
   residual->AddDomainIntegrator(*qfunction, input_operators, output_operators, displacement_ir, solid_domain_attr, derivatives);

   // Set parameters (mesh nodes): this must be updated if the mesh moves
   residual->SetParameters({mesh_nodes});

   
   ///----- 3. Assemble linear form for the rhs
   Fform = std::make_unique<ParLinearForm>(fes);

   // 3.1 Boundary loads
   for (auto &boundary_load : boundary_loads)
   {
      // Create ramped coefficient
      ramped_boundary_loads.push_back(new ScalarVectorProductCoefficient(1.0, *boundary_load.coeff));
      Fform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*ramped_boundary_loads.back()), boundary_load.attr);
   }

   // 3.2 Body acceleration (volumetric terms)
   for (auto &body_force : body_forces)
   {
      // Create ramped coefficient
      ramped_body_forces.push_back(new ScalarVectorProductCoefficient(1.0, *body_force.coeff));
      Fform->AddDomainIntegrator(new VectorDomainLFIntegrator(*ramped_body_forces.back()), body_force.attr);
   }

   UpdateRHS();

   setup = true;
   tmp_bdr_attrs.DeleteAll();
   tmp_domain_attrs.DeleteAll();
}

template <int dim>
void ElasticityOperator<dim>::Update()
{}

template <int dim>
void ElasticityOperator<dim>::ProjectEssentialBCs(Vector &U, ParGridFunction &u_gf)
{
   ramp_factor = load_ramping ? ramp_factor : 1.0;

   for (auto &prescribed_disp : prescribed_displacements)
   {
      ScalarVectorProductCoefficient scaled_coeff(ramp_factor, *prescribed_disp.coeff);
      u_gf.ProjectBdrCoefficient(scaled_coeff, prescribed_disp.attr);
   }
   u_gf.GetTrueDofs(U);
   U.SetSubVector(fixed_tdof_list, 0.0);
}

template <int dim>
void ElasticityOperator<dim>::SetRampingStep(int step)
{
   if (!load_ramping)
   {
      return;
   }

   current_ramping_step = step;

   // NOTE: here we could implement different ramping strategies, for now it's linear
   ramp_factor = current_ramping_step < load_ramping_steps ? static_cast<real_t>(current_ramping_step) / static_cast<real_t>(load_ramping_steps) : 1.0;

   // Update ramped coefficients for boundary loads (iterate over std::vector ramped_boundary_loads)
   for (auto &ramped_boundary_load : ramped_boundary_loads)
   {
      ramped_boundary_load->SetAConst(ramp_factor);
   }

   // Update ramped coefficients for body forces (iterate over std::vector ramped_body_forces)
   for (auto &ramped_body_force : ramped_body_forces)
   {
      ramped_body_force->SetAConst(ramp_factor);
   }
}

template <int dim>
void ElasticityOperator<dim>::UpdateRHS()
{
   // TODO: check what else we need to do here. also the residual and jacobian?
   Fform->Update();
   Fform->Assemble();
   Fform->ParallelAssemble(B);
}



// --------------------------------------------------------------------------------- //
//// ----- Mult() and GetGradient() methods -----
// --------------------------------------------------------------------------------- //

template <int dim>
void ElasticityOperator<dim>::Mult(const Vector &U, Vector &Y) const
{
   // This needs to be called if the mesh has moved
   // residual->SetParameters({mesh_nodes});

   // Compute the residual R(U) = K(U) - F
   residual->Mult(U, Y);
   Y -= B;

   // Set the residual at Dirichlet dofs to zero
   // This includes both fixed tdofs and prescribed displacements
   Y.SetSubVector(ess_tdof_list, 0.0);
}

template <int dim>
Operator& ElasticityOperator<dim>::GetGradient(const Vector &U) const
{
   // Update the cached state and create/update the Jacobian operator
   jacobian_op = std::make_unique<ElasticityJacobianOperator<dim>>(this, U);
   return *jacobian_op;
}


// Explicit template instantiation
namespace mfem {
namespace elasticity_ecm2 {
template class ElasticityOperator<2>;
template class ElasticityOperator<3>;
} // namespace elasticity_ecm2
} // namespace mfem