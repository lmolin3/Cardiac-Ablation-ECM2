#include "heat_bchandler.hpp"

using namespace mfem;
using namespace heat;

BCHandler::BCHandler(std::shared_ptr<ParMesh> mesh, bool verbose)
    : pmesh(mesh), verbose(verbose)
{
    ParSubMesh *submesh = dynamic_cast<ParSubMesh *>(pmesh.get());
    if (submesh)
    {
        // Provided mesh is a ParSubMesh --> Get the parent mesh max attributes
        max_bdr_attributes = submesh->GetParent()->bdr_attributes.Max();
    }
    else
    {
        // Provided mesh is a ParMesh
        max_bdr_attributes = pmesh->bdr_attributes.Max();
    }

    // initialize vectors of essential attributes
    dirichlet_attr.SetSize(max_bdr_attributes);
    dirichlet_attr = 0;
    dirichlet_attr_tmp.SetSize(max_bdr_attributes);
    dirichlet_attr_tmp = 0;
    neumann_attr.SetSize(max_bdr_attributes);
    neumann_attr = 0;
    neumann_vec_attr.SetSize(max_bdr_attributes);
    neumann_vec_attr = 0;
    robin_attr.SetSize(max_bdr_attributes);
    robin_attr = 0;
    neumann_attr_tmp.SetSize(max_bdr_attributes);
    neumann_attr_tmp = 0;
    robin_attr_tmp.SetSize(max_bdr_attributes);
    robin_attr_tmp = 0;
}

/// Dirichlet BCS
void BCHandler::AddDirichletBC(Coefficient *coeff, Array<int> &attr)
{
    // Check size of attributes provided
    MFEM_ASSERT(attr.Size() == max_bdr_attributes, // Modified to >= to account for SubMesh indexing
                "Size of attributes array does not match mesh attributes.");

    // Append to the list of Dirichlet BCs
    dirichlet_dbcs.emplace_back(attr, coeff);

    // Check for duplicate
    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT((dirichlet_attr[i] && robin_attr[i] && neumann_attr[i] && neumann_vec_attr[i] && attr[i]) == 0,
                    "Duplicate boundary definition detected.");
        if (attr[i] == 1)
        {
            dirichlet_attr[i] = 1;
        }
    }

    // Output
    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Adding Temperature Dirichlet BC to boundary attributes: ";
        for (int i = 0; i < attr.Size(); ++i)
        {
            if (attr[i] == 1)
            {
                mfem::out << i + 1 << " ";
            }
        }
        mfem::out << std::endl;
    }
}

void BCHandler::AddDirichletBC(ScalarFuncT func, Array<int> &attr)
{
    AddDirichletBC(new FunctionCoefficient(func), attr);
}

void BCHandler::AddDirichletBC(double coeff_val, Array<int> &attr)
{
    auto coeff = new ConstantCoefficient(coeff_val);
    AddDirichletBC(coeff, attr);
}

void BCHandler::AddDirichletBC(Coefficient *coeff, int &attr)
{
    // Create array for attributes and mark given mark given mesh boundary
    dirichlet_attr_tmp = 0;
    dirichlet_attr_tmp[attr - 1] = 1;

    // Call AddDirichletBC accepting array of essential attributes
    AddDirichletBC(coeff, dirichlet_attr_tmp);
}

void BCHandler::AddDirichletBC(ScalarFuncT func, int &attr)
{
    AddDirichletBC(new FunctionCoefficient(func), attr);
}

void BCHandler::AddDirichletBC(double coeff_val, int &attr)
{
    auto coeff = new ConstantCoefficient(coeff_val);
    AddDirichletBC(coeff, attr);
}

/// Neumann BCS
void BCHandler::AddNeumannBC(Coefficient *coeff, Array<int> &attr)
{
    // Check size of attributes provided
    MFEM_ASSERT(attr.Size() == max_bdr_attributes,
                "Size of attributes array does not match mesh attributes.");

    // Append to the list of Neumann BCs
    neumann_bcs.emplace_back(attr, coeff);

    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT((dirichlet_attr[i] && robin_attr[i] && neumann_attr[i] && neumann_vec_attr[i] && attr[i]) == 0,
                    "Trying to enforce Neumann bc on dirichlet boundary.");
        if (attr[i] == 1)
        {
            neumann_attr[i] = 1;
        }
    }

    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Adding Neumann BC to boundary attributes: ";
        for (int i = 0; i < attr.Size(); ++i)
        {
            if (attr[i] == 1)
            {
                mfem::out << i + 1 << " ";
            }
        }
        mfem::out << std::endl;
    }
}

void BCHandler::AddNeumannBC(ScalarFuncT func, Array<int> &attr)
{
    AddNeumannBC(new FunctionCoefficient(func), attr);
}

void BCHandler::AddNeumannBC(Coefficient *coeff, int &attr)
{
    // Create array for attributes and mark given mark given mesh boundary
    neumann_attr_tmp = 0;
    neumann_attr_tmp[attr - 1] = 1;

    // Call AddDirichletBC accepting array of essential attributes
    AddNeumannBC(coeff, neumann_attr_tmp);
}

void BCHandler::AddNeumannBC(double val, int &attr)
{
    auto coeff = new ConstantCoefficient(val);
    AddNeumannBC(coeff, attr);
}

/// Neumann Vector BCS
void BCHandler::AddNeumannVectorBC(VectorCoefficient *coeff, Array<int> &attr)
{
    // Check size of attributes provided
    MFEM_ASSERT(attr.Size() == max_bdr_attributes,
                "Size of attributes array does not match mesh attributes.");

    // Append to the list of Neumann BCs
    neumann_vec_bcs.emplace_back(attr, coeff);

    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT((dirichlet_attr[i] && robin_attr[i] && neumann_attr[i] && neumann_vec_attr[i] && attr[i]) == 0,
                    "Trying to enforce Neumann bc on dirichlet boundary.");
        if (attr[i] == 1)
        {
            neumann_vec_attr[i] = 1;
        }
    }

    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Adding Neumann BC (vector) to boundary attributes: ";
        for (int i = 0; i < attr.Size(); ++i)
        {
            if (attr[i] == 1)
            {
                mfem::out << i + 1 << " ";
            }
        }
        mfem::out << std::endl;
    }
}

void BCHandler::AddNeumannVectorBC(VecFuncT func, Array<int> &attr)
{
    AddNeumannVectorBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void BCHandler::AddNeumannVectorBC(VectorCoefficient *coeff, int &attr)
{
    // Create array for attributes and mark given mark given mesh boundary
    neumann_attr_tmp = 0;
    neumann_attr_tmp[attr - 1] = 1;

    // Call AddDirichletBC accepting array of essential attributes
    AddNeumannVectorBC(coeff, neumann_attr_tmp);
}

void BCHandler::AddNeumannVectorBC(VecFuncT func, int &attr)
{
    AddNeumannVectorBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

/// Robin BCS
void BCHandler::AddRobinBC(Coefficient *h_coeff, Coefficient *T0_coeff, Array<int> &attr)
{
    // Check size of attributes provided
    MFEM_ASSERT(attr.Size() == max_bdr_attributes,
                "Size of attributes array does not match mesh attributes.");

    // Append to the list of Robin BCs
    robin_bcs.emplace_back(attr, h_coeff, T0_coeff);

    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT((dirichlet_attr[i] && robin_attr[i] && neumann_attr[i] && neumann_vec_attr[i] && attr[i]) == 0,
                    "Trying to enforce Robin bc on dirichlet boundary.");
        if (attr[i] == 1)
        {
            robin_attr[i] = 1;
        }
    }

    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Adding Robin BC to boundary attributes: ";
        for (int i = 0; i < attr.Size(); ++i)
        {
            if (attr[i] == 1)
            {
                mfem::out << i + 1 << " ";
            }
        }
        mfem::out << std::endl;
    }
}

void BCHandler::AddRobinBC(ScalarFuncT h_func, ScalarFuncT T0_func, Array<int> &attr)
{
    AddRobinBC(new FunctionCoefficient(h_func), new FunctionCoefficient(T0_func), attr);
}

void BCHandler::AddRobinBC(Coefficient *h_coeff, Coefficient *T0_coeff, int &attr)
{
    // Create array for attributes and mark given mark given mesh boundary
    robin_attr_tmp = 0;
    robin_attr_tmp[attr - 1] = 1;

    // Call AddDirichletBC accepting array of essential attributes
    AddRobinBC(h_coeff, T0_coeff, robin_attr_tmp);
}

void BCHandler::AddRobinBC(double h_val, double T0_val, int &attr)
{
    auto h_coeff = new ConstantCoefficient(h_val);
    auto T0_coeff = new ConstantCoefficient(T0_val);
    AddRobinBC(h_coeff, T0_coeff, attr);
}

/// Update time dependent boundary conditions
void BCHandler::UpdateTimeDirichletBCs(double new_time)
{
    for (auto &dirichlet_bc : dirichlet_dbcs)
    {
        dirichlet_bc.coeff->SetTime(new_time);
    }
}

void BCHandler::UpdateTimeNeumannBCs(double new_time)
{
    for (auto &neumann_bc : neumann_bcs)
    {
        neumann_bc.coeff->SetTime(new_time);
    }
}

void BCHandler::UpdateTimeNeumannVectorBCs(double new_time)
{
    for (auto &neumann_vec_bc : neumann_vec_bcs)
    {
        neumann_vec_bc.coeff->SetTime(new_time);
    }
}

void BCHandler::UpdateTimeRobinBCs(double new_time)
{
    for (auto &robin_bc : robin_bcs)
    {
        robin_bc.h_coeff->SetTime(new_time);
        robin_bc.T0_coeff->SetTime(new_time);
    }
}

void BCHandler::SetTime(double new_time)
{
    time = new_time;
    UpdateTimeDirichletBCs(new_time);
    UpdateTimeNeumannBCs(new_time);
    UpdateTimeNeumannVectorBCs(new_time);
    UpdateTimeRobinBCs(new_time);
}