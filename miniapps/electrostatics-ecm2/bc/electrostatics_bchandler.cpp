// BCHandler_NS.cpp

#include "electrostatics_bchandler.hpp"

using namespace mfem;
using namespace electrostatics;

BCHandler::BCHandler(std::shared_ptr<ParMesh> mesh, bool verbose)
    : pmesh(mesh), verbose(verbose)
{
    max_bdr_attributes = pmesh->bdr_attributes.Max();

    // initialize vectors of essential attributes
    dirichlet_attr.SetSize(max_bdr_attributes);
    dirichlet_attr = 0;
    dirichlet_EField_attr.SetSize(max_bdr_attributes);
    dirichlet_EField_attr = 0;
    dirichlet_attr_tmp.SetSize(max_bdr_attributes);
    dirichlet_attr_tmp = 0;
    neumann_attr.SetSize(max_bdr_attributes);
    neumann_attr = 0;
    neumann_vec_attr.SetSize(max_bdr_attributes);
    neumann_vec_attr = 0;
    neumann_attr_tmp.SetSize(max_bdr_attributes);
    neumann_attr_tmp = 0;
}

void BCHandler::AddDirichletBC(Coefficient *coeff, Array<int> &attr)
{
    // Check size of attributes provided
    MFEM_ASSERT(attr.Size() == max_bdr_attributes,
                "Size of attributes array does not match mesh attributes.");

    dirichlet_dbcs.emplace_back(attr, coeff);

    // Check for duplicate
    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT((dirichlet_attr[i] && dirichlet_EField_attr[i] && neumann_attr[i] && neumann_vec_attr[i] && attr[i]) == 0,
                    "Duplicate boundary definition detected.");
        if (attr[i] == 1)
        {
            dirichlet_attr[i] = 1;
        }
    }

    // Output
    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Adding Potential Dirichlet BC to boundary attributes: ";
        for (int i = 0; i < attr.Size(); ++i)
        {
            if (attr[i] == 1)
            {
                mfem::out << i+1 << " ";
            }
        }
        mfem::out << std::endl;
    }
}

void BCHandler::AddDirichletBC(ScalarFuncT func, Array<int> &attr)
{
    AddDirichletBC(new FunctionCoefficient(func), attr);
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

void BCHandler::AddDirichletEFieldBC(Vector &EField, Array<int> &attr)
{

    MFEM_ASSERT( EField.Size() == pmesh->Dimension(),
                "Electric field vector doesn't match mesh dimension.");

    auto func = [EField](const Vector &x)
    {
        double phi_val = 0.0;

        for (int i = 0; i < x.Size(); i++)
        {
            phi_val -= x(i) * EField(i);
        }

        return phi_val;
    };

    auto coeff = new FunctionCoefficient(func);

    dirichlet_EField_dbcs.emplace_back(attr, coeff);

    // Check for duplicate
    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT((dirichlet_attr[i] && dirichlet_EField_attr[i] && neumann_attr[i] && neumann_vec_attr[i] && attr[i]) == 0,
                    "Duplicate boundary definition detected.");
        if (attr[i] == 1)
        {
            dirichlet_attr[i] = 1;
        }
    }

    // Output
    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Adding Potential Dirichlet BC (Uniform EField) to boundary attributes: ";
        for (int i = 0; i < attr.Size(); ++i)
        {
            if (attr[i] == 1)
            {
                mfem::out << i+1 << " ";
            }
        }
        mfem::out << std::endl;
    }
}

void BCHandler::AddDirichletEFieldBC(Vector &EField, int &attr)
{
    // Create array for attributes and mark given mark given mesh boundary
    dirichlet_attr_tmp = 0;
    dirichlet_attr_tmp[attr - 1] = 1;

    // Call AddDirichletBC accepting array of essential attributes
    AddDirichletEFieldBC(EField, dirichlet_attr_tmp);
}


void BCHandler::AddNeumannBC(Coefficient *coeff, Array<int> &attr)
{
    // Check size of attributes provided
    MFEM_ASSERT(attr.Size() == max_bdr_attributes,
                "Size of attributes array does not match mesh attributes.");

    neumann_bcs.emplace_back(attr, coeff);

    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT((dirichlet_attr[i] && dirichlet_EField_attr[i] && neumann_attr[i] && neumann_vec_attr[i] && attr[i]) == 0,
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
                mfem::out << i+1 << " ";
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
void BCHandler::AddNeumannVectorBC(VectorCoefficient *coeff, Array<int> &attr, bool own)
{
    // Check size of attributes provided
    MFEM_ASSERT(attr.Size() == max_bdr_attributes,
                "Size of attributes array does not match mesh attributes.");

    // Append to the list of Neumann BCs
    neumann_vec_bcs.emplace_back(attr, coeff, own);

    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT((dirichlet_attr[i] && dirichlet_EField_attr[i] && neumann_attr[i] && neumann_vec_attr[i] && attr[i]) == 0,
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

void BCHandler::AddNeumannVectorBC(VecFuncT func, Array<int> &attr, bool own)
{
    AddNeumannVectorBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr, own);
}

void BCHandler::AddNeumannVectorBC(VectorCoefficient *coeff, int &attr, bool own)
{
    // Create array for attributes and mark given mark given mesh boundary
    neumann_attr_tmp = 0;
    neumann_attr_tmp[attr - 1] = 1;

    // Call AddDirichletBC accepting array of essential attributes
    AddNeumannVectorBC(coeff, neumann_attr_tmp, own);
}

void BCHandler::AddNeumannVectorBC(VecFuncT func, int &attr, bool own)
{
    AddNeumannVectorBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr, own);
}

void BCHandler::UpdateTimeDirichletBCs(double new_time)
{
    for (auto &dirichlet_dbc : dirichlet_dbcs)
    {
        dirichlet_dbc.coeff->SetTime(new_time);
    }

    for (auto &dirichlet_dbc : dirichlet_EField_dbcs)
    {
        dirichlet_dbc.coeff->SetTime(new_time);
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