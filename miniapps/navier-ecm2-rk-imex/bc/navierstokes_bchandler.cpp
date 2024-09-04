// BCHandler_NS.cpp

#include "navierstokes_bchandler.hpp"

using namespace mfem;
using namespace navier;

BCHandler::BCHandler(std::shared_ptr<ParMesh> mesh, bool verbose)
    : pmesh(mesh), verbose(verbose)
{
    // initialize vectors of essential attributes
    vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());
    vel_ess_attr = 0;
    vel_ess_attr_x.SetSize(pmesh->bdr_attributes.Max());
    vel_ess_attr_x = 0;
    vel_ess_attr_y.SetSize(pmesh->bdr_attributes.Max());
    vel_ess_attr_y = 0;
    vel_ess_attr_z.SetSize(pmesh->bdr_attributes.Max());
    vel_ess_attr_z = 0;
    trac_attr_tmp.SetSize(pmesh->bdr_attributes.Max());
    trac_attr_tmp = 0;
    pres_ess_attr.SetSize(pmesh->bdr_attributes.Max());
    pres_ess_attr = 0;
    traction_attr.SetSize(pmesh->bdr_attributes.Max());
    traction_attr = 0;
    ess_attr_tmp.SetSize(pmesh->bdr_attributes.Max());
    ess_attr_tmp = 0;
}

void BCHandler::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr)
{
    vel_dbcs.emplace_back(attr, coeff);

    // Check for duplicate
    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT(((vel_ess_attr[i] || vel_ess_attr_x[i] || vel_ess_attr_y[i] || vel_ess_attr_z[i]) && attr[i]) == 0,
                    "Duplicate boundary definition detected.");
        if (attr[i] == 1)
        {
            vel_ess_attr[i] = 1;
        }
    }

    // Output
    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Adding Velocity Dirichlet BC (full) to boundary attributes: ";
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

void BCHandler::AddVelDirichletBC(VecFuncT func, Array<int> &attr)
{
    AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void BCHandler::AddVelDirichletBC(Coefficient *coeff, Array<int> &attr, int &dir)
{
    // Add bc container to list of componentwise velocity bcs
    vel_dbcs_xyz.emplace_back(attr, coeff, dir);

    // Check for duplicate and add attributes for current bc to global list (for that specific component)
    for (int i = 0; i < attr.Size(); ++i)
    {
        switch (dir)
        {
        case 0: // x
            dir_string = "x";
            MFEM_ASSERT(((vel_ess_attr[i] || vel_ess_attr_x[i]) && attr[i]) == 0,
                        "Duplicate boundary definition for x component detected.");
            if (attr[i] == 1)
            {
                vel_ess_attr_x[i] = 1;
            }
            break;
        case 1: // y
            dir_string = "y";
            MFEM_ASSERT(((vel_ess_attr[i] || vel_ess_attr_y[i]) && attr[i]) == 0,
                        "Duplicate boundary definition for y component detected.");
            if (attr[i] == 1)
            {
                vel_ess_attr_y[i] = 1;
            }
            break;
        case 2: // z
            dir_string = "z";
            MFEM_ASSERT(((vel_ess_attr[i] || vel_ess_attr_z[i]) && attr[i]) == 0,
                        "Duplicate boundary definition for z component detected.");
            if (attr[i] == 1)
            {
                vel_ess_attr_z[i] = 1;
            }
            break;
        default:;
        }
    }

    // Output
    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Adding Velocity Dirichlet BC ( " << dir_string << " component) to boundary attributes: " << std::endl;
        for (int i = 0; i < attr.Size(); ++i)
        {
            if (attr[i] == 1)
            {
                mfem::out << i << ", ";
            }
        }
        mfem::out << std::endl;
    }
}

void BCHandler::AddVelDirichletBC(VectorCoefficient *coeff, int &attr)
{
    // Create array for attributes and mark given mark given mesh boundary
    ess_attr_tmp = 0;
    ess_attr_tmp[attr - 1] = 1;

    // Call AddVelDirichletBC accepting array of essential attributes
    AddVelDirichletBC(coeff, ess_attr_tmp);
}

void BCHandler::AddVelDirichletBC(VecFuncT func, int &attr)
{
    AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void BCHandler::AddVelDirichletBC(Coefficient *coeff, int &attr, int &dir)
{
    // Create array for attributes and mark given mark given mesh boundary
    ess_attr_tmp = 0;
    ess_attr_tmp[attr - 1] = 1;

    // Call AddVelDirichletBC accepting array of essential attributes
    AddVelDirichletBC(coeff, ess_attr_tmp, dir);
}

void BCHandler::AddPresDirichletBC(Coefficient *coeff, Array<int> &attr)
{
    pres_dbcs.emplace_back(attr, coeff);

    // Check for duplicate
    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT((pres_ess_attr[i] && attr[i]) == 0,
                    "Duplicate boundary definition detected.");
        if (attr[i] == 1)
        {
            pres_ess_attr[i] = 1;
        }
    }

    // Output
    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Adding Pressure Dirichlet BC (full) to boundary attributes: ";
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

void BCHandler::AddPresDirichletBC(ScalarFuncT func, Array<int> &attr)
{
    AddPresDirichletBC(new FunctionCoefficient(func), attr);
}

void BCHandler::AddPresDirichletBC(Coefficient *coeff, int &attr)
{
    // Create array for attributes and mark given mark given mesh boundary
    ess_attr_tmp = 0;
    ess_attr_tmp[attr - 1] = 1;

    // Call AddVelDirichletBC accepting array of essential attributes
    AddPresDirichletBC(coeff, ess_attr_tmp);
}

void BCHandler::AddPresDirichletBC(ScalarFuncT func, int &attr)
{
    AddPresDirichletBC(new FunctionCoefficient(func), attr);
}

void BCHandler::AddTractionBC(VectorCoefficient *coeff, Array<int> &attr)
{
    traction_bcs.emplace_back(attr, coeff);

    for (int i = 0; i < attr.Size(); ++i)
    {
        MFEM_ASSERT(((vel_ess_attr[i] || vel_ess_attr_x[i] || vel_ess_attr_y[i] || vel_ess_attr_z[i]) && attr[i]) == 0,
                    "Trying to enforce traction bc on dirichlet boundary.");
        if (attr[i] == 1)
        {
            traction_attr[i] = 1;
        }
    }

    if (verbose && pmesh->GetMyRank() == 0)
    {
        mfem::out << "Adding Traction (Neumann) BC to boundary attributes: ";
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

void BCHandler::AddTractionBC(VecFuncT func, Array<int> &attr)
{
    AddTractionBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void BCHandler::AddTractionBC(VectorCoefficient *coeff, int &attr)
{
    // Create array for attributes and mark given mark given mesh boundary
    trac_attr_tmp = 0;
    trac_attr_tmp[attr - 1] = 1;

    // Call AddVelDirichletBC accepting array of essential attributes
    AddTractionBC(coeff, trac_attr_tmp);
}

void BCHandler::UpdateTimeVelocityBCs(double new_time)
{
    for (auto &vel_dbc : vel_dbcs)
    {
        vel_dbc.coeff->SetTime(new_time);
    }

    for (auto &vel_dbc : vel_dbcs_xyz)
    {
        vel_dbc.coeff->SetTime(new_time);
    }
}

void BCHandler::UpdateTimePressureBCs(double new_time)
{
    for (auto &pres_dbc : pres_dbcs)
    {
        pres_dbc.coeff->SetTime(new_time);
    }
}