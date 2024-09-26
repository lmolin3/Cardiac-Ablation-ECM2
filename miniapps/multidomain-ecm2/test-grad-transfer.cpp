// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// A 3D domain comprised of an outer box with a cylinder shaped inside is used.
//
// This test is based on the test case in source/tests/unit/fem/test_gslib.cpp
// We test the transfer of the temperature field gradient from the cylinder to the box.
//
// Compute k∇T_cylinder on cylinder wall and transfer it to the box wall.

#include "mfem.hpp"
#include "utils.hpp"

#include <fstream>
#include <iostream>
#include <memory>

using namespace mfem;

using namespace ecm2_utils;

IdentityMatrixCoefficient *Id = NULL;

int main(int argc, char *argv[])
{
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// 1. Initialize MPI and Hypre
    ///////////////////////////////////////////////////////////////////////////////////////////////

    Mpi::Init();
    Hypre::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// 2. Parse command-line options.
    ///////////////////////////////////////////////////////////////////////////////////////////////

    // FE
    int order = 3;
    // Mesh
    int serial_ref_levels = 0;
    int parallel_ref_levels = 0;
    // Postprocessing
    const char *outfolder = "";

    // Transfer type
    int transfer_type = 0;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                   "Number of serial refinement levels.");
    args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                   "Number of parallel refinement levels.");
    args.AddOption(&outfolder, "-of", "--out-folder",
                   "Output folder.");
    args.AddOption(&transfer_type, "-tt", "--transfer-type",
                   "Transfer type: 0 - SubMesh Transfer Map, 1 - GSLIB");
    args.ParseCheck();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// 3. Create serial Mesh and parallel
    ///////////////////////////////////////////////////////////////////////////////////////////////

    Mesh *serial_mesh = new Mesh("multidomain-hex.mesh");
    ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
    delete serial_mesh;

    parent_mesh.UniformRefinement();

    // Create the sub-domains and accompanying Finite Element spaces from
    // corresponding attributes. This specific mesh has two domain attributes and
    // 9 boundary attributes.
    Array<int> cylinder_domain_attributes(1);
    cylinder_domain_attributes[0] = 1;

    Array<int> block_domain_attributes(1);
    block_domain_attributes[0] = 2;

    auto cylinder_submesh =
        ParSubMesh::CreateFromDomain(parent_mesh, cylinder_domain_attributes);

    auto block_submesh = ParSubMesh::CreateFromDomain(parent_mesh,
                                                      block_domain_attributes);

    // Create the Finite Element spaces and grid functions
    H1_FECollection fec(order, parent_mesh.Dimension());
    ParFiniteElementSpace fes_cylinder(&cylinder_submesh, &fec);
    ParFiniteElementSpace fes_cylinder_grad(&cylinder_submesh, &fec, parent_mesh.Dimension());

    ParFiniteElementSpace fes_block(&block_submesh, &fec);
    ParFiniteElementSpace fes_block_grad(&block_submesh, &fec, parent_mesh.Dimension());

    ParGridFunction temperature_cylinder_gf(&fes_cylinder);
    ParGridFunction temperature_cylinder_grad_gf(&fes_cylinder_grad);
    ParGridFunction temperature_cylinder_grad_exact_gf(&fes_cylinder_grad);
    temperature_cylinder_gf = 0.0;
    temperature_cylinder_grad_gf = 0.0;
    temperature_cylinder_grad_exact_gf = 0.0;

    ParGridFunction temperature_block_gf(&fes_block);
    ParGridFunction temperature_custom_block_gf(&fes_block);
    ParGridFunction temperature_block_grad_gf(&fes_block_grad);
    ParGridFunction temperature_block_K_grad_gf(&fes_block_grad);
    temperature_block_gf = 0.0;
    temperature_custom_block_gf = 0.0;
    temperature_block_grad_gf = 0.0;

    // Create transfer map
    auto grad_temperature_cylinder_to_block_map = ParSubMesh::CreateTransferMap(temperature_cylinder_grad_gf, temperature_block_grad_gf);
    auto temperature_cylinder_to_block_map = ParSubMesh::CreateTransferMap(temperature_cylinder_gf, temperature_block_gf);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// 4. Set up coefficients
    ///////////////////////////////////////////////////////////////////////////////////////////////

    int sdim = parent_mesh.SpaceDimension();
    auto Id = new IdentityMatrixCoefficient(sdim);

    Array<int> attr_cyl(0), attr_block(0);
    attr_cyl.Append(1), attr_block.Append(2);

    double kval_cyl = 1.0;   // W/mK
    double kval_block = 1.0; // W/mK

    // Conductivity
    Array<MatrixCoefficient *> coefs_k_cyl(0);
    coefs_k_cyl.Append(new ScalarMatrixProductCoefficient(kval_cyl, *Id));
    PWMatrixCoefficient *Kappa_cyl = new PWMatrixCoefficient(sdim, attr_cyl, coefs_k_cyl);

    Array<MatrixCoefficient *> coefs_k_block(0);
    coefs_k_block.Append(new ScalarMatrixProductCoefficient(kval_block, *Id));
    PWMatrixCoefficient *Kappa_block = new PWMatrixCoefficient(sdim, attr_block, coefs_k_block);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// 5. Set up coefficient for transfer
    ///////////////////////////////////////////////////////////////////////////////////////////////

    // Define Temperature analytic field
    // f(x,y,z) = x^2 + y^2
    auto func = [](const Vector &x)
    {
        const int dim = x.Size();
        double res = 0.0;
        res = std::pow(x(0), 2) + std::pow(x(1), 2);
        return res;
    };

    // Define Gradient of Temperature analytic field
    // ∇f(x,y,z) = [2*x,2*y,0]
    auto func_grad = [](const Vector &x, Vector &p)
    {
        const int dim = x.Size();
        p.SetSize(dim);
        p(0) = 2.0 * x(0);
        p(1) = 2.0 * x(1);
        p(2) = 0.0;
    };

    Array<int> block_outer_wall_attributes(block_submesh.bdr_attributes.Max());
    block_outer_wall_attributes = 0;
    block_outer_wall_attributes[0] = 1;
    block_outer_wall_attributes[1] = 1;
    block_outer_wall_attributes[2] = 1;
    block_outer_wall_attributes[3] = 1;

    Array<int> block_inner_wall_attributes(block_submesh.bdr_attributes.Max());
    block_inner_wall_attributes = 0;
    block_inner_wall_attributes[8] = 1;

    Array<int> inner_cylinder_wall_attributes(cylinder_submesh.bdr_attributes.Max());
    inner_cylinder_wall_attributes = 0;
    inner_cylinder_wall_attributes[8] = 1;

    // Setup ouput
    ParaViewDataCollection paraview_dc_cylinder("Heat-Cylinder", &cylinder_submesh);
    ParaViewDataCollection paraview_dc_block("Heat-Block", &block_submesh);

    paraview_dc_cylinder.SetPrefixPath(outfolder);
    paraview_dc_cylinder.SetDataFormat(VTKFormat::BINARY);
    paraview_dc_cylinder.SetCompressionLevel(9);
    paraview_dc_cylinder.RegisterField("Temperature", &temperature_cylinder_gf);
    paraview_dc_cylinder.RegisterField("Temperature-Gradient", &temperature_cylinder_grad_gf);
    paraview_dc_cylinder.RegisterField("Temperature-Gradient-Exact", &temperature_cylinder_grad_exact_gf);

    paraview_dc_block.SetPrefixPath(outfolder);
    paraview_dc_block.SetDataFormat(VTKFormat::BINARY);
    paraview_dc_block.SetCompressionLevel(9);
    paraview_dc_block.RegisterField("Temperature", &temperature_block_gf);
    paraview_dc_block.RegisterField("Temperature-Gradient", &temperature_block_grad_gf);
    if (transfer_type == 0)
    {
        paraview_dc_block.RegisterField("K-Grad-Temperature", &temperature_block_K_grad_gf);
    }
    else if (transfer_type == 1)
    {
        paraview_dc_block.RegisterField("Temperature-Custom", &temperature_custom_block_gf);
    }

    if (order > 1)
    {
        paraview_dc_cylinder.SetHighOrderOutput(true);
        paraview_dc_cylinder.SetLevelsOfDetail(order);

        paraview_dc_block.SetHighOrderOutput(true);
        paraview_dc_block.SetLevelsOfDetail(order);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// 6. Setup coefficient
    ///////////////////////////////////////////////////////////////////////////////////////////////

    // Outer wall
    ConstantCoefficient Tout(1.0);
    temperature_block_gf.ProjectBdrCoefficient(Tout, block_outer_wall_attributes);
    temperature_custom_block_gf.ProjectBdrCoefficient(Tout, block_outer_wall_attributes);

    // Cylinder exact solution
    FunctionCoefficient f(func);
    temperature_cylinder_gf.ProjectCoefficient(f);

    // Cylinder gradient exact solution
    VectorFunctionCoefficient f_grad(cylinder_submesh.Dimension(), func_grad);
    temperature_cylinder_grad_exact_gf.ProjectCoefficient(f_grad);

    // Coefficient to compute the gradient of the temperature field on the cylinder
    GradientGridFunctionCoefficient temperature_gradient_cylinder_coeff(&temperature_cylinder_gf);
    VectorGridFunctionCoefficient *grad_temperature_wall_block_coeff = new VectorGridFunctionCoefficient(&temperature_block_grad_gf); // uses grad computed and transferred from cylinder
    MatrixVectorProductCoefficient *k_grad_T_wall = new MatrixVectorProductCoefficient(*Kappa_block, *grad_temperature_wall_block_coeff);
    // ScalarVectorProductCoefficient *neg_k_grad_T_wall = new ScalarVectorProductCoefficient(-1.0, *k_grad_T_wall);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// 7. Transfer the gradient of the temperature field on the cylinder onto the box
    ///////////////////////////////////////////////////////////////////////////////////////////////

    Vector qoi_dst;

    if (transfer_type == 0)
    {
        // Compute gradient from the temperature field on the cylinder
        temperature_cylinder_grad_gf.ProjectCoefficient(temperature_gradient_cylinder_coeff);

        // Transfer the gradient from the cylinder to the box
        grad_temperature_cylinder_to_block_map.Transfer(temperature_cylinder_grad_gf, temperature_block_grad_gf);
        temperature_block_K_grad_gf.ProjectBdrCoefficient(*k_grad_T_wall, block_inner_wall_attributes);

        // Transfer the temperature field from the cylinder to the box
        temperature_cylinder_to_block_map.Transfer(temperature_cylinder_gf, temperature_block_gf);
    }
    else if (transfer_type == 1)
    {

        // 1. Find points on the destination mesh (box)
        // Get all BE with the attribute corresponding to the inner wall of the box (at what index they are in mesh.GetNBE)
        std::vector<int> block_element_idx;
        Vector block_element_coords;
        ComputeBdrQuadraturePointsCoords(block_inner_wall_attributes, fes_block, block_element_idx, block_element_coords);

        // 2. Setup GSLIB finder on the source mesh (cylinder)
        FindPointsGSLIB finder(MPI_COMM_WORLD);
        finder.Setup(cylinder_submesh);
        finder.FindPoints(block_element_coords, Ordering::byVDIM);

        // 3. Compute QoI (gradient of the temperature field) on the source mesh (cylinder)
        // Send information to MPI ranks that own the element corresponding to each point.
        int qoi_size_on_qp = sdim;
        Vector qoi_src; // QoI vector, used to store qoi_src in qoi_func and in call to GSLIB interpolator

        auto qoi_func = [&](ElementTransformation &Tr, int pt_idx, int num_pts)
        {
            Vector gradloc(qoi_src.GetData() + pt_idx * sdim, sdim);
            temperature_cylinder_gf.GetGradient(Tr, gradloc);
        };

        GSLIBInterpolate(finder, fes_cylinder, qoi_func, qoi_src, qoi_dst, qoi_size_on_qp);

        // 4. Compute QoI and update grid functions on the destination mesh (box)
        const IntegrationRule &ir_face = (fes_block_grad.GetBE(block_element_idx[0]))->GetNodes();
        auto box_grad = mfem::Reshape(qoi_dst.ReadWrite(), sdim,
                                      ir_face.GetNPoints(), block_element_idx.size());

        int dof, idx, be_idx;
        Vector loc_values, shape;
        int vdim = fes_block_grad.GetVDim();
        for (int be = 0; be < block_element_idx.size(); be++) // iterate over each BE on interface boundary and construct FE value from quadrature point
        {
            Array<int> vdofs;
            be_idx = block_element_idx[be];
            fes_block_grad.GetBdrElementVDofs(be_idx, vdofs);
            const FiniteElement *fe = fes_block_grad.GetBE(be_idx);
            dof = fe->GetDof();
            loc_values.SetSize(dof * sdim);
            // shape.SetSize(dof);
            // const IntegrationRule &ir = fe->GetNodes();
            auto ordering = fes_block_grad.GetOrdering();
            for (int qp = 0; qp < dof; qp++)
            {
                // const IntegrationPoint &ip = ir.IntPoint(qp);
                //  fe->CalcShape(ip, shape);
                for (int d = 0; d < sdim; d++)
                {
                    idx = ordering == Ordering::byVDIM ? qp * sdim + d : dof * d + qp;
                    loc_values(idx) = box_grad(d, qp, be);
                    loc_values(idx) *= kval_block;
                    // loc_values(idx) *= -1.0;
                }
            }
            temperature_block_grad_gf.SetSubVector(vdofs, loc_values);
        }

        // Transfer the temperature field from the cylinder to the box
        qoi_src.SetSize(0); // QoI vector, used to store qoi_src in qoi_func and in call to GSLIB interpolator
        qoi_dst.SetSize(0);
        auto qoi_func_2 = [&](ElementTransformation &Tr, int pt_idx, int num_pts)
        {
            double T_loc = temperature_cylinder_gf.GetValue(Tr);
            qoi_src(pt_idx) = T_loc;
        };

        GSLIBInterpolate(finder, fes_cylinder, qoi_func_2, qoi_src, qoi_dst);

        // 4. Compute QoI and update grid functions on the destination mesh (box)
        const IntegrationRule &ir_face_T = (fes_block.GetBE(block_element_idx[0]))->GetNodes();
        auto box_T = mfem::Reshape(qoi_dst.ReadWrite(), ir_face_T.GetNPoints(), block_element_idx.size());
        int idx_global;
        vdim = fes_block.GetVDim();
        for (int be = 0; be < block_element_idx.size(); be++) // iterate over each BE on interface boundary and construct FE value from quadrature point
        {
            Array<int> vdofs;
            be_idx = block_element_idx[be];
            fes_block.GetBdrElementVDofs(be_idx, vdofs);
            const FiniteElement *fe = fes_block.GetBE(be_idx);
            dof = fe->GetDof();
            loc_values.SetSize(dof * sdim);
            // shape.SetSize(dof);
            // const IntegrationRule &ir = fe->GetNodes();
            auto ordering = fes_block.GetOrdering();
            for (int qp = 0; qp < dof; qp++)
            {
                // const IntegrationPoint &ip = ir.IntPoint(qp);
                //  fe->CalcShape(ip, shape);
                for (int d = 0; d < sdim; d++)
                {
                    idx = ordering == Ordering::byVDIM ? qp * sdim + d : dof * d + qp;
                    idx_global = be * dof * sdim + qp * sdim + d; // ordering byVDIM
                    loc_values(idx) = box_T(qp, be);
                }
            }
            temperature_custom_block_gf.SetSubVector(vdofs, loc_values);
        }

        // Interpolate temperature
        // finder.SetL2AvgType(FindPointsGSLIB::ARITHMETIC);
        qoi_dst.SetSize(0);
        finder.Interpolate(block_element_coords, temperature_cylinder_gf, qoi_dst, Ordering::byVDIM);

        // 4. Compute QoI and update grid functions on the destination mesh (box)
        auto box_T2 = mfem::Reshape(qoi_dst.ReadWrite(), ir_face_T.GetNPoints(), block_element_idx.size());
        vdim = fes_block.GetVDim();
        for (int be = 0; be < block_element_idx.size(); be++) // iterate over each BE on interface boundary and construct FE value from quadrature point
        {
            Array<int> vdofs;
            be_idx = block_element_idx[be];
            fes_block.GetBdrElementVDofs(be_idx, vdofs);
            const FiniteElement *fe = fes_block.GetBE(be_idx);
            dof = fe->GetDof();
            loc_values.SetSize(dof * sdim);
            // shape.SetSize(dof);
            // const IntegrationRule &ir = fe->GetNodes();
            auto ordering = fes_block.GetOrdering();
            for (int qp = 0; qp < dof; qp++)
            {
                // const IntegrationPoint &ip = ir.IntPoint(qp);
                //  fe->CalcShape(ip, shape);
                for (int d = 0; d < sdim; d++)
                {
                    idx = ordering == Ordering::byVDIM ? qp * sdim + d : dof * d + qp;
                    idx_global = be * dof * sdim + qp * sdim + d; // ordering byVDIM
                    loc_values(idx) = box_T(qp, be);
                }
            }
            temperature_block_gf.SetSubVector(vdofs, loc_values);
        }
    }

    // Output
    if (Mpi::Root())
    {
        std::cout << "Writing Paraview files ..." << std::endl;
    }
    paraview_dc_block.SetCycle(0);
    paraview_dc_block.SetTime(0.0);
    paraview_dc_block.Save();

    paraview_dc_cylinder.SetCycle(0);
    paraview_dc_cylinder.SetTime(0.0);
    paraview_dc_cylinder.Save();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// 6. Interpolate the gradient of the temperature field on the cylinder onto the box
    ///////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// 7. Cleanup
    ///////////////////////////////////////////////////////////////////////////////////////////////

    delete Kappa_cyl;
    delete Kappa_block;

    // Delete the MatrixCoefficient objects at the end of main
    for (int i = 0; i < coefs_k_block.Size(); i++)
    {
        delete coefs_k_block[i];
    }

    for (int i = 0; i < coefs_k_cyl.Size(); i++)
    {
        delete coefs_k_cyl[i];
    }

    delete Id;

    return 0;
}

void ComputeBdrQPCoords(ParMesh &mesh, Array<int> bdry_attributes, ParFiniteElementSpace &fes, std::vector<int> &bdry_element_idx, Vector &bdry_element_coords)
{
    // Mesh space dimension
    int sdim = mesh.SpaceDimension();

    // Get the boundary elements with the specified attributes
    bdry_element_idx.clear();
    for (int be = 0; be < mesh.GetNBE(); be++)
    {
        const int bdr_el_attr = mesh.GetBdrAttribute(be);
        if (bdry_attributes[bdr_el_attr - 1] == 0)
        {
            continue;
        }
        bdry_element_idx.push_back(be);
    }

    // Extract the coordinates of the quadrature points for each selected boundary element
    const IntegrationRule &ir_face = (fes.GetBE(bdry_element_idx[0]))->GetNodes();
    bdry_element_coords.SetSize(bdry_element_idx.size() *
                                ir_face.GetNPoints() * sdim);
    bdry_element_coords = 0.0;

    auto pec = mfem::Reshape(bdry_element_coords.ReadWrite(), sdim,
                             ir_face.GetNPoints(), bdry_element_idx.size());

    for (int be = 0; be < bdry_element_idx.size(); be++)
    {
        int be_idx = bdry_element_idx[be];
        const FiniteElement *fe = fes.GetBE(be_idx);
        ElementTransformation *Tr = fes.GetBdrElementTransformation(be_idx);
        const IntegrationRule &ir_face = fe->GetNodes();

        for (int qp = 0; qp < ir_face.GetNPoints(); qp++)
        {
            const IntegrationPoint &ip = ir_face.IntPoint(qp);

            Vector x(sdim);
            Tr->Transform(ip, x);

            for (int d = 0; d < sdim; d++)
            {
                pec(d, qp, be) = x(d);
            }
        }
    }

    return;
}

void GSLIBInterpolate(FindPointsGSLIB &finder, ParFiniteElementSpace &fes, qoi_func_t qoi_func, Vector &qoi_src, Vector &qoi_dst, int qoi_size_on_qp)
{
    // Extarct space dimension and FE space from the mesh
    int sdim = (fes.GetParMesh())->SpaceDimension();

    // Distribute internal GSLIB info to the corresponding mpi-rank for each point.
    Array<unsigned int>
        recv_elem,
        recv_code;   // Element and GSLIB code
    Vector recv_rst; // (Reference) coordinates of the quadrature points
    finder.DistributePointInfoToOwningMPIRanks(recv_elem, recv_rst, recv_code);
    int npt_recv = recv_elem.Size();

    // Compute qoi locally (on source side)
    qoi_src.SetSize(npt_recv * qoi_size_on_qp);
    for (int i = 0; i < npt_recv; i++)
    {
        // Get the element index
        const int e = recv_elem[i];

        // Get the quadrature point
        IntegrationPoint ip;
        ip.Set3(recv_rst(sdim * i + 0), recv_rst(sdim * i + 1),
                recv_rst(sdim * i + 2));

        // Get the element transformation
        ElementTransformation *Tr = fes.GetElementTransformation(e);
        Tr->SetIntPoint(&ip);

        // Compute the qoi_src at quadrature point (it will change the qoi_src vector)
        qoi_func(*Tr, i, npt_recv);
    }

    // Transfer the QoI from the source mesh to the destination mesh at quadrature points
    finder.DistributeInterpolatedValues(qoi_src, qoi_size_on_qp, Ordering::byVDIM, qoi_dst);
}