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

   // Partition type:
   // 0) METIS_PartGraphRecursive (sorted neighbor lists)
   // 1) METIS_PartGraphKway      (sorted neighbor lists) (default)
   // 2) METIS_PartGraphVKway     (sorted neighbor lists)
   // 3) METIS_PartGraphRecursive
   // 4) METIS_PartGraphKway
   // 5) METIS_PartGraphVKway
   int partition_type = 1;
   int np = num_procs;
   int *partitioning = serial_mesh->GeneratePartitioning(np, partition_type);

    ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh);
    ExportMeshwithPartitioning(outfolder, *serial_mesh, partitioning);
    delete[] partitioning;
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
    ParGridFunction temperature_block_grad_gf(&fes_block_grad);
    temperature_block_gf = 0.0;
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
    ScalarMatrixProductCoefficient *Kappa_cyl = new ScalarMatrixProductCoefficient(kval_cyl, *Id);
    ScalarMatrixProductCoefficient *Kappa_block = new ScalarMatrixProductCoefficient(kval_block, *Id);

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
    paraview_dc_cylinder.RegisterField("Temperature-Gradient-Exact", &temperature_cylinder_grad_exact_gf);

    paraview_dc_block.SetPrefixPath(outfolder);
    paraview_dc_block.SetDataFormat(VTKFormat::BINARY);
    paraview_dc_block.SetCompressionLevel(9);
    paraview_dc_block.RegisterField("Temperature", &temperature_block_gf);
    paraview_dc_block.RegisterField("Temperature-Gradient", &temperature_block_grad_gf);

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

    // Cylinder exact solution
    FunctionCoefficient f(func);
    temperature_cylinder_gf.ProjectCoefficient(f);

    // Cylinder gradient exact solution
    VectorFunctionCoefficient f_grad(cylinder_submesh.Dimension(), func_grad);
    temperature_cylinder_grad_exact_gf.ProjectCoefficient(f_grad);

    // Coefficient to compute the gradient of the temperature field on the cylinder
    GradientGridFunctionCoefficient temperature_gradient_cylinder_coeff(&temperature_cylinder_gf);
    VectorGridFunctionCoefficient *grad_temperature_wall_block_coeff = new VectorGridFunctionCoefficient(&temperature_block_grad_gf); // uses grad computed and transferred from cylinder

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

        // Transfer the temperature field from the cylinder to the box
        temperature_cylinder_to_block_map.Transfer(temperature_cylinder_gf, temperature_block_gf);
    }
    else if (transfer_type == 1)
    {

        // 1. Find points on the destination mesh (box)
        // Get all BE with the attribute corresponding to the inner wall of the box (at what index they are in mesh.GetNBE)
        std::vector<int> block_element_idx;
        Vector block_element_coords;
        ecm2_utils::ComputeBdrQuadraturePointsCoords(block_inner_wall_attributes, fes_block, block_element_idx, block_element_coords);

        // 2. Setup GSLIB finder on the source mesh (cylinder)
        FindPointsGSLIB finder(MPI_COMM_WORLD);
        finder.Setup(cylinder_submesh);
        finder.FindPoints(block_element_coords, Ordering::byVDIM);

        // 3. Compute QoI (gradient of the temperature field) on the source mesh (cylinder)
        // Send information to MPI ranks that own the element corresponding to each point.
        int vdim = fes_block_grad.GetVDim();
        int qoi_size_on_qp = vdim;
        Vector qoi_src; // QoI vector, used to store qoi_src in cylinder_grad_func and in call to GSLIB interpolator

        auto cylinder_grad_func = [&](ElementTransformation &Tr, int pt_idx, const IntegrationPoint &ip)
        {
            Vector gradloc(qoi_src.GetData() + pt_idx * vdim, vdim);
            temperature_cylinder_gf.GetGradient(Tr, gradloc);
        };

        // 4. Compute QoI and update grid functions on the destination mesh (box)
        ecm2_utils::GSLIBInterpolate(finder, fes_cylinder, cylinder_grad_func, qoi_src, qoi_dst, qoi_size_on_qp);
        ecm2_utils::TransferQoIToDest(block_element_idx, fes_block_grad, qoi_dst, temperature_block_grad_gf);

        // Interpolate temperature
        qoi_dst.SetSize(0);
        finder.Interpolate(block_element_coords, temperature_cylinder_gf, qoi_dst, Ordering::byVDIM);
        ecm2_utils::TransferQoIToDest(block_element_idx, fes_block, qoi_dst, temperature_block_gf);

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
        /// 6. Cleanup
        ///////////////////////////////////////////////////////////////////////////////////////////////

        delete Kappa_cyl;
        delete Kappa_block;
        delete Id;

        return 0;
    }
}
