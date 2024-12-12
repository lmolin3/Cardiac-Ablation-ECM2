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
// Sample run: 
//             mpirun -np 10 ./test-transfer-two-domains-overlapping-partition -o 1 -of ./Output/TransferOverlappingPartition

#include "mfem.hpp"
#include "utils.hpp"

#include <fstream>
#include <sstream>
#include <sys/stat.h> // Include for mkdir
#include <iostream>
#include <memory>

using namespace mfem;

using InterfaceTransfer = ecm2_utils::InterfaceTransfer;
using TransferBackend = InterfaceTransfer::Backend;

void ExportMeshwithPartitioning(const std::string &outfolder, Mesh &mesh, const int *partitioning_);

static constexpr real_t Tfluid = 303.15;    // Fluid temperature
static constexpr real_t Tcylinder = 293.15; // Cylinder temperature

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
   int order = 1;
   // Mesh
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   // Postprocessing
   bool paraview = true;
   const char *outfolder = "";

   OptionsParser args(argc, argv);
   // FE
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   // Mesh
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   // Postprocessing
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview", "--no-paraview",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&outfolder, "-of", "--out-folder",
                  "Output folder.");

   args.ParseCheck();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh and parallel
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nLoading mesh... \033[0m";

   // Load serial mesh
   Mesh *serial_mesh = new Mesh("../../data/three-domains.msh");
   int sdim = serial_mesh->SpaceDimension();

   for (int l = 0; l < serial_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }

   serial_mesh->EnsureNodes();
   
   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   // Generate mesh partitioning

   if (Mpi::Root())
      mfem::out << "Generating partitioning and creating parallel mesh... \033[0m";

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

   // Create parallel mesh
   ParMesh parent_mesh = ParMesh(MPI_COMM_WORLD, *serial_mesh, partitioning, partition_type);
   ExportMeshwithPartitioning(outfolder, *serial_mesh, partitioning);
   delete[] partitioning;
   delete serial_mesh;

   for (int l = 0; l < parallel_ref_levels; l++)
   {
      parent_mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      mfem::out << "\033[34mdone." << std::endl;
      mfem::out << "Creating sub-meshes... \033[0m";
   }

   // Create the sub-domains for the cylinder, solid and fluid domains
   AttributeSets &attr_sets = parent_mesh.attribute_sets;
   AttributeSets &bdr_attr_sets = parent_mesh.bdr_attribute_sets;

   Array<int> fluid_domain_attribute = attr_sets.GetAttributeSet("Fluid");
   Array<int> cylinder_domain_attribute = attr_sets.GetAttributeSet("Cylinder");

   Array<int> fluid_cylinder_interface_marker = bdr_attr_sets.GetAttributeSetMarker("Cylinder-Fluid");

   auto fluid_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, fluid_domain_attribute));

   auto cylinder_submesh =
       std::make_shared<ParSubMesh>(ParSubMesh::CreateFromDomain(parent_mesh, cylinder_domain_attribute));

   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Create GridFunctions 
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\033[34m\nCreating FE space and GridFunctions... \033[0m";

   // Create the Finite Element spaces and grid functions
   H1_FECollection fec(order, parent_mesh.Dimension());

   ParFiniteElementSpace fes_cylinder(cylinder_submesh.get(), &fec);
   ParFiniteElementSpace fes_cylinder_grad(cylinder_submesh.get(), &fec, sdim);
   ParFiniteElementSpace fes_fluid(fluid_submesh.get(), &fec);
   ParFiniteElementSpace fes_fluid_grad(fluid_submesh.get(), &fec, sdim);

   ParGridFunction temperature_cylinder_gf(&fes_cylinder); temperature_cylinder_gf = 0.0;
   ParGridFunction grad_cylinder_exact_gf(&fes_cylinder_grad); grad_cylinder_exact_gf = 0.0;
   ParGridFunction temperature_fluid_gf(&fes_fluid);    temperature_fluid_gf = 0.0;
   ParGridFunction temperature_fluid_gf_copy(&fes_fluid); temperature_fluid_gf_copy = 0.0;
   ParGridFunction grad_fluid_gf(&fes_fluid_grad); grad_fluid_gf = 0.0;


   if (Mpi::Root())
      mfem::out << "\033[34mdone." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Setup interface transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////


   // Create the submesh transfer map
   if (Mpi::Root())
      mfem::out << "\033[34mCreating InterfaceTransfer (Native)...\033[0m";
   
   ecm2_utils::InterfaceTransfer cylinder_to_fluid_native(temperature_cylinder_gf, temperature_fluid_gf, fluid_cylinder_interface_marker, ecm2_utils::InterfaceTransfer::Backend::Native);

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   // Setup GSLIB interpolation
   if (Mpi::Root())
      mfem::out << "\033[34mCreating InterfaceTransfer (GSLIB)...\033[0m";


   ecm2_utils::InterfaceTransfer cylinder_to_fluid_gslib(temperature_cylinder_gf, temperature_fluid_gf, fluid_cylinder_interface_marker, ecm2_utils::InterfaceTransfer::Backend::GSLIB);

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;


   // Define function to compute gradient on source mesh (cylinder)
   int vdim = fes_fluid_grad.GetVDim();
   int qoi_size_on_qp = vdim;
   // Vector qoi_src, qoi_dst; // QoI vector, used to store qoi_src in cylinder_grad_func and in call to GSLIB interpolator
   auto cylinder_grad_func = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      temperature_cylinder_gf.GetGradient(Tr, qoi_loc);
   };


   if (Mpi::Root())
      mfem::out << "\033[34mSetting up DataCollection...\033[0m";

   // Setup ouput
   ParaViewDataCollection paraview_dc_cylinder("Cylinder", cylinder_submesh.get());
   ParaViewDataCollection paraview_dc_fluid("Fluid", fluid_submesh.get());
   if (paraview)
   {
      paraview_dc_fluid.SetPrefixPath(outfolder);
      paraview_dc_fluid.SetDataFormat(VTKFormat::BINARY);
      if(order > 1 )
      {
         paraview_dc_fluid.SetHighOrderOutput(true);
         paraview_dc_fluid.SetLevelsOfDetail(order);
      }
      paraview_dc_fluid.RegisterField("Temperature-Submesh", &temperature_fluid_gf);
      paraview_dc_fluid.RegisterField("Temperature-GSLIB", &temperature_fluid_gf_copy);
      paraview_dc_fluid.RegisterField("Gradient", &grad_fluid_gf);

      paraview_dc_cylinder.SetPrefixPath(outfolder);
      paraview_dc_cylinder.SetDataFormat(VTKFormat::BINARY);
      if(order > 1 )
      {
         paraview_dc_cylinder.SetHighOrderOutput(true);
         paraview_dc_cylinder.SetLevelsOfDetail(order);
      }
      paraview_dc_cylinder.RegisterField("Temperature", &temperature_cylinder_gf);
      paraview_dc_cylinder.RegisterField("Gradient-exact", &grad_cylinder_exact_gf);
   }

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Define Temperature analytic field
   // f(x,y,z) = (x-2.5)^2 + (y-2.5)^2
   auto func = [](const Vector &x)
   {
       const int dim = x.Size();
       double res = 0.0;
       res = std::pow(x(0) - 2.5, 2) + std::pow(x(1) - 2.5, 2);
       return res;
   };

   // Define Gradient of Temperature analytic field
   // ∇f(x,y,z) = [2*(x-2.5), 2*(y-2.5), 0]
   auto func_grad = [](const Vector &x, Vector &p)
   {
       const int dim = x.Size();
       p.SetSize(dim);
       p(0) = 2.0 * (x(0) - 2.5);
       p(1) = 2.0 * (x(1) - 2.5);
       p(2) = 0.0;
   };

   if (Mpi::Root())
      mfem::out << "\033[34m\nTransfer solution from cylinder --> fluid domain... \033[0m" << std::endl;

   // Initial conditions on cylinder
   FunctionCoefficient Tfunc(func);
   //ConstantCoefficient Tcyl(Tcylinder);
   temperature_cylinder_gf.ProjectCoefficient(Tfunc);

   ConstantCoefficient Tfluid(Tfluid);
   temperature_fluid_gf.ProjectCoefficient(Tfluid);

   VectorFunctionCoefficient grad_Tfunc(vdim, func_grad);
   grad_cylinder_exact_gf.ProjectCoefficient(grad_Tfunc);

   // Export before transfer
   if ( paraview )
   {
      paraview_dc_cylinder.SetCycle(0); paraview_dc_cylinder.SetTime(0.0); paraview_dc_cylinder.Save();
      paraview_dc_fluid.SetCycle(0); paraview_dc_fluid.SetTime(0.0); paraview_dc_fluid.Save();
   }

   StopWatch chrono;
   real_t t_native, t_gslib, t_gslib_grad;

   // Transfer with SubMesh Transfer Map
   if (Mpi::Root())
      mfem::out << "Using SubMesh Transfer Map...";

   chrono.Start();
   cylinder_to_fluid_native.Interpolate(temperature_cylinder_gf, temperature_fluid_gf);
   chrono.Stop();
   t_native = chrono.RealTime();

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   // Transfer with GSLIB
   if (Mpi::Root())
      mfem::out << "Using GSLIB...";

   chrono.Clear();
   chrono.Start();
   cylinder_to_fluid_gslib.Interpolate(temperature_cylinder_gf, temperature_fluid_gf_copy);
   chrono.Stop();
   t_gslib = chrono.RealTime();

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   // Transfer gradient with GSLIB
   if (Mpi::Root())
      mfem::out << "Using GSLIB for gradient...";

   chrono.Clear();
   chrono.Start();
   cylinder_to_fluid_gslib.InterpolateQoI(cylinder_grad_func, grad_fluid_gf);
   chrono.Stop();
   t_gslib_grad = chrono.RealTime();

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   // Export after transfer
   if (paraview)
   {
      paraview_dc_cylinder.SetCycle(1); paraview_dc_cylinder.SetTime(0.1); paraview_dc_cylinder.Save();
      paraview_dc_fluid.SetCycle(1); paraview_dc_fluid.SetTime(0.1); paraview_dc_fluid.Save();
   }

   if (Mpi::Root())
      mfem::out << "\033[34mdone.\033[0m" << std::endl;

   if (Mpi::Root())
   {
      mfem::out << "Transfer time (Native): " << t_native << " s" << std::endl;
      mfem::out << "Transfer time (GSLIB): " << t_gslib << " s" << std::endl;
      mfem::out << "Transfer time (GSLIB Gradient): " << t_gslib_grad << " s" << std::endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   return 0;
}

void ExportMeshwithPartitioning(const std::string &outfolder, Mesh &mesh, const int *partitioning_)
{
   // Extract the partitioning
   Array<int> partitioning;
   partitioning.MakeRef(const_cast<int *>(partitioning_), mesh.GetNE(), false);

   // Assign partitioning to the mesh
   FiniteElementCollection *attr_fec = new L2_FECollection(0, mesh.Dimension());
   FiniteElementSpace *attr_fespace = new FiniteElementSpace(&mesh, attr_fec);
   GridFunction attr(attr_fespace);
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      attr(i) = partitioning[i] + 1;
   }

   // Create paraview datacollection
   ParaViewDataCollection paraview_dc("Partitioning", &mesh);
   paraview_dc.SetPrefixPath(outfolder);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetCompressionLevel(9);
   paraview_dc.RegisterField("partitioning", &attr);
   paraview_dc.Save();

   delete attr_fespace;
   delete attr_fec;
}