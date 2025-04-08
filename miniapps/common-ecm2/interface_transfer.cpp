#include "interface_transfer.hpp"

namespace mfem
{

    namespace ecm2_utils
    {

        InterfaceTransfer::InterfaceTransfer(ParGridFunction &src_gf, ParGridFunction &dst_gf, Array<int> &bdr_attributes_, Backend backend_, MPI_Comm comm_)
            : src_fes(src_gf.ParFESpace()), dst_fes(dst_gf.ParFESpace()), bdr_attributes(bdr_attributes_), backend(backend_),
              src_mesh(src_gf.ParFESpace()->GetParMesh()), dst_mesh(dst_gf.ParFESpace()->GetParMesh()), sdim(src_mesh->SpaceDimension()), comm(comm_),
              finder(comm_)
        {
#ifdef MFEM_USE_MPI
            MPI_Comm_rank(comm, &myid);
            MPI_Comm_size(comm, &nranks);
            recvcounts_gather.SetSize(nranks);
            recvcounts_scatter.SetSize(nranks);
            recvcounts.SetSize(nranks);
            displs_gather.SetSize(nranks);
            displs_scatter.SetSize(nranks);
#endif
            /// Setup finder (depending on the backend)

            if (backend == Backend::Native || backend == Backend::Hybrid)
            {
                // Check that the source and destination meshes are (Par)Submeshes by dynamic_cast
#ifdef MFEM_USE_MPI
                ParSubMesh *src_submesh = dynamic_cast<ParSubMesh *>(src_mesh);
                ParSubMesh *dst_submesh = dynamic_cast<ParSubMesh *>(dst_mesh);
                MFEM_ASSERT(src_submesh != nullptr && dst_submesh != nullptr,
                            "InterfaceTransfer: Source and destination meshes must be ParSubMeshes.");
#else
                SubMesh *src_submesh = dynamic_cast<SubMesh *>(src_mesh);
                SubMesh *dst_submesh = dynamic_cast<SubMesh *>(dst_mesh);
                MFEM_ASSERT(src_submesh != nullptr && dst_submesh != nullptr,
                            "InterfaceTransfer: Source and destination meshes must be SubMeshes.");
#endif
                // Crete a ParTransferMap object
                transfer_map = new ParTransferMap(src_gf, dst_gf);
            }

            if (backend == Backend::GSLIB || backend == Backend::Hybrid)
            {
                // Setup specific for GSLIB functionalities (determine QP coordinates on interface, setup FindPointsGSLIB instance)
                SetupGSLIB();
            }
        }

        InterfaceTransfer::~InterfaceTransfer()
        {
            if (transfer_map)
            {
                delete transfer_map;
            }

            if (backend == Backend::GSLIB || backend == Backend::Hybrid)
            {
                finder.FreeData();
            }

            // Nothing else to clean, InterfaceTransfer doesn't take ownership of the mesh or FiniteElementSpace or Mesh
        }

        void InterfaceTransfer::Interpolate(ParGridFunction &src_gf, ParGridFunction &dst_gf)
        {
            if (backend == Backend::Native || backend == Backend::Hybrid)
            {
                transfer_map->Transfer(src_gf, dst_gf);
            }
            else // Backend::GSLIB
            {
                finder.Interpolate(src_gf, interp_vals);  
                TransferQoIToDestGf(interp_vals, dst_gf, src_gf.FESpace()->GetOrdering());
            }
        }

        void InterfaceTransfer::InterpolateQoI(qoi_func_t qoi_func, ParGridFunction &dst_gf)
        {
            MFEM_ASSERT(backend == Backend::GSLIB || backend == Backend::Hybrid,
                        "InterpolateQoI is not supported for the Native backend, and requires GSLIB.");

            // Extract space dimension and FE space from the mesh
            ParFiniteElementSpace *fes_dst = dst_gf.ParFESpace();
            int vdim = fes_dst->GetVDim();

            // Distribute internal GSLIB info to the corresponding mpi-rank for each point.
            Array<unsigned int>
                recv_elem,
                recv_code;   // Element and GSLIB code
            Vector recv_rst; // (Reference) coordinates of the quadrature points
            finder.DistributePointInfoToOwningMPIRanks(recv_elem, recv_rst, recv_code);
            int npt_recv = recv_elem.Size();

            // Compute qoi locally (on source mesh)
            qoi_src.SetSize(npt_recv * vdim);
            qoi_loc.SetSize(vdim);
            for (int i = 0; i < npt_recv; i++)
            {
                // Get the element index
                const int e = recv_elem[i];

                // Get the quadrature point
                IntegrationPoint ip;
                if (sdim == 2)
                    ip.Set2(recv_rst(sdim * i + 0), recv_rst(sdim * i + 1));
                else if (sdim == 3)
                    ip.Set3(recv_rst(sdim * i + 0), recv_rst(sdim * i + 1), recv_rst(sdim * i + 2));

                // Get the element transformation
                ElementTransformation *Tr = src_fes->GetElementTransformation(e);
                Tr->SetIntPoint(&ip);

                // Extract the local qoi vector
                qoi_loc.MakeRef(qoi_src, i * vdim, vdim);

                // Compute the qoi_src at quadrature point (it will change the qoi_src vector)
                qoi_func(*Tr, ip, qoi_loc);
            }

            // Transfer the QoI from the source mesh to the destination mesh at quadrature points
            finder.DistributeInterpolatedValues(qoi_src, vdim, Ordering::byVDIM, qoi_dst);

            // Transfer the QoI to the destination grid function
            TransferQoIToDestGf(qoi_dst, dst_gf, Ordering::byVDIM);
        }

        inline void InterfaceTransfer::TransferQoIToDestGf(const Vector &dst_vec, ParGridFunction &dst_gf, const int ordering)
        {            
            // NOTE: ordering of dst_vec qill be:
            //       * byVDIM/byNODES, depending on field_in (if using Interpolate)
            //       * byVDIM (if using InterpolateQoI)
            //       either way, it's not necessarily the same as dst_gf ordering

            // Extract fe space from the destination grid function
            ParFiniteElementSpace *dst_fes = dst_gf.ParFESpace();
            auto dst_ordering = dst_fes->GetOrdering();
            const int dof = dst_fes->GetTypicalBE()->GetNodes().GetNPoints();
            int vdim = dst_fes->GetVDim();

            // In parallel we need to scatter the interpolated data so that each rank only has the right portion of the data
#ifdef MFEM_USE_MPI
            // Compute recvcounts and displacements
            for (int i = 0; i < nranks; i++)
            {
                recvcounts_scatter[i] = vdim * recvcounts[i];
            }

            displs_scatter[0] = 0; // The first rank starts at index 0
            for (int i = 1; i < nranks; i++)
            {
                displs_scatter[i] = displs_scatter[i - 1] + recvcounts_scatter[i - 1];
            }

            // Scatter the global dst_vec to get the local part for this rank
            int local_size = recvcounts_scatter[myid]; // Local size for this rank
            if (local_dst_vec.Size() != local_size)
            {
                local_dst_vec.SetSize(local_size);
            }          

            MPI_Scatterv(dst_vec.GetData(), recvcounts_scatter.GetData(), displs_scatter.GetData(), MFEM_MPI_REAL_T,
                         local_dst_vec.GetData(), local_size, MFEM_MPI_REAL_T, 0, comm);
#else
            local_dst_vec = dst_vec; // In serial, the local vector is the same as the global vector
#endif

            // Transfer the QoI to the destination grid function

            int idx_in, idx_out, be_idx;
            int offset1, offset2;
            Vector loc_values(dof * vdim);
            Array<int> vdofs(dof * vdim);
            int nbe = bdr_element_idx.Size();

            // NOTE: could be further optimized by considering combined cases for ordering of vector and gf
            if (ordering == Ordering::byVDIM)
            {
                for (int be = 0; be < nbe; be++)
                {
                    be_idx = bdr_element_idx[be];
                    dst_fes->GetBdrElementVDofs(be_idx, vdofs);

                    offset1 = vdim * dof * be;

                    for (int qp = 0; qp < dof; qp++)
                    {
                        offset2 = offset1 + vdim * qp;
                        for (int d = 0; d < vdim; d++)
                        {
                            idx_in = offset2 + d;
                            idx_out = dst_ordering == Ordering::byVDIM ? qp * vdim + d : dof * d + qp;
                            loc_values(idx_out) = local_dst_vec(idx_in);
                        }
                    }

                    dst_gf.SetSubVector(vdofs, loc_values);
                }
            }
            else // ordering == Ordering::byNODES
            {
                for (int be = 0; be < nbe; be++)
                {
                    be_idx = bdr_element_idx[be];
                    dst_fes->GetBdrElementVDofs(be_idx, vdofs);

                    offset1 = nbe * dof;

                    for (int d = 0; d < vdim; d++)
                    {
                        offset2 = offset1 * d + be * dof;
                        for (int qp = 0; qp < dof; qp++)
                        {
                            idx_in = offset2 + qp;
                            idx_out = dst_ordering == Ordering::byVDIM ? qp * vdim + d : dof * d + qp;
                            loc_values(idx_out) = local_dst_vec(idx_in);
                        }
                    }

                    dst_gf.SetSubVector(vdofs, loc_values);
                }
            }
        }

        void InterfaceTransfer::GSLIBAttrToMarker(int max_attr, const Array<unsigned int> &elems, Array<int> &marker)
        {
            // Convert the element indices to markers
            marker.SetSize(max_attr);
            marker = 0;
            for (const auto &elem : elems)
            {
                if (elem <= 0 || elem > static_cast<unsigned int>(max_attr))
                    continue;
                int attr = static_cast<int>(elem);
                marker[attr - 1] = 1;
            }

            return;
        }

        void InterfaceTransfer::GetElementIdx(Array<int> &elem_idx)
        {
            MFEM_ASSERT(backend == Backend::GSLIB || backend == Backend::Hybrid,
                        "GetElementIdx is not supported for the Native backend, and requires GSLIB.");

            auto &finder_elem = finder.GetElem();
            GSLIBAttrToMarker(src_mesh->GetNE(), finder_elem, elem_idx);
        }

        void InterfaceTransfer::SetupGSLIB()
        {
            /// 1. Find boundary elements with the specified attributes (on the destination mesh) --> LOCAL
            for (int be = 0; be < dst_mesh->GetNBE(); be++)
            {
                const int bdr_el_attr = dst_mesh->GetBdrAttribute(be);
                if (bdr_attributes[bdr_el_attr - 1] == 0)
                {
                    continue;
                }
                bdr_element_idx.Append(be);
            }

            /// 2. Compute coordinates of quadrature points on bdr_element_idx (on the destination mesh) --> LOCAL
            const IntegrationRule &ir_face = (dst_fes->GetTypicalBE())->GetNodes();
            int bdr_coords_size = bdr_element_idx.Size() * ir_face.GetNPoints() * sdim;

            Vector bdr_element_coords_loc;
            bdr_element_coords_loc.SetSize(bdr_coords_size);
            bdr_element_coords_loc = 0.0;

            auto pec = Reshape(bdr_element_coords_loc.ReadWrite(), sdim, ir_face.GetNPoints(), bdr_element_idx.Size());
            for (int be = 0; be < bdr_element_idx.Size(); be++)
            {
                int be_idx = bdr_element_idx[be];
                const FiniteElement *fe = dst_fes->GetBE(be_idx);
                ElementTransformation *Tr = dst_fes->GetBdrElementTransformation(be_idx);
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


  //  NOTE: This is not working for now, keeping it commented out to make the multidomain examples work 
        //Simulations for dirichlet-neumann multidomain example works without
        //However, we must take into account mpi communication of the coordinates
        //The GSLIB finder will take care of identifying on which rank the points are located after finding them

            // In parallel we need to gather the coordinates from all ranks so the GSLIB finder can search all points
            // The FindPoints will take care of identifying on which rank the points are located
#ifdef MFEM_USE_MPI
            // First reduce the size of the vector
            int global_size = 0;
            MPI_Allreduce(&bdr_coords_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
            bdr_element_coords.SetSize(global_size);

            // Gather the sizes from all ranks
            MPI_Allgather(&bdr_coords_size, 1, MPI_INT, recvcounts_gather.GetData(), 1, MPI_INT, comm);

            for (int i = 0; i < nranks; i++)
            {
                recvcounts[i] = recvcounts_gather[i] / sdim; // Convert from elements * sdim to elements
            }

            // Compute displacements
            displs_gather[0] = 0;
            for (int i = 1; i < nranks; i++)
            {
                displs_gather[i] = displs_gather[i - 1] + recvcounts_gather[i - 1];
            }

            // Now gather the coordinates from all ranks
            MPI_Allgatherv(bdr_element_coords_loc.GetData(), bdr_coords_size, MFEM_MPI_REAL_T,
                           bdr_element_coords.GetData(), recvcounts_gather.GetData(), displs_gather.GetData(), MFEM_MPI_REAL_T, comm);
#else
            bdr_element_coords = bdr_element_coords_loc;
#endif 

            // Default parameters
            const real_t rel_bbox_el = 0.1;
            const real_t newton_tol = 1.0e-12;
            const int npts_at_once = 256;
            finder.Setup(*src_mesh, rel_bbox_el, newton_tol, npts_at_once);
            finder.FindPoints(bdr_element_coords, Ordering::byVDIM); // --> Because the coordinates are stored byVDIM
            auto procs = finder.GetProc();
            auto finder_codes = finder.GetCode();
            for (auto &code : finder_codes)
            {
                if (code != 1)
                {
                    MFEM_ABORT("Point is not found on the boundary");
                }
            }
        }

    } // namespace ecm2_utils

} // namespace mfem
