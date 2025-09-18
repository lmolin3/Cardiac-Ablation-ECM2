#include "interface_transfer.hpp"

namespace mfem
{

    namespace ecm2_utils
    {

        void ReorderByVDIM(const Vector &v_in, Vector &v_out, int vdim)
        {
            int npts = v_in.Size() / vdim;
            v_out.SetSize(vdim * npts);

            auto d_v_in = v_in.Read();
            auto d_v_out = v_out.Write();

            mfem::forall(npts, [=] MFEM_HOST_DEVICE(int i)
                         {
                for (int d = 0; d < vdim; d++)
                {
                    d_v_out[vdim * i + d] = d_v_in[d * npts + i];
                }
            });
        }


        /// INTERFACE TRANSFER CLASS
        InterfaceTransfer::InterfaceTransfer(ParFiniteElementSpace *src_fes_, ParFiniteElementSpace *dst_fes_,
                                             Array<int> &bdr_attributes_, Backend backend_, MPI_Comm comm_, bool use_undeformed_coords_)
            : src_fes(src_fes_), dst_fes(dst_fes_), bdr_attributes(bdr_attributes_), backend(backend_),
              src_mesh(src_fes_->GetParMesh()), dst_mesh(dst_fes_->GetParMesh()), sdim(src_mesh->SpaceDimension()), comm(comm_),
              use_undeformed_coords(use_undeformed_coords_)
        {
#ifdef MFEM_USE_MPI
            MPI_Comm_rank(comm, &myid);
            MPI_Comm_size(comm, &nranks);
            recvcounts_gather.SetSize(nranks);
            recvcounts_scatter.SetSize(nranks);
            recvcounts.SetSize(nranks);
            displs_gather.SetSize(nranks);
            displs_scatter.SetSize(nranks);
            finder = new FindPointsGSLIB(comm);
#else
            finder = new FindPointsGSLIB();
#endif
            /// Store indices of boundary elements on both source and destination meshes
            // Check if the provided bdr_attributes are compatible with the destination mesh
            MFEM_ASSERT(bdr_attributes.Size() >= dst_mesh->bdr_attributes.Max(),
                        "InterfaceTransfer: bdr_attributes size must match the number of boundary elements in the destination mesh.");

            /// 1. Find boundary elements with the specified attributes (on the destination mesh) --> LOCAL
            for (int be = 0; be < dst_mesh->GetNBE(); be++)
            {
                const int bdr_el_attr = dst_mesh->GetBdrAttribute(be);
                if (bdr_attributes[bdr_el_attr - 1] == 0)
                {
                    continue;
                }
                dst_bdr_element_idx.Append(be);
            }

            // For convenience, we will also store the boundary elements for the source mesh (to test the alternative Interpolation)
            for (int be = 0; be <  src_mesh->GetNBE(); be++)
            {
                const int bdr_el_attr = src_mesh->GetBdrAttribute(be);
                if (bdr_attributes[bdr_el_attr - 1] == 0)
                {
                    continue;
                }
                src_bdr_element_idx.Append(be);
            }

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
                // Create a ParTransferMap object
                transfer_map = new ParTransferMap(src_fes, dst_fes);
            }

            if (backend == Backend::GSLIB || backend == Backend::Hybrid)
            {
                // Setup specific for GSLIB functionalities (determine QP coordinates on interface, setup FindPointsGSLIB instance)
                SetupGSLIB();
                undeformed_bdr_element_coords = bdr_element_coords;
                gslib_initialized = true; // Mark that the initial setup has been done
            }

            tmp_src_gf.SetSpace(src_fes); tmp_src_gf = 0.0;
        }

        InterfaceTransfer::~InterfaceTransfer()
        {
            if (transfer_map)
            {
                delete transfer_map;
            }

            if (backend == Backend::GSLIB || backend == Backend::Hybrid)
            {
                finder->FreeData();
                delete finder;
            }

            // Nothing else to clean, InterfaceTransfer doesn't take ownership of the mesh or FiniteElementSpace or Mesh
        }

        void InterfaceTransfer::Update( VectorCoefficient *dx )
        {
            if (use_undeformed_coords)
            {
                // If we are using undeformed coordinates, we don't need to update anything
                return;
            }

            
            if (backend == Backend::GSLIB || backend == Backend::Hybrid)
            {
                // Setup GSLIB again
                finder->FreeData();
                delete finder;
                finder = nullptr;
#ifdef MFEM_USE_MPI
                finder = new FindPointsGSLIB(comm);
#else
                finder = new FindPointsGSLIB();
#endif
                SetupGSLIB(dx);
            }

            if (backend == Backend::Native || backend == Backend::Hybrid)
            {
                // Update Native backend: recreate the ParTransferMap
                if (transfer_map)
                {
                    delete transfer_map;
                    transfer_map = nullptr;
                }

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
                // Create a ParTransferMap object
                transfer_map = new ParTransferMap(src_fes, dst_fes);
            }
        }

        void InterfaceTransfer::GSLIBUseUndeformedCoords(bool use_undeformed_coords_)
        {
            use_undeformed_coords = use_undeformed_coords_;

            Update(); // Reinitialize the finder with the new undeformed coordinates
        }

        void InterfaceTransfer::Interpolate(ParGridFunction &src_gf, ParGridFunction &dst_gf)
        {
            if (backend == Backend::Native || backend == Backend::Hybrid)
            {
                transfer_map->Transfer(src_gf, dst_gf);
            }
            else // Backend::GSLIB
            {
                // Interpolate the source grid function and store in the vector interp_vals
                Vector tmp_interp_vals;
                finder->Interpolate(src_gf, tmp_interp_vals);

                auto vec_ordering = src_gf.FESpace()->GetOrdering();
#ifdef MFEM_USE_MPI
                // Before scattering, ensure that for Vector GridFunctions (vdim > 1) with byNODES ordering,
                // we reorder the data byVDIM
                // @note: If src_gf is a Vector GridFunction (vdim > 1) with byNODES ordering,
                // then in parallel, interp_vals will be structured as:
                // [ x1 x2 ... xn | x1 x2 ... xn | ... | y1 y2 ... yn | ... | z1 z2 ... zn ]
                // where each block corresponds to a rank. This global byNODES ordering
                // means MPIScatterv will not distribute tmp_interp_vals correctly to each rank.
                int vdim = src_gf.FESpace()->GetVDim();
                if (vec_ordering == Ordering::byNODES && vdim > 1)
                {
                    ReorderByVDIM(tmp_interp_vals, interp_vals, vdim);
                    vec_ordering = Ordering::byVDIM; // Update the ordering to reflect that vector is now byVDIM
                }
                else
                {
                    // If the ordering is already byVDIM or if vdim == 1, we can use dst_vec directly
                    interp_vals.MakeRef(tmp_interp_vals, 0, tmp_interp_vals.Size());
                }
#else
                // If we are not using MPI, we can directly use the tmp_interp_vals
                interp_vals.MakeRef(tmp_interp_vals, 0, tmp_interp_vals.Size());
#endif // MFEM_USE_MPI

                // Transfer the interpolated values to the destination grid function
                TransferQoIToDestGf(interp_vals, dst_gf, vec_ordering);
            }
        }

        void InterfaceTransfer::InterpolateQoI(qoi_func_t qoi_func, ParGridFunction &dst_gf, bool bdr_version)
        {
            if (bdr_version)
                InterpolateQoIv2(qoi_func, dst_gf);
            else
                InterpolateQoIv1(qoi_func, dst_gf);
        }

        void InterfaceTransfer::InterpolateQoIv1(qoi_func_t qoi_func, ParGridFunction &dst_gf)
        {
            MFEM_ASSERT(backend == Backend::GSLIB || backend == Backend::Hybrid,
                        "InterpolateQoIv1 is not supported for the Native backend, and requires GSLIB.");

            // Extract space dimension and FE space from the mesh
            ParFiniteElementSpace *fes_dst = dst_gf.ParFESpace();
            int vdim = fes_dst->GetVDim();

            // Distribute internal GSLIB info to the corresponding mpi-rank for each point.
            Array<unsigned int>
                recv_elem,
                recv_code;   // Element and GSLIB code
            Vector recv_rst; // (Reference) coordinates of the quadrature points
            finder->DistributePointInfoToOwningMPIRanks(recv_elem, recv_rst, recv_code);
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
            finder->DistributeInterpolatedValues(qoi_src, vdim, Ordering::byVDIM, qoi_dst);

            // Transfer the QoI to the destination grid function
            TransferQoIToDestGf(qoi_dst, dst_gf, Ordering::byVDIM);
        }


        void InterfaceTransfer::InterpolateQoIv2(qoi_func_t qoi_bdr_func, ParGridFunction &dst_gf)
        {
            // Check that vdim of dst_gf matches the expected dimension
            MFEM_ASSERT(dst_gf.ParFESpace()->GetVDim() == src_fes->GetVDim(),
                        "InterfaceTransfer::InterpolateQoIv2: Destination grid function must have the same VDim as the source finite element space.");

            int nbe = src_bdr_element_idx.Size();
            int vdim = src_fes->GetVDim();

            qoi_loc.SetSize(vdim);
            auto ordering = src_fes->GetOrdering();
            tmp_src_gf = 0.0; // Reset the temporary source grid function

            // ----- 1. Interpolate QoI over the boundary elements (of selected interface) of the source mesh -----
            Array<int> vdofs;
            for (int i = 0; i < nbe; i++)
            {
                // Get the boundary element, boundary element transformation, and vdofs
                int be_idx = src_bdr_element_idx[i];
                const FiniteElement *be = src_fes->GetBE(be_idx);
                ElementTransformation *Tr = src_fes->GetBdrElementTransformation(be_idx);
                const IntegrationRule &ir_face = be->GetNodes();

                const int ndofs = ir_face.GetNPoints();
                src_fes->GetBdrElementVDofs(be_idx, vdofs);
                tmp_src_gf.GetSubVector(vdofs, qoi_src);

                for (int qp = 0; qp < ndofs; qp++)
                {
                    // Get the quadrature point
                    const IntegrationPoint &ip = ir_face.IntPoint(qp);
                    Tr->SetIntPoint(&ip);

                    // Evaluate the QoI at the quadrature point
                    qoi_bdr_func(*Tr, ip, qoi_loc);

                    // Set the local QoI vector (vdim) into the vdofs qoi_src vector, ordered either by NODES or by VDIM
                    int idx;
                    for (int d = 0; d < vdim; d++)
                    {
                        idx = (ordering == Ordering::byVDIM) ? qp * vdim + d : d * ndofs + qp;
                        qoi_src(idx) = qoi_loc(d);
                    }
                }

                // Set the grid function values at the boundary element
                tmp_src_gf.SetSubVector(vdofs, qoi_src);
            }

            tmp_src_gf.SetTrueVector();      
            tmp_src_gf.SetFromTrueVector();

            // ----- 2. Transfer the QoI to the destination grid function -----
            this->Interpolate(tmp_src_gf, dst_gf);
        }


        inline void InterfaceTransfer::TransferQoIToDestGf(const Vector &dst_vec, ParGridFunction &dst_gf, const mfem::Ordering::Type src_ordering)
        {
            // @note: ordering of dst_vec will be:
            //       * byVDIM/byNODES, depending on field_in (if using Interpolate)
            //       * byVDIM (if using InterpolateQoI)
            //       either way, it's not necessarily the same as dst_gf ordering

            // Extract fe space from the destination grid function
            ParFiniteElementSpace *dst_gf_fes = dst_gf.ParFESpace();
            const int dof = dst_gf_fes->GetTypicalTraceElement()->GetNodes().GetNPoints();
            int vdim = dst_gf_fes->GetVDim();

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
                local_dst_vec.SetSize(local_size);

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
            int nbe = dst_bdr_element_idx.Size();

            // NOTE: could be further optimized by considering combined cases for ordering of vector and gf
            if (src_ordering == Ordering::byVDIM) // dst_vec is ordered byVDIM
            {
                for (int be = 0; be < nbe; be++)
                {
                    be_idx = dst_bdr_element_idx[be];
                    dst_gf_fes->GetBdrElementVDofs(be_idx, vdofs);

                    offset1 = vdim * dof * be;

                    for (int qp = 0; qp < dof; qp++)
                    {
                        offset2 = offset1 + vdim * qp;
                        for (int d = 0; d < vdim; d++)
                        {
                            idx_in = offset2 + d;
                            idx_out = dof * d + qp;  // vdofs is returned byNODES regardless of the dst_ordering, so we populate loc_values accordingly
                            loc_values(idx_out) = local_dst_vec(idx_in);
                        }
                    }

                    dst_gf.SetSubVector(vdofs, loc_values);
                }
            }
            else // src_ordering == Ordering::byNODES (i.e. dst_vec is ordered byNODES)
            {
                for (int be = 0; be < nbe; be++)
                {
                    be_idx = dst_bdr_element_idx[be];
                    dst_gf_fes->GetBdrElementVDofs(be_idx, vdofs);

                    offset1 = nbe * dof;

                    for (int d = 0; d < vdim; d++)
                    {
                        offset2 = offset1 * d + be * dof;
                        for (int qp = 0; qp < dof; qp++)
                        {
                            idx_in = offset2 + qp;
                            idx_out = dof * d + qp;  // vdofs is returned byNODES regardless of the dst_ordering, so we populate loc_values accordingly
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

            auto &finder_elem = finder->GetElem();
            GSLIBAttrToMarker(src_mesh->GetNE(), finder_elem, elem_idx);
        }

        void InterfaceTransfer::SetupGSLIB(  VectorCoefficient *dx )
        {
            // Check if VectorCoefficient dx has right dimension
            if (dx)
            {
                MFEM_ASSERT(dx->GetVDim() == sdim,
                            "InterfaceTransfer::SetupGSLIB: dx must have the same dimension as the mesh space dimension.");
            }

            if (!gslib_initialized || !use_undeformed_coords)
            {
                /// Compute coordinates of quadrature points on dst_bdr_element_idx (on the destination mesh) --> LOCAL
                const IntegrationRule &ir_face = (dst_fes->GetTypicalTraceElement())->GetNodes();
                int bdr_coords_size = dst_bdr_element_idx.Size() * ir_face.GetNPoints() * sdim;

                Vector bdr_element_coords_loc;
                bdr_element_coords_loc.SetSize(bdr_coords_size);
                bdr_element_coords_loc = 0.0;
                x.SetSize(sdim);
                x = 0.0;
                dx_val.SetSize(sdim);
                dx_val = 0.0;

                bool compute_offset = (dx != nullptr) && !use_undeformed_coords;

                auto pec = Reshape(bdr_element_coords_loc.ReadWrite(), sdim, ir_face.GetNPoints(), dst_bdr_element_idx.Size());
                for (int be = 0; be < dst_bdr_element_idx.Size(); be++)
                {
                    int be_idx = dst_bdr_element_idx[be];
                    const FiniteElement *fe = dst_fes->GetBE(be_idx);
                    ElementTransformation *Tr = dst_fes->GetBdrElementTransformation(be_idx);
                    const IntegrationRule &ir_face = fe->GetNodes();

                    for (int qp = 0; qp < ir_face.GetNPoints(); qp++)
                    {
                        const IntegrationPoint &ip = ir_face.IntPoint(qp);
                        Tr->SetIntPoint(&ip);

                        dx_val = 0.0;
                        if (compute_offset)
                        {
                            // If dx is provided, compute the offset
                            dx->Eval(dx_val, *Tr, ip);
                        }

                        Tr->Transform(ip, x);
                        for (int d = 0; d < sdim; d++)
                        {
                            pec(d, qp, be) = x(d) + dx_val(d);
                        }
                    }
                }

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
            }

            // Set target bdry coordinates
            Vector &target_coords = (use_undeformed_coords && gslib_initialized) ? undeformed_bdr_element_coords : bdr_element_coords;

            // Default parameters
            const real_t rel_bbox_el = 0.1;
            const real_t newton_tol = 1.0e-12;
            const int npts_at_once = 256;
            finder->Setup(*src_mesh, rel_bbox_el, newton_tol, npts_at_once);
            finder->SetDistanceToleranceForPointsFoundOnBoundary(boundary_tol);
            finder->FindPoints(target_coords, Ordering::byVDIM); // --> Because the coordinates are stored byVDIM
            auto procs = finder->GetProc();
            auto finder_codes = finder->GetCode();
            int idx = 0;
            for (auto &code : finder_codes)
            {
                if (code == 2)
                {
                    mfem::out << "Point NOT found in the mesh, on rank " << myid << " for " << name << std::endl;
                    mfem::out << "Point coordinates: " << target_coords(idx*sdim) << ", " << target_coords(idx*sdim + 1);
                    if (sdim == 3)
                    {
                        mfem::out << ", " << target_coords(idx*sdim + 2);
                    }
                    mfem::out << ")." << std::endl;
                    mfem::out << "Point undeformed coordinates: " << undeformed_bdr_element_coords(idx*sdim) << ", " << undeformed_bdr_element_coords(idx*sdim + 1);
                    if (sdim == 3)
                    {
                        mfem::out << ", " << undeformed_bdr_element_coords(idx*sdim + 2);
                    }
                    mfem::out << ")." << std::endl;
                    MFEM_ABORT("Aborting!");
                }
                idx ++; // Move to the next point
            }
        }

    } // namespace ecm2_utils

} // namespace mfem
