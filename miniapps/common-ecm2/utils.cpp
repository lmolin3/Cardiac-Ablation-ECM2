#include "utils.hpp"

namespace mfem
{

    namespace ecm2_utils
    {
        void print_matrix(const DenseMatrix &A)
        {
            std::cout << std::scientific;
            std::cout << "{";
            for (int i = 0; i < A.NumRows(); i++)
            {
                std::cout << "{";
                for (int j = 0; j < A.NumCols(); j++)
                {
                    std::cout << A(i, j);
                    if (j < A.NumCols() - 1)
                    {
                        std::cout << ", ";
                    }
                }
                if (i < A.NumRows() - 1)
                {
                    std::cout << "}, ";
                }
                else
                {
                    std::cout << "}";
                }
            }
            std::cout << "}\n";
            std::cout << std::fixed;
            std::cout << std::endl
                      << std::flush; // Debugging print
        }

        // Mult and AddMult for full matrix (using matrices modified with WliminateRowsCols)
        void FullMult(HypreParMatrix *mat, HypreParMatrix *mat_e, Vector &x, Vector &y)
        {
            mat->Mult(x, y);      // y =  mat x
            mat_e->AddMult(x, y); // y += mat_e x
        }

        void FullAddMult(HypreParMatrix *mat, HypreParMatrix *mat_e, Vector &x, Vector &y, double a)
        {
            mat->AddMult(x, y, a);   // y +=  a mat x
            mat_e->AddMult(x, y, a); // y += a mat_e x
        }

        void MeanZero(ParGridFunction &gf, ParLinearForm *mass_lf, real_t volume)
        {
            // Make sure not to recompute the inner product linear form if already provided
            if (mass_lf == nullptr)
            {
                ConstantCoefficient onecoeff(1.0);
                mass_lf = new ParLinearForm(gf.ParFESpace());
                auto *dlfi = new DomainLFIntegrator(onecoeff);
                mass_lf->AddDomainIntegrator(dlfi);
                mass_lf->Assemble();

                ParGridFunction one_gf(gf.ParFESpace());
                one_gf.ProjectCoefficient(onecoeff);

                volume = mass_lf->operator()(one_gf);
            }

            if (volume == 0.0)
            {
                ConstantCoefficient onecoeff(1.0);
                ParGridFunction one_gf(gf.ParFESpace());
                one_gf.ProjectCoefficient(onecoeff);
                volume = mass_lf->operator()(one_gf);
            }

            real_t integ = mass_lf->operator()(gf);
            gf -= integ / volume;
        }

        void Orthogonalize(Vector &v, const MPI_Comm &comm)
        {
            real_t loc_sum = v.Sum();
            real_t global_sum = 0.0;
            int loc_size = v.Size();
            int global_size = 0;

            MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, comm);

            v -= global_sum / static_cast<real_t>(global_size);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                              Mesh utils                                              ///
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                     InterfaceTransfer utils                                         ///
        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Implement logic && operator for Array<int>
        Array<int> operator&&(const Array<int> &a, const Array<int> &b)
        {
            MFEM_ASSERT(a.Size() == b.Size(), "Arrays must have the same size.");
            Array<int> result(a.Size());
            for (int i = 0; i < a.Size(); i++)
            {
                result[i] = a[i] && b[i];
            }
            return result;
        }

        InterfaceTransfer::InterfaceTransfer(ParGridFunction &src_gf, ParGridFunction &dst_gf, Array<int> &bdr_attributes_, Backend backend_)
            : src_fes(src_gf.ParFESpace()), dst_fes(dst_gf.ParFESpace()), bdr_attributes(bdr_attributes_), backend(backend_),
              src_mesh(src_gf.ParFESpace()->GetParMesh()), dst_mesh(dst_gf.ParFESpace()->GetParMesh()), sdim(src_mesh->SpaceDimension()),
              finder(MPI_COMM_WORLD)
        {
            // Find boundary elements with the specified attributes (on the destination mesh)
            for (int be = 0; be < dst_mesh->GetNBE(); be++)
            {
                const int bdr_el_attr = dst_mesh->GetBdrAttribute(be);
                if (bdr_attributes[bdr_el_attr - 1] == 0)
                {
                    continue;
                }
                bdr_element_idx.Append(be);
            }

            // Compute the coordinates of the quadrature points on the boundary elements with the specified attributes (on the destination mesh)
            const IntegrationRule &ir_face = (dst_fes->GetTypicalBE())->GetNodes();
            bdr_element_coords.SetSize(bdr_element_idx.Size() * ir_face.GetNPoints() * sdim);
            bdr_element_coords = 0.0;

            auto pec = Reshape(bdr_element_coords.ReadWrite(), sdim, ir_face.GetNPoints(), bdr_element_idx.Size());
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

            // Setup finder (depeding on the backend)
            if (backend == Backend::Native || backend == Backend::Hybrid)
            {
                transfer_map = new ParTransferMap(src_gf, dst_gf);
            }

            if (backend == Backend::GSLIB || backend == Backend::Hybrid)
            {
                finder.Setup(*src_mesh);
                finder.FindPoints(bdr_element_coords, Ordering::byVDIM); // --> Because the coordinates are stored byVDIM
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

            // Nothing else to clean, InterfaceTransfer doesn't take ownership of the mesh or FiniteElementSpace
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
                TransferQoIToDestGf(interp_vals, dst_gf);
            }
        }


        void InterfaceTransfer::InterpolateQoI(qoi_func_t qoi_func, ParGridFunction &dst_gf)
        {
            MFEM_VERIFY(backend == Backend::GSLIB || backend == Backend::Hybrid,
                        "InterpolateQoI is not supported for the Native backend, and requires GSLIB.");

            // Extract space dimension and FE space from the mesh
            ParFiniteElementSpace* fes_dst = dst_gf.ParFESpace();
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
                ip.Set3(recv_rst(sdim * i + 0), recv_rst(sdim * i + 1),
                        recv_rst(sdim * i + 2));

                // Get the element transformation
                ElementTransformation *Tr = src_fes->GetElementTransformation(e);
                Tr->SetIntPoint(&ip);

                // Extract the local qoi vector
                qoi_loc.MakeRef(qoi_src, i*vdim, vdim);

                // Compute the qoi_src at quadrature point (it will change the qoi_src vector)
                qoi_func(*Tr, ip, qoi_loc);
            }

            // Transfer the QoI from the source mesh to the destination mesh at quadrature points
            finder.DistributeInterpolatedValues(qoi_src, vdim, Ordering::byVDIM, qoi_dst);

            // Transfer the QoI to the destination grid function
            TransferQoIToDestGf(qoi_dst, dst_gf);
        }
        

        inline void InterfaceTransfer::TransferQoIToDestGf(const Vector &dst_vec, ParGridFunction &dst_gf)
        {
            // Extract fe space from the destination grid function
            ParFiniteElementSpace *dst_fes = dst_gf.ParFESpace();
            int vdim = dst_fes->GetVDim();
            auto ordering = dst_fes->GetOrdering();
            const int dof = dst_fes->GetTypicalBE()->GetNodes().GetNPoints();

            int idx, be_idx, qp_idx;
            qoi_loc.SetSize(vdim);
            Vector loc_values(dof * vdim);
            Array<int> vdofs(dof * vdim);
            for (int be = 0; be < bdr_element_idx.Size(); be++)
            {
                be_idx = bdr_element_idx[be];
                dst_fes->GetBdrElementVDofs(be_idx, vdofs);
                for (int qp = 0; qp < dof; qp++)
                {
                    qp_idx = be * dof + qp;
                    qoi_loc = Vector(dst_vec.GetData() + qp_idx * vdim, vdim);
                    for (int d = 0; d < vdim; d++)
                    {
                        idx = ordering == Ordering::byVDIM ? qp * vdim + d : dof * d + qp;
                        loc_values(idx) = qoi_loc(d);
                    }
                }
                dst_gf.SetSubVector(vdofs, loc_values);
            }
        }
        

        void InterfaceTransfer::GSLIBAttrToMarker(int max_attr, const Array<unsigned int> elems, Array<int> &marker)
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


        void InterfaceTransfer::GetElementIdx( Array<int> &elem_idx ) 
        {
            MFEM_VERIFY(backend == Backend::GSLIB || backend == Backend::Hybrid,
                        "GetElementIdx is not supported for the Native backend, and requires GSLIB.");

            auto &finder_elem = finder.GetElem();
            GSLIBAttrToMarker(src_mesh->GetNE(), finder.GetElem(), elem_idx);
        }


    } // namespace ecm2_utils

} // namespace mfem
