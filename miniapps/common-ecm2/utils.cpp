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

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                     GSLIB Interpolation utils                                         ///
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////

        void FindBdryElements(ParMesh *mesh, Array<int> &bdry_attributes, Array<int> &bdry_element_idx)
        {
            // Get the boundary elements with the specified attributes
            bdry_element_idx.DeleteAll();
            for (int be = 0; be < mesh->GetNBE(); be++)
            {
                const int bdr_el_attr = mesh->GetBdrAttribute(be);
                if (bdry_attributes[bdr_el_attr - 1] == 0)
                {
                    continue;
                }
                bdry_element_idx.Append(be);
            }

            // Print the number of boundary elements on each MPI core
            //int local_bdry_element_count = bdry_element_idx.Size();
            //mfem::out << "Number of boundary elements, for MPI Core " << mesh->GetMyRank() << ": " << local_bdry_element_count << std::endl;

            return;
        }

        
        void ComputeBdrQuadraturePointsCoords(ParFiniteElementSpace *fes, Array<int> &bdry_element_idx, Vector &bdry_element_coords)
        {
            // Return if fes is nullptr (useful for cases in which Mpi communicator is split)
            if (!fes || bdry_element_idx.Size() == 0)
                return;

            // Extract the coordinates of the quadrature points for each selected boundary element
            int sdim = fes->GetParMesh()->SpaceDimension();
            const IntegrationRule &ir_face = (fes->GetTypicalBE())->GetNodes();
            bdry_element_coords.SetSize(bdry_element_idx.Size() *
                                        ir_face.GetNPoints() * sdim);
            bdry_element_coords = 0.0;

            auto pec = Reshape(bdry_element_coords.ReadWrite(), sdim,
                               ir_face.GetNPoints(), bdry_element_idx.Size());

            for (int be = 0; be < bdry_element_idx.Size(); be++)
            {
                int be_idx = bdry_element_idx[be];
                const FiniteElement *fe = fes->GetBE(be_idx);
                ElementTransformation *Tr = fes->GetBdrElementTransformation(be_idx);
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


        void GSLIBAttrToMarker(int max_attr, const Array<unsigned int> elems, Array<int> &marker)
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

        void GSLIBInterpolate(FindPointsGSLIB &finder, const Array<int> &bdry_element_idx, ParFiniteElementSpace *fes, qoi_func_t qoi_func, ParGridFunction &dest_gf, int qoi_size_on_qp)
        {
            // Return if fes is nullptr (useful for cases in which Mpi communicator is split)
            if (!fes)
                return;

            // Extract space dimension and FE space from the mesh
            int sdim = (fes->GetParMesh())->SpaceDimension();
            int vdim = fes->GetVDim();

            // Distribute internal GSLIB info to the corresponding mpi-rank for each point.
            Array<unsigned int>
                recv_elem,
                recv_code;   // Element and GSLIB code
            Vector recv_rst; // (Reference) coordinates of the quadrature points
            finder.DistributePointInfoToOwningMPIRanks(recv_elem, recv_rst, recv_code);
            int npt_recv = recv_elem.Size();

            // Compute qoi locally (on source side)
            Vector qoi_loc, qoi_src, qoi_dst; 
            qoi_src.SetSize(npt_recv * qoi_size_on_qp);
            qoi_loc.SetSize(qoi_size_on_qp);
            for (int i = 0; i < npt_recv; i++)
            {
                // Get the element index
                const int e = recv_elem[i];

                // Get the quadrature point
                IntegrationPoint ip;
                ip.Set3(recv_rst(sdim * i + 0), recv_rst(sdim * i + 1),
                        recv_rst(sdim * i + 2));

                // Get the element transformation
                ElementTransformation *Tr = fes->GetElementTransformation(e);
                Tr->SetIntPoint(&ip);

                // Extract the local qoi vector
                qoi_loc.MakeRef(qoi_src, i*vdim, vdim);

                // Compute the qoi_src at quadrature point (it will change the qoi_src vector)
                qoi_func(*Tr, ip, qoi_loc);
            }

            // Transfer the QoI from the source mesh to the destination mesh at quadrature points
            finder.DistributeInterpolatedValues(qoi_src, qoi_size_on_qp, Ordering::byVDIM, qoi_dst);

            // Transfer the QoI to the destination grid function
            TransferQoIToDest(bdry_element_idx, qoi_dst, dest_gf);
        }


        void GSLIBTransfer( FindPointsGSLIB &finder, const Array<int> &bdry_element_idx, ParGridFunction &src_gf, ParGridFunction &dest_gf)
        {
            // Extract space dimension and FE space from the mesh
            ParFiniteElementSpace *fes = src_gf.ParFESpace();
            int sdim = (fes->GetParMesh())->SpaceDimension();
            int vdim = fes->GetVDim();

            // Distribute internal GSLIB info to the corresponding mpi-rank for each point.
            Array<unsigned int>
                recv_elem,
                recv_code;   // Element and GSLIB code
            Vector recv_rst; // (Reference) coordinates of the quadrature points
            finder.DistributePointInfoToOwningMPIRanks(recv_elem, recv_rst, recv_code);
            int npt_recv = recv_elem.Size();

            // Compute qoi locally (on source side)
            Vector qoi_loc, qoi_src, qoi_dst; 
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
                ElementTransformation *Tr = fes->GetElementTransformation(e);
                Tr->SetIntPoint(&ip);

                // Extract the local qoi vector
                qoi_loc.MakeRef(qoi_src, i*vdim, vdim);

                // Compute the qoi_src at quadrature point (it will change the qoi_src vector)
                src_gf.GetVectorValue(*Tr, ip, qoi_loc);
            }

            // Transfer the QoI from the source mesh to the destination mesh at quadrature points
            finder.DistributeInterpolatedValues(qoi_src, vdim, Ordering::byVDIM, qoi_dst);

            // Fill the grid function on destination mesh with the interpolated values
            TransferQoIToDest(bdry_element_idx, qoi_dst, dest_gf);
        }


        void GSLIBTransfer_old( FindPointsGSLIB &finder, const Array<int> &bdry_element_idx, ParGridFunction &src_gf, ParGridFunction &dest_gf)
        {
            // Interpolate the grid function on the source mesh to the destination mesh
            Vector interp_vals;
            finder.Interpolate(src_gf, interp_vals);

            // Fill the grid function on destination mesh with the interpolated values
            TransferQoIToDest(bdry_element_idx, interp_vals, dest_gf);    
        }

        inline void TransferQoIToDest(const Array<int> &bdry_element_idx, const Vector &dest_vec, ParGridFunction &dest_gf)
        {
            // Extract fe space from the destination grid function
            ParFiniteElementSpace *dest_fes = dest_gf.ParFESpace();
            int vdim = dest_fes->GetVDim();
            auto ordering = dest_fes->GetOrdering();
            const int dof = dest_fes->GetTypicalBE()->GetNodes().GetNPoints();

            int idx, be_idx, qp_idx;
            Vector qoi_loc(vdim), loc_values(dof * vdim);
            Array<int> vdofs(dof * vdim);
            for (int be = 0; be < bdry_element_idx.Size(); be++)
            {
                be_idx = bdry_element_idx[be];
                dest_fes->GetBdrElementVDofs(be_idx, vdofs);
                for (int qp = 0; qp < dof; qp++)
                {
                    qp_idx = be * dof + qp;
                    qoi_loc = Vector(dest_vec.GetData() + qp_idx * vdim, vdim);
                    for (int d = 0; d < vdim; d++)
                    {
                        idx = ordering == Ordering::byVDIM ? qp * vdim + d : dof * d + qp;
                        loc_values(idx) = qoi_loc(d);
                    }
                }
                dest_gf.SetSubVector(vdofs, loc_values);
            }
        }


    } // namespace ecm2_utils

} // namespace mfem
