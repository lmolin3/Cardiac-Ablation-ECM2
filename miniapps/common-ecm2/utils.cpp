#include "utils.hpp"

namespace mfem
{

    namespace ecm2_utils
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                          Linalg utils                                                ///
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                     GSLIB Interpolation utils                                         ///
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////

        void ComputeBdrQuadraturePointsCoords(Array<int> bdry_attributes, ParFiniteElementSpace &fes, std::vector<int> &bdry_element_idx, Vector &bdry_element_coords)
        {
            // Mesh
            ParMesh &mesh = *fes.GetParMesh();
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

            auto pec = Reshape(bdry_element_coords.ReadWrite(), sdim,
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
                qoi_func(*Tr, i, ip);
            }

            // Transfer the QoI from the source mesh to the destination mesh at quadrature points
            finder.DistributeInterpolatedValues(qoi_src, qoi_size_on_qp, Ordering::byVDIM, qoi_dst);
        }

        void TransferQoIToDest(const std::vector<int> &elem_idx, const ParFiniteElementSpace &dest_fes, const Vector &dest_vec, ParGridFunction &dest_gf)
        {

            int sdim = dest_fes.GetMesh()->SpaceDimension();
            int vdim = dest_fes.GetVDim();

            int dof, idx, be_idx, qp_idx;
            Vector qoi_loc(vdim), loc_values;
            for (int be = 0; be < elem_idx.size(); be++) // iterate over each BE on interface boundary and construct FE value from quadrature point
            {
                Array<int> vdofs;
                be_idx = elem_idx[be];
                dest_fes.GetBdrElementVDofs(be_idx, vdofs);
                const FiniteElement *fe = dest_fes.GetBE(be_idx);
                dof = fe->GetDof();
                loc_values.SetSize(dof * vdim);
                auto ordering = dest_fes.GetOrdering();
                for (int qp = 0; qp < dof; qp++)
                {
                    qp_idx = be * dof + qp;
                    qoi_loc = Vector(dest_vec.GetData() + qp_idx * vdim, vdim);
                    for (int d = 0; d < vdim; d++)
                    {
                        idx = ordering == Ordering::byVDIM ? qp * sdim + d : dof * d + qp;
                        loc_values(idx) = qoi_loc(d);
                    }
                }
                dest_gf.SetSubVector(vdofs, loc_values);
            }
        }

    } // namespace ecm2_utils

} // namespace mfem
