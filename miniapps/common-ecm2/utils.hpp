#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <memory>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif // MFEM_USE_MPI

#include "mfem.hpp"

#include "../../linalg/dtensor.hpp"

#ifndef MFEM_ECM2_UTILS_HPP
#define MFEM_ECM2_UTILS_HPP

namespace mfem
{

    namespace ecm2_utils
    {
        /// Typedefs

        // Vector and Scalar functions (time independent)
        using VecFunc = void(const Vector &x, Vector &u);
        using ScalarFunc = double(const Vector &x);

        // Vector and Scalar functions (time dependent)
        using VecFuncT = void(const Vector &x, double t, Vector &u);
        using ScalarFuncT = double(const Vector &x, double t);
        using qoi_func_t = std::function<void(ElementTransformation &, int, const IntegrationPoint &)>; // QoI function type for GSLIB interpolation

        void print_matrix(const DenseMatrix &A);


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                          Solver utils                                                ///
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Struct to pass slver parameters
        struct SolverParams
        {
            double rtol = 1e-6;
            double atol = 1e-10;
            int maxIter = 1000;
            int pl = 0;

            SolverParams(double rtol_ = 1e-6, double atol_ = 1e-10, int maxIter_ = 1000, int pl_ = 0)
                : rtol(rtol_), atol(atol_), maxIter(maxIter_), pl(pl_) {}
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                          Coefficient utils                                           ///
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /// Container for vector coefficient holding: coeff and mesh attribute (useful for BCs and forcing terms).
        class VecCoeffContainer
        {
        public:
            VecCoeffContainer(Array<int> attr, VectorCoefficient *coeff_, bool own = true)
                : attr(attr)
            {
                this->coeff = coeff_;
            }

            VecCoeffContainer(VecCoeffContainer &&obj)
            {
                // Deep copy the attribute array
                this->attr = obj.attr;
                this->own = obj.own;

                // Move the coefficient pointer
                this->coeff = obj.coeff;
                obj.coeff = nullptr;
            }

            ~VecCoeffContainer()
            {
                if (own)
                {
                    delete coeff;
                    coeff = nullptr;
                }
            }

            Array<int> attr;
            VectorCoefficient *coeff = nullptr;
            bool own;
        };

        /// Container for vector coefficient holding: coeff and mesh attribute (useful for BCs and forcing terms).
        class CustomNeumannContainer
        {
        public:
            CustomNeumannContainer(Array<int> attr, Coefficient *alpha_, ParGridFunction *u_, Coefficient *beta_, ParGridFunction *p_, bool own = true)
                : attr(attr), own(own)
            {
                this->u = u_;
                this->p = p_;
                this->alpha = alpha_;
                this->beta = beta_;
            }

            CustomNeumannContainer(CustomNeumannContainer &&obj)
            {
                // Deep copy the attribute array
                this->attr = obj.attr;
                this->own = obj.own;

                // Move the coefficient pointer
                this->u = obj.u;
                this->p = obj.p;
                this->alpha = obj.alpha;
                this->beta = obj.beta;
                obj.u = nullptr;
                obj.p = nullptr;
                obj.alpha = nullptr;
                obj.beta = nullptr;
            }

            ~CustomNeumannContainer()
            {
                if (own)
                {
                    delete u;
                    delete p;
                    delete alpha;
                    delete beta;
                    u = nullptr;
                    p = nullptr;
                    alpha = nullptr;
                    beta = nullptr;
                }
            }

            Array<int> attr;
            ParGridFunction *u = nullptr;
            ParGridFunction *p = nullptr;
            Coefficient *alpha = nullptr;
            Coefficient *beta = nullptr;
            bool own;
        };

        /// Container for coefficient holding: coeff, mesh attribute id (i.e. not the full array)
        class CoeffContainer
        {
        public:
            CoeffContainer(Array<int> attr, Coefficient *coeff, bool own = true)
                : attr(attr), coeff(coeff), own(own)
            {
            }

            CoeffContainer(CoeffContainer &&obj)
            {
                // Deep copy the attribute and direction
                this->attr = obj.attr;
                this->own = obj.own;

                // Move the coefficient pointer
                this->coeff = obj.coeff;
                obj.coeff = nullptr;
            }

            ~CoeffContainer()
            {
                if (own)
                {
                    delete coeff;
                    coeff = nullptr;
                }
            }

            Array<int> attr;
            Coefficient *coeff;
            bool own;
        };

        /// Container for componentwise coefficient holding: coeff, mesh attribute id (i.e. not the full array) and direction (x,y,z) (useful for componentwise BCs).
        class CompCoeffContainer : public CoeffContainer
        {
        public:
            // Constructor for CompCoeffContainer
            CompCoeffContainer(Array<int> attr, Coefficient *coeff, int dir, bool own = true)
                : CoeffContainer(attr, coeff, own), dir(dir)
            {
            }

            // Move Constructor
            CompCoeffContainer(CompCoeffContainer &&obj)
                : CoeffContainer(std::move(obj))
            {
                dir = obj.dir;
            }

            // Destructor
            ~CompCoeffContainer() {}

            int dir;
        };

        /// Container for coefficient used for Robin bcs (n.Grad(u) + a u = b) holding: h_coeff, T0_coeff, mesh attribute id (i.e. not the full array)
        class RobinCoeffContainer
        {
        public:
            RobinCoeffContainer(Array<int> attr, Coefficient *h_coeff, Coefficient *T0_coeff, bool own = true)
                : attr(attr), h_coeff(h_coeff), T0_coeff(T0_coeff), own(own)
            {
                hT0_coeff = new ProductCoefficient(*h_coeff, *T0_coeff);
            }

            RobinCoeffContainer(RobinCoeffContainer &&obj)
            {
                // Deep copy the attribute and direction
                this->attr = obj.attr;
                this->own = obj.own;

                // Move the coefficient pointer
                this->h_coeff = obj.h_coeff;
                this->T0_coeff = obj.T0_coeff;
                this->hT0_coeff = obj.hT0_coeff;
                obj.h_coeff = nullptr;
                obj.T0_coeff = nullptr;
                obj.hT0_coeff = nullptr;
            }

            ~RobinCoeffContainer()
            {
                if (own)
                {
                    delete h_coeff;
                    delete T0_coeff;
                    delete hT0_coeff;
                    h_coeff = nullptr;
                    T0_coeff = nullptr;
                    hT0_coeff = nullptr;
                }
            }

            Array<int> attr;
            Coefficient *h_coeff;
            Coefficient *T0_coeff;
            ProductCoefficient *hT0_coeff;
            bool own;
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                          Linalg utils                                                ///
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /** @brief Matrix vector multiplication with the original uneliminated
        matrix.  The original matrix is \f$ mat + mat_e \f$ so we have:
        \f$ y = mat x + mat_e x \f$ */
        void FullMult(HypreParMatrix *mat, HypreParMatrix *mat_e, Vector &x, Vector &y);

        /** @brief Addition of matrix vector multiplication with the original uneliminated
           matrix.  The original matrix is \f$ mat + mat_e \f$ so we have:
           \f$ y += a ( mat x + mat_e x ) \f$ */
        void FullAddMult(HypreParMatrix *mat, HypreParMatrix *mat_e, Vector &x, Vector &y, double a = 1.0);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                              Mesh utils                                              ///
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        void ExportMeshwithPartitioning(const std::string &outfolder, Mesh &mesh, const int *partitioning_);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                          GSLIB utils                                                ///
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Compute the coordinates of the quadrature points on the boundary elements with the specified attributes (on Destination mesh)
        // Note: when using SubMeshes, provide bdry_attributes from the parent mesh, to ensure that
        void ComputeBdrQuadraturePointsCoords(Array<int> &bdry_attributes, ParFiniteElementSpace &fes, std::vector<int> &bdry_element_idx, Vector &bdry_element_coords);

        // GSLIB interpolation of the QoI (given by qoi_func) on the source mesh and transfer to the destination mesh quadrature points
        void GSLIBInterpolate(FindPointsGSLIB &finder, ParFiniteElementSpace &fes, qoi_func_t qoi_func, Vector &qoi_src, Vector &qoi_dst, int qoi_size_on_qp = 1);

        // Fill the grid function on destination mesh with the QoI vector (on quadrature points) on the destination mesh computed with GSLIBInterpolate
        void TransferQoIToDest(const std::vector<int> &elem_idx, const ParFiniteElementSpace &dest_fes, const Vector &dest_vec, ParGridFunction &dest_gf);

    } // namespace ecm2_utils

} // namespace mfem

#endif