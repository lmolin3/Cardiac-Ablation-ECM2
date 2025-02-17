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
        using qoi_func_t = std::function<void(ElementTransformation &, const IntegrationPoint &, Vector& qoi_src)>; // QoI function type for GSLIB interpolation

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
                : attr(attr), own(own)
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
        * matrix.  The original matrix.  The original matrix is $ mat + mat_e $ so we have:
        * $ y = mat x + mat_e x $
        */
        void FullMult(HypreParMatrix *mat, HypreParMatrix *mat_e, Vector &x, Vector &y);

        /** @brief Addition of matrix vector multiplication with the original uneliminated
        * matrix.  The original matrix is $ mat + mat_e $ so we have:
        * $ y += a ( mat x + mat_e x ) $
        */
        void FullAddMult(HypreParMatrix *mat, HypreParMatrix *mat_e, Vector &x, Vector &y, double a = 1.0);

        /// Remove mean from a Vector.
        /**
         * Modify the Vector @a v by subtracting its mean using
         * \f$v = v - \frac{\sum_i^N v_i}{N} \f$
         */
        void Orthogonalize(Vector &v, const MPI_Comm &comm);

        /// Remove the mean from a ParGridFunction.
        /**
         * Modify the ParGridFunction @a v by subtracting its mean using
         * \f$ v = v - \int_\Omega \frac{v}{vol(\Omega)} dx \f$.
         */
        void MeanZero(ParGridFunction &v, ParLinearForm *mass_lf = nullptr, real_t volume = 0.0);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                              Mesh utils                                              ///
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        void ExportMeshwithPartitioning(const std::string &outfolder, Mesh &mesh, const int *partitioning_);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                          Boundary transfer utils                                                ///
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Logical and operation between two arrays
        Array<int> operator&&(const Array<int> &a, const Array<int> &b);
       
        /**
         * @class InterfaceTransfer
         * @brief A class to handle the transfer of data between different grid functions.
         *
         * This class provides methods to interpolate data between source and destination grid functions,
         * as well as methods to handle quantities of interest (QoI).
         */
        class InterfaceTransfer
        {
            public:
        
                /**
                 * @enum Backend
                 * @brief Enumeration for different backend types.
                 */
                enum class Backend {Native, GSLIB, Hybrid};
        
                /**
                 * @brief Constructor for InterfaceTransfer.
                 * 
                 * @param src_gf Source grid function.
                 * @param dst_gf Destination grid function.
                 * @param bdr_attributes_ Boundary attributes.
                 * @param backend_ Backend type (default is Hybrid).
                 */
                InterfaceTransfer(ParGridFunction &src_gf, ParGridFunction &dst_gf, Array<int> &bdr_attributes_, InterfaceTransfer::Backend backend_ = InterfaceTransfer::Backend::GSLIB);
        
                /**
                 * @brief Destructor for InterfaceTransfer.
                 */
                ~InterfaceTransfer();
        
                /**
                 * @brief Interpolates data from source grid function to destination grid function.
                 * 
                 * @param src_gf Source grid function.
                 * @param dst_gf Destination grid function.
                 */
                void Interpolate(ParGridFunction &src_gf, ParGridFunction &dst_gf);
        
                /**
                 * @brief Interpolates quantities of interest (QoI) to the destination grid function.
                 * 
                 * @param qoi_func Function to compute the quantity of interest.
                 * @param dst_gf Destination grid function.
                 */
                void InterpolateQoI(qoi_func_t qoi_func, ParGridFunction &dst_gf);
        
                /**
                 * @brief Gets the boundary element indices.
                 * 
                 * @return A constant reference to the array of boundary element indices.
                 */
                const Array<int>& GetBdrElementIdx() const { return bdr_element_idx; }
        
                /**
                 * @brief Gets the boundary element coordinates.
                 * 
                 * @return A constant reference to the vector of boundary element coordinates.
                 */
                const Vector& GetBdrElementCoords() const { return bdr_element_coords; }
        
                /**
                 * @brief Gets the element indices.
                 * 
                 * @param elem_idx Array to store the element indices.
                 */
                void GetElementIdx(Array<int> &elem_idx);
        
            private:
                /**
                 * @brief Transfers quantities of interest (QoI) to the destination grid function.
                 * 
                 * @param dst_vec Destination vector.
                 * @param dst_gf Destination grid function.
                 */
                inline void TransferQoIToDestGf(const Vector &dst_vec, ParGridFunction &dst_gf);
        
                /**
                 * @brief Converts GSLIB attributes to marker.
                 * 
                 * @param max_attr Maximum attribute value.
                 * @param elems Array of elements.
                 * @param marker Array to store the marker.
                 */
                void GSLIBAttrToMarker(int max_attr, const Array<unsigned int> elems, Array<int> &marker);
        
                InterfaceTransfer::Backend backend; ///< Backend type.
        
                ParTransferMap *transfer_map; ///< Transfer map.
                FindPointsGSLIB finder; ///< GSLIB finder.
        
                ParFiniteElementSpace *src_fes; ///< Source finite element space.
                ParFiniteElementSpace *dst_fes; ///< Destination finite element space.
                ParMesh *src_mesh; ///< Source mesh.
                ParMesh *dst_mesh; ///< Destination mesh.
                int sdim; ///< Spatial dimension.
        
                Array<int> bdr_attributes; ///< Boundary attributes.
                Array<int> bdr_element_idx; ///< Boundary element indices.
                Vector bdr_element_coords; ///< Boundary element coordinates.
        
                mutable Vector interp_vals; ///< Interpolated values.
                mutable Vector qoi_loc, qoi_src, qoi_dst; ///< Quantities of interest.
        };

        /**
         * @class BidirectionalInterfaceTransfer
         * @brief A class to handle bidirectional transfer of data between different grid functions across interface.
         *
         * This class provides methods to interpolate data in both forward (S -> D) and backward directions (S <- D)
         * between source and destination grid functions, as well as methods to handle quantities of interest (QoI).
         * The source and destination are determined by the order of the grid functions passed to the constructor.
         */
        class BidirectionalInterfaceTransfer 
        {
            public:
                /**
                 * @brief Constructor for BidirectionalInterfaceTransfer.
                 * 
                 * @param src_gf Source grid function.
                 * @param dst_gf Destination grid function.
                 * @param bdr_attributes_ Boundary attributes.
                 * @param backend_ Backend type (default is GSLIB).
                 */
                BidirectionalInterfaceTransfer(ParGridFunction &src_gf, ParGridFunction &dst_gf, Array<int> &bdr_attributes_, InterfaceTransfer::Backend backend_ = InterfaceTransfer::Backend::GSLIB)
                    : forward_transfer(src_gf, dst_gf, bdr_attributes_, backend_), backward_transfer(dst_gf, src_gf, bdr_attributes_, backend_) {}
        
                ~BidirectionalInterfaceTransfer() = default;
        
                /**
                 * @brief Interpolates data from source mesh to destination mesh (forward direction).
                 * 
                 * @param src_gf Grid function on the source mesh.
                 * @param dst_gf Grid function on the destination mesh.
                 */
                void InterpolateForward(ParGridFunction &src_gf, ParGridFunction &dst_gf) { forward_transfer.Interpolate(src_gf, dst_gf); }
        
                /**
                 * @brief Interpolates data from destination grid function to source grid function (backward direction).
                 * 
                 * @param src_gf Grid function on the destination mesh.
                 * @param dst_gf Grid function on the source mesh.
                 */
                void InterpolateBackward(ParGridFunction &src_gf, ParGridFunction &dst_gf) { backward_transfer.Interpolate(src_gf, dst_gf); }
        
                /**
                 * @brief Interpolates quantities of interest (QoI) to the destination mesh (forward direction).
                 * 
                 * @param qoi_func Function to compute the quantity of interest.
                 * @param dst_gf Grid function on the destination mesh.
                 */
                void InterpolateQoIForward(qoi_func_t qoi_func, ParGridFunction &dst_gf) { forward_transfer.InterpolateQoI(qoi_func, dst_gf); }
        
                /**
                 * @brief Interpolates quantities of interest (QoI) to the source mesh (backward direction).
                 * 
                 * @param qoi_func Function to compute the quantity of interest.
                 * @param dst_gf Grid function on the source mesh.
                 */
                void InterpolateQoIBackward(qoi_func_t qoi_func, ParGridFunction &dst_gf) { backward_transfer.InterpolateQoI(qoi_func, dst_gf); }
        
                const Array<int>& GetBdrElementIdxSrc() const { return forward_transfer.GetBdrElementIdx(); }
                const Array<int>& GetBdrElementIdxDst() const { return backward_transfer.GetBdrElementIdx(); }
                const Vector& GetBdrElementCoordsSrc() const { return forward_transfer.GetBdrElementCoords(); }
                const Vector& GetBdrElementCoordsDst() const { return backward_transfer.GetBdrElementCoords(); }
                void GetElementIdxSrc(Array<int> &elem_idx) { forward_transfer.GetElementIdx(elem_idx); }
                void GetElementIdxDst(Array<int> &elem_idx) { backward_transfer.GetElementIdx(elem_idx); }
        
            private:
                InterfaceTransfer forward_transfer; ///< Forward transfer object.
                InterfaceTransfer backward_transfer; ///< Backward transfer object.
        };

    } // namespace ecm2_utils

} // namespace mfem

#endif