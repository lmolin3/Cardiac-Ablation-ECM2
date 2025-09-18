#ifndef MFEM_ECM2_INTERFACE_TRANSFER_HPP
#define MFEM_ECM2_INTERFACE_TRANSFER_HPP

#include "mfem.hpp"

#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <memory>

#include "../../linalg/dtensor.hpp"
#include "general/forall.hpp"

namespace mfem
{

    namespace ecm2_utils
    {
        using qoi_func_t = std::function<void(ElementTransformation &, const IntegrationPoint &, Vector &qoi_src)>; // QoI function type for GSLIB interpolation

        /**
         * @brief Reorders the vector by VDIM.
         *
         * @param v_in Input vector.
         * @param v_out Output vector.
         * @param vdim VDIM to reorder by.
         *
         * This method is used to reorder the vector such that the elements are ordered by VDIM.
         */
        void ReorderByVDIM(const Vector &v_in, Vector &v_out, int vdim);

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
            enum class Backend
            {
                Native,
                GSLIB,
                Hybrid
            };

            /**
             * @brief Constructor for InterfaceTransfer.
             *
             * @param src_fes Source finite element space.
             * @param dst_fes Destination finite element space.
             * @param bdr_attributes_ Boundary attributes (Array<int> compatible with destination mesh bdr_attributes,
             * with non-zero values for attributes associated with the desired interface).
             * @param backend_ Backend type (default is Hybrid).
             * @param comm_ MPI communicator (default is MPI_COMM_WORLD).
             * @param use_undeformed_coords_ If true, uses undeformed coordinates for the interface transfer
             *                               (i.e. the ones computed in the constructor after calling SetupGSLIB()).
             *                               If false, uses the current coordinates of the source mesh (e.g. deformed in FSI).
             *                               Default behaviour is true, for FSI simulations, depending on the direction of transfer it 
             *                               may be necessary to set this flag to false.
             */
            InterfaceTransfer(ParFiniteElementSpace *src_fes_, ParFiniteElementSpace *dst_fes_,
                              Array<int> &bdr_attributes_, InterfaceTransfer::Backend backend_ = InterfaceTransfer::Backend::GSLIB,
                              MPI_Comm comm_ = MPI_COMM_WORLD, bool use_undeformed_coords_ = true);

            /**
             * @brief Destructor for InterfaceTransfer.
             */
            ~InterfaceTransfer();

            /**
             * @brief Update InterfaceTransfer (GSLIB).
             *
             * This method updates the Finder.
             * If GSLIB is enabled:
             * 1. Finding the new coordinates of the boundary dofs
             * 2. Setup the GSLIB Finder
             * 3. Find the points in the GSLIB Finder
             *
             * If Native backend is used we recreate the TransferMap.
             * @note We don't need to provide the updated src mesh as long as it's the same pointed at construction time.
             * 
             * @param dx Optional offset coefficient defining the offset of the interface transfer (passed to SetupGSLIB()).
             */
            void Update( VectorCoefficient *dx = nullptr );

            /**
             * @brief Set flag to use undeformed coordinates for the interface transfer.
             * @param use_undeformed_coords_ If true, uses undeformed coordinates for the interface transfer.
             *
             * This method is used to set the flag that determines whether the undeformed coordinates should be used for the interface transfer.
             * Internally resets the GSLIB finder object.
             */
            void GSLIBUseUndeformedCoords(bool use_undeformed_coords_);

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
             * @param bdr_version If true, uses the boundary version of the interpolation.
             *
             *  This is the public API for the private InterpolateQoI methods.
             */
            void InterpolateQoI(qoi_func_t qoi_func, ParGridFunction &dst_gf, bool bdr_version = false);

            /**
             * @brief Gets the boundary element indices found by GSLIB.
             *
             * @return A constant reference to the array of boundary element indices.
             */
            const Array<int> &GetBdrElementIdx() const { return dst_bdr_element_idx; }

            /**
             * @brief Gets the boundary element coordinates found by GSLIB.
             *
             * @return A constant reference to the vector of boundary element coordinates.
             */
            const Vector &GetBdrElementCoords() const { return bdr_element_coords; }

            /**
             * @brief Gets the element indices found by GSLIB.
             *
             * @param elem_idx Array to store the element indices.
             */
            void GetElementIdx(Array<int> &elem_idx);

            /**
             * @brief Set name of the interface transfer.
             *  @param name_ Name of the interface transfer.
             */
            void SetName(const std::string &name_) { name = name_; }

            /**
             * @brief Set the boundary tolerance for GSLIB finder to find points on the boundary.
             * @param tol Tolerance value.
             * This method sets the boundary tolerance for the GSLIB finder to find points on the boundary.
             * @note This is useful when the source and destination meshes are not in the same configuration (e.g. FSI).
             */
            void SetBoundaryDistanceTolerance(real_t tol) { boundary_tol = tol; }

        private:
            /**
             * @brief Sets up the GSLIB finder.
             * 1. Find boundary elements with the specified attributes (on the destination mesh)
             * 2. Find the coordinates of the boundary elements
             * 3. (MPI) Gather the coordinates found from all ranks
             * 4. Setup FinPointsGSLIB ( Setup(), FindPoints() )
             * 
             * @param dx Optional offset coefficient defining the offset of the interface transfer.
             *                  Useful when the working with moving meshes (e.g. FSI), where the source and destination
             *                  meshes may be not be both in the deformed or reference configuration.
             *
             *  @note: In the first call to SetupGSLIB in the constructor, we don't pass the dx coefficient, as we assume that the source mesh is in the reference configuration.
             *  If the source mesh is already in the deformed configuration, we need to call Update() with the dx coefficient.
             */
            void SetupGSLIB( VectorCoefficient *dx = nullptr );


            /**
             * @brief Transfers quantities of interest (QoI) to the destination grid function.
             *
             * @param dst_vec Destination vector from interpolation.
             * @param dst_gf Destination grid function.
             * @param ordering Ordering of elements in dst_vec.
             *
             * @note: ordering of dst_vec will be:
             * byVDIM/byNODES, depending on field_in (if using Interpolate)
             * byVDIM (if using InterpolateQoI)
             * either way, it's not necessarily the same as dst_gf ordering
             */
            inline void TransferQoIToDestGf(const Vector &dst_vec, ParGridFunction &dst_gf, const mfem::Ordering::Type src_ordering);

            /**
             * @brief Interpolates quantities of interest (QoI) to the destination grid function.
             *
             * @param qoi_func Function to compute the quantity of interest.
             * @param dst_gf Destination grid function.
             */
            void InterpolateQoIv1(qoi_func_t qoi_func, ParGridFunction &dst_gf);

            /**
             * @brief Interpolates quantities of interest (QoI) on the boundary to the destination grid function.
             *
             * @param qoi_bdr_func Function to compute the quantity of interest on the boundary.
             * @param dst_gf Destination grid function.
             *
             * @note This method differs from InterpolateQoI in that it iterates over the boundary elements,
             * whether the GSLIB finder gives element indices. This means that the ElementTransformation Tr, is
             * the transformation for the i-th element and not the i-th boundary element.
             * So for example computing normals with CalcOrtho will fail.
             *
             * We assume that the (Par)FiniteElementSpace for this gf has the same vdim as the original src_fes provided to the constructor.
             *
             * Internally we iterate over boundary elements on the source mesh, populate a GridFunction on the source
             * for the QoI specified by qoi_bdr_func, and then transfer it to the destination grid function, using the internal
             * InterfaceTransfer::Interpolate method.
             */
            void InterpolateQoIv2(qoi_func_t qoi_bdr_func, ParGridFunction &dst_gf);

            /**
             * @brief Converts GSLIB attributes to marker.
             *
             * @param max_attr Maximum attribute value.
             * @param elems Array of elements.
             * @param marker Array to store the marker.
             */
            void GSLIBAttrToMarker(int max_attr, const Array<unsigned int> &elems, Array<int> &marker);

            InterfaceTransfer::Backend backend; ///< Backend type.

            ParTransferMap *transfer_map = nullptr; ///< Transfer map.  --> For now i don't care about serial, but in that case we'd need to handle two TransferMap and ParTransferMap based on MFEM_USE_MPI
            FindPointsGSLIB* finder = nullptr;      ///< GSLIB finder.

            ParFiniteElementSpace *src_fes; ///< Source finite element space.
            ParFiniteElementSpace *dst_fes; ///< Destination finite element space.
            ParMesh *src_mesh;              ///< Source mesh.
            ParMesh *dst_mesh;              ///< Destination mesh.
            int sdim;                       ///< Spatial dimension.
            Array<int> bdr_attributes;      ///< Boundary attributes.
            Array<int> dst_bdr_element_idx; ///< Destination boundary element indices.
            Array<int> src_bdr_element_idx; ///< Source boundary element indices.
            Vector bdr_element_coords;      ///< Boundary element coordinates.
            Vector undeformed_bdr_element_coords; ///< Undeformed boundary element coordinates.

            mutable ParGridFunction tmp_src_gf;       ///< Source grid function.
            mutable Vector interp_vals;               ///< Interpolated values.
            mutable Vector qoi_loc, qoi_src, qoi_dst; ///< Quantities of interest.
            mutable Vector local_dst_vec;             ///< Local destination vector.
            mutable Vector reordered_dst_vec;         ///< Reordered destination vector.

            real_t boundary_tol = 1e-8; ///< Boundary tolerance for GSLIB finder to find points on the boundary.

            mutable Vector x, dx_val; 

            std::string name = "InterfaceTransfer"; ///< Name of the interface transfer.

            bool use_undeformed_coords; ///< If true, uses undeformed coordinates for the interface transfer.
            bool gslib_initialized = false;

#ifdef MFEM_USE_MPI
            MPI_Comm comm;                                                ///< MPI communicator.
            int myid, nranks;                                             ///< MPI rank and number of ranks.
            Array<int> recvcounts_gather, recvcounts_scatter, recvcounts; ///< Array of receive counts.
            Array<int> displs_gather, displs_scatter;                     ///< Array of displacements.
#endif
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
             *
             * @note The bdr_attributes_ should be compatible with both source and destination meshes,
             * as an InterfaceTransfer object is created for both forward and backward transfers.
             * If the meshes are (Par)SubMeshes, create bdr_attributes_ from the parent mesh.
             */
            BidirectionalInterfaceTransfer(ParFiniteElementSpace *src_fes, ParFiniteElementSpace *dst_fes, Array<int> &bdr_attributes_,
                                           InterfaceTransfer::Backend backend_ = InterfaceTransfer::Backend::GSLIB, MPI_Comm comm = MPI_COMM_WORLD,
                                           bool use_undeformed_coords_forward_ = false, bool use_undeformed_coords_backward_ = false)
                : forward_transfer(src_fes, dst_fes, bdr_attributes_, backend_, comm, use_undeformed_coords_forward_), backward_transfer(dst_fes, src_fes, bdr_attributes_, backend_, comm, use_undeformed_coords_backward_)
                {
                    forward_transfer.SetName("BidirectionalInterfaceTransfer::Forward");
                    backward_transfer.SetName("BidirectionalInterfaceTransfer::Backward");
                }

                ~BidirectionalInterfaceTransfer() = default;

            /**
             * @brief Update the GSLIB finder.
             *
             * This method updates the GSLIB finder for forward/backward transfers (or both by default).
             *
             * @param forward If true, updates the forward transfer GSLIB finder.
             * @param backward If true, updates the backward transfer GSLIB finder.
             * @param dx_forward Optional offset coefficient for interface coordinates in the forward transfer.
             * @param dx_backward Optional offset coefficient for interface coordinates in the backward transfer.
             *
             */
            void Update(bool forward = true, bool backward = true,  VectorCoefficient *dx_forward = nullptr, VectorCoefficient *dx_backward = nullptr)
            {
                if (backward)
                {
                    backward_transfer.Update(dx_backward);
                }
                if (forward)
                {
                    forward_transfer.Update(dx_forward);
                }
            }

            void GSLIBUseUndeformedCoords(bool use_undeformed_coords_forward, bool use_undeformed_coords_backward)
            {
                forward_transfer.GSLIBUseUndeformedCoords(use_undeformed_coords_forward);
                backward_transfer.GSLIBUseUndeformedCoords(use_undeformed_coords_backward);
            }

            /**
             * @brief Interpolates data from source mesh to destination mesh (forward direction).
             *
             * @param src_gf Grid function on the source mesh.
             * @param dst_gf Grid function on the destination mesh.
             */
            void InterpolateForward(ParGridFunction &src_gf, ParGridFunction &dst_gf)
            {
                forward_transfer.Interpolate(src_gf, dst_gf);
            }

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
            void InterpolateQoIForward(qoi_func_t qoi_func, ParGridFunction &dst_gf, bool bdr_version = false) { forward_transfer.InterpolateQoI(qoi_func, dst_gf, bdr_version); }

            /**
             * @brief Interpolates quantities of interest (QoI) to the source mesh (backward direction).
             *
             * @param qoi_func Function to compute the quantity of interest.
             * @param dst_gf Grid function on the source mesh.
             */
            void InterpolateQoIBackward(qoi_func_t qoi_func, ParGridFunction &dst_gf, bool bdr_version = false) { backward_transfer.InterpolateQoI(qoi_func, dst_gf, bdr_version); }

            /**
             * @brief Set the boundary tolerance for GSLIB finder to find points on the boundary.
             * @param tol_forward Tolerance value for forward transfer.
             * @param tol_backward Tolerance value for backward transfer.
             * This method sets the boundary tolerance for the GSLIB finder to find points on the boundary.
             */
            void SetBoundaryDistanceTolerance(real_t tol_forward, real_t tol_backward)
            {
                forward_transfer.SetBoundaryDistanceTolerance(tol_forward);
                backward_transfer.SetBoundaryDistanceTolerance(tol_backward);
            }

            const Array<int> &GetBdrElementIdxSrc() const { return forward_transfer.GetBdrElementIdx(); }
            const Array<int> &GetBdrElementIdxDst() const { return backward_transfer.GetBdrElementIdx(); }
            const Vector &GetBdrElementCoordsSrc() const { return forward_transfer.GetBdrElementCoords(); }
            const Vector &GetBdrElementCoordsDst() const { return backward_transfer.GetBdrElementCoords(); }
            void GetElementIdxSrc(Array<int> &elem_idx) { forward_transfer.GetElementIdx(elem_idx); }
            void GetElementIdxDst(Array<int> &elem_idx) { backward_transfer.GetElementIdx(elem_idx); }

        private:
            InterfaceTransfer forward_transfer;  ///< Forward transfer object.
            InterfaceTransfer backward_transfer; ///< Backward transfer object.
        };

    } // namespace ecm2_utils

} // namespace mfem

#endif