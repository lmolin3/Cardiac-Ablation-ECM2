#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <memory>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif // MFEM_USE_MPI

#include "mfem.hpp"

#include "../../linalg/dtensor.hpp"

#ifndef MFEM_ECM2_INTERFACE_TRANSFER_HPP
#define MFEM_ECM2_INTERFACE_TRANSFER_HPP

namespace mfem
{

    namespace ecm2_utils
    {
        using qoi_func_t = std::function<void(ElementTransformation &, const IntegrationPoint &, Vector& qoi_src)>; // QoI function type for GSLIB interpolation

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
                InterfaceTransfer(ParGridFunction &src_gf, ParGridFunction &dst_gf,
                                  Array<int> &bdr_attributes_, InterfaceTransfer::Backend backend_ = InterfaceTransfer::Backend::GSLIB,
                                  MPI_Comm comm_ = MPI_COMM_WORLD);
        
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
                 * @brief Gets the boundary element indices found by GSLIB.
                 * 
                 * @return A constant reference to the array of boundary element indices.
                 */
                const Array<int>& GetBdrElementIdx() const { return bdr_element_idx; }
        
                /**
                 * @brief Gets the boundary element coordinates found by GSLIB.
                 * 
                 * @return A constant reference to the vector of boundary element coordinates.
                 */
                const Vector& GetBdrElementCoords() const { return bdr_element_coords; }
        
                /**
                 * @brief Gets the element indices found by GSLIB.
                 * 
                 * @param elem_idx Array to store the element indices.
                 */
                void GetElementIdx(Array<int> &elem_idx);
        
            private:

                /**
                 * @brief Sets up the GSLIB finder.
                 * 1. Find boundary elements with the specified attributes (on the destination mesh)
                 * 2. Find the coordinates of the boundary elements
                 * 3. (MPI) Gather the coordinates found from all ranks
                 * 4. Setup FinPointsGSLIB ( Setup(), FindPoints() )
                 */
                void SetupGSLIB();


                /**
                 * @brief Transfers quantities of interest (QoI) to the destination grid function.
                 * 
                 * @param dst_vec Destination vector.
                 * @param dst_gf Destination grid function.
                 * @param ordering Ordering of elements in dst_vec.
                 */
                inline void TransferQoIToDestGf(const Vector &dst_vec, ParGridFunction &dst_gf, const int ordering);
        
                /**
                 * @brief Converts GSLIB attributes to marker.
                 * 
                 * @param max_attr Maximum attribute value.
                 * @param elems Array of elements.
                 * @param marker Array to store the marker.
                 */
                void GSLIBAttrToMarker(int max_attr, const Array<unsigned int> &elems, Array<int> &marker);
        
                InterfaceTransfer::Backend backend; ///< Backend type.

                ParTransferMap *transfer_map = nullptr; ///< Transfer map.
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
                mutable Vector local_dst_vec; ///< Local destination vector.
                
#ifdef MFEM_USE_MPI
                MPI_Comm comm;    ///< MPI communicator.
                int myid, nranks; ///< MPI rank and number of ranks.
                Array<int> recvcounts_gather, recvcounts_scatter, recvcounts; ///< Array of receive counts.
                Array<int> displs_gather, displs_scatter; ///< Array of displacements.
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
                 */
                BidirectionalInterfaceTransfer(ParGridFunction &src_gf, ParGridFunction &dst_gf, Array<int> &bdr_attributes_, InterfaceTransfer::Backend backend_ = InterfaceTransfer::Backend::GSLIB, MPI_Comm comm = MPI_COMM_WORLD)
                    : forward_transfer(src_gf, dst_gf, bdr_attributes_, backend_, comm), backward_transfer(dst_gf, src_gf, bdr_attributes_, backend_, comm) {}
        
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