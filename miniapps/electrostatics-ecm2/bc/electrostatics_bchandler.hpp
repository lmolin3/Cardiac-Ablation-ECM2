#ifndef BCHANDLER_ELECTROSTATICS_HPP
#define BCHANDLER_ELECTROSTATICS_HPP

#include <mfem.hpp>
#include "utils.hpp"

namespace mfem
{

    // Include functions from ecm2_utils namespace
    using namespace ecm2_utils;

    namespace electrostatics // Needed since more solvers will have a class BCHandler
    {

        class BCHandler
        {

        public:
            // Constructor
            BCHandler(std::shared_ptr<ParMesh> pmesh, bool verbose = true);

            /**
             * \brief Add Dirichlet BC for electric potential using Coefficient and list of essential mesh attributes.
             *
             * Add a Dirichlet boundary condition for electric potential to internal list of essential bcs passing Coefficient
             * and list of essential mesh attributes (they will be applied at setup time).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddDirichletBC(Coefficient *coeff, Array<int> &attr);

            /**
             * \brief Add Dirichlet BC for electric potential using Scalar function and list of essential mesh attributes.
             *
             * Add a Dirichlet boundary condition for electric potential to internal list of essential bcs passing function
             * and list of essential mesh attributes (they will be applied at setup time).
             *
             * \param func Pointer to ScalarFuncT
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddDirichletBC(ScalarFuncT *func, Array<int> &attr);

            /**
             * \brief Add Dirichlet BC for electric potential using Coefficient and specific mesh attribute.
             *
             * Add a Dirichlet boundary condition for electric potential to internal list of essential bcs passing Coefficient,
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Boundary attribute
             *
             */
            void AddDirichletBC(Coefficient *coeff, int &attr);

            /**
             * \brief Add Dirichlet BC for electric potential passing ScalarFuncT and specific mesh attribute.
             *
             * Add a Dirichlet boundary condition for electric potential to internal list of essential bcs passing ScalarFuncT
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param func Pointer to ScalarFuncT
             * \param attr Boundary attribute
             *
             */
            void AddDirichletBC(ScalarFuncT *func, int &attr);

            /**
             * \brief Add Dirichlet BC for electric potential using real_t and specific mesh attribute.
             *
             * Add a Dirichlet boundary condition for electric potential to internal list of essential bcs passing real_t,
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param coeff_val Value of the Dirichlet BC
             * \param attr Boundary attribute
             *
             */
            void AddDirichletBC(real_t coeff_val, int &attr);

            /**
             * \brief Add Dirichlet BC for electric potential using Coefficient and list of essential mesh attributes.
             *
             * Add a Dirichlet boundary condition for electric potential to internal list of essential bcs passing Coefficient
             * and list of essential mesh attributes (they will be applied at setup time).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddDirichletEFieldBC(Vector &EField, Array<int> &attr);

            /**
             * \brief Add Dirichlet BC for electric potential using Coefficient and specific mesh attribute.
             *
             * Add a Dirichlet boundary condition for electric potential to internal list of essential bcs passing Coefficient,
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Boundary attribute
             *
             */
            void AddDirichletEFieldBC(Vector &EField, int &attr);


            /**
             * \brief Add Neumann BC using Coefficient and list of essential boundaries.
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using Coefficient, and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddNeumannBC(Coefficient *coeff, Array<int> &attr);

            /**
             * \brief Add Neumann BC using ScalarFuncT and list of essential boundaries.
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using ScalarFuncT and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddNeumannBC(ScalarFuncT *coeff, Array<int> &attr);

            /**
             * \brief Add Neumann BC using Coefficient and specific mesh attribute.
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using Coefficient, and specific mesh attribute (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Boundary attribute
             *
             */
            void AddNeumannBC(Coefficient *coeff, int &attr);

            /**
             * \brief Add Neumann BC using ScalarFuncT and specific mesh attribute.
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using ScalarFuncT and specific mesh attribute(they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param func Pointer to ScalarFuncT
             * \param attr Boundary attribute
             *
             */
            void AddNeumannBC(ScalarFuncT *func, int &attr);

            /**
             * \brief Add Neumann BC using real_t and specific mesh attribute.
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using real_t and specific mesh attribute(they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param val Neumann value
             * \param attr Boundary attribute
             *
             */
            void AddNeumannBC(real_t val, int &attr);


            /**
             * \brief Add Neumann BC using VectorCoefficient and list of essential boundaries.
             *
             * (\vec{f} \cdot \vec{n}, v)
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using VectorCoefficient, and list of active mesh boundaries (they will be applied at setup time by adding BoundaryNormalIntegrators to the rhs).
             *
             * \param coeff Pointer to VectorCoefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddNeumannVectorBC(VectorCoefficient *coeff, Array<int> &attr, bool own = true);

            /**
             * \brief Add Neumann BC using VecFuncT and list of essential boundaries.
             *
             * (\vec{f} \cdot \vec{n}, v)
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using VecFuncT and list of active mesh boundaries (they will be applied at setup time by adding BoundaryNormalIntegrators to the rhs).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddNeumannVectorBC(VecFuncT *coeff, Array<int> &attr, bool own = true);

            /**
             * \brief Add Neumann BC using VectorCoefficient and specific mesh attribute.
             *
             * (\vec{f} \cdot \vec{n}, v)
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using VectorCoefficient, and specific mesh attribute (they will be applied at setup time by adding BoundaryNormalIntegrators to the rhs).
             *
             * \param coeff Pointer to VectorCoefficient
             * \param attr Boundary attribute
             *
             */
            void AddNeumannVectorBC(VectorCoefficient *coeff, int &attr, bool own = true);

            /**
             * \brief Add Neumann BC using VecFuncT h (heat transfer coefficient)nd specific mesh attribute.
             *
             * (\vec{f} \cdot \vec{n}, v)
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using VecFuncT and specific mesh attribute(they will be applied at setup time by adding BoundaryNormalIntegrators to the rhs).
             *
             * \param func Pointer to VecFuncT
             * \param attr Boundary attribute
             *
             */
            void AddNeumannVectorBC(VecFuncT *func, int &attr, bool own = true);


            /**
             * \brief Add Robin BC using Coefficient and list of essential boundaries.
             *
             * μ1 ∇φ⋅n + α1 φ =  μ2 ∇φ2⋅n + α2 φ2
             * 
             * Add a Robin boundary condition to internal list of Robin bcs,
             * using Coefficient, and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             * \param alpha_1 Pointer to Coefficient
             * \param alpha_2 Pointer to Coefficient
             * \param phi_2 Pointer to Coefficient
             * \param gradPhi_2 Pointer to Coefficient
             * \param mu2 Pointer to Coefficient
             *
             * Note: If μ2 is not provided, we assume gradPhi_2 = μ2 ∇φ2 
             */ 
            void AddRobinBC(Coefficient *alpha_1, Coefficient *alpha_2, Coefficient *phi_2, VectorCoefficient *gradPhi_2, Coefficient *mu2, Array<int> &attr, bool own = true); 


            /**
             * \brief Add Robin BC using Coefficient and specific mesh attribute.
             */
            void AddRobinBC(Coefficient *alpha_1, Coefficient *alpha_2, Coefficient *phi_2, VectorCoefficient *gradPhi_2, Coefficient *mu2, int &attr, bool own = true);


            /** \brief Add Robin BC using Coefficient and list of essential boundaries.
             * This version assumes gradPhi_2 = μ2 ∇φ2, since μ2 is not provided.
             */
            void AddRobinBC(Coefficient *alpha_1, Coefficient *alpha_2, Coefficient *phi_2, VectorCoefficient *mu2_gradPhi_2, Array<int> &attr, bool own = true);

            /** \brief Add Robin BC using Coefficient and specific mesh attribute.
             * This version assumes gradPhi_2 = μ2 ∇φ2, since μ2 is not provided.
             */
            void AddRobinBC(Coefficient *alpha_1, Coefficient *alpha_2, Coefficient *phi_2, VectorCoefficient *mu2_gradPhi_2, int &attr, bool own = true);


            /**
             * \brief Update the time in the velocity BCs coefficients.
             *
             * Update the time in the velocity BCs coefficients.
             *
             * \param new_time New time value.
             *
             */
            void UpdateTimeDirichletBCs(real_t new_time);

            /**
             * \brief Update the time in the pressure BCs coefficients.
             *
             * Update the time in the pressure BCs coefficients.
             *
             * \param new_time New time value.
             *
             */
            void UpdateTimeNeumannBCs(real_t new_time);

            /**
             * \brief Update the time in the Vector Neumann BCs coefficients.
             *
             * Update the time in the vector neumann BCs coefficients.
             *
             * \param new_time New time value.
             *
             */
            void UpdateTimeNeumannVectorBCs(real_t new_time);

            /**
             * \brief Update the time in the Robin BCs coefficients.
             *
             * Update the time in the Robin BCs coefficients.
             *
             * \param new_time New time value.
             *
             */
            void UpdateTimeRobinBCs(real_t new_time);

            // Getters

            // Getter for electric potential bcs
            std::vector<CoeffContainer> &GetDirichletDbcs()
            {
                return dirichlet_dbcs;
            }

            // Getter for electric potential bcs (producing uniform electric field)
            std::vector<CoeffContainer> &GetDirichletEFieldDbcs()
            {
                return dirichlet_EField_dbcs;
            }

            // Getter for Neumann_bcs
            std::vector<CoeffContainer> &GetNeumannBcs()
            {
                return neumann_bcs;
            }

            // Getter for Neumann_vector_bcs
            std::vector<VecCoeffContainer> &GetNeumannVectorBcs()
            {
                return neumann_vec_bcs;
            }

            // Getter for General Robin bcs
            std::vector<GeneralRobinContainer> &GetRobinBcs()
            {
                return robin_bcs;
            }

            // Getter for electric potential bcs
            Array<int> &GetDirichletAttr()
            {
                return dirichlet_attr;
            }


            // Getter for electric potential bcs (producing uniform electric field)
            Array<int> &GetDirichletEFieldAttr()
            {
                return dirichlet_EField_attr;
            }

            // Getter for Neumann_attr
            Array<int> &GetNeumannAttr()
            {
                return neumann_attr;
            }

            // Getter for Neumann_vector_attr
            Array<int> &GetNeumannVectorAttr()
            {
                return neumann_vec_attr;
            }

            // Getter for General Robin attr
            Array<int> &GetRobinAttr()
            {
                return robin_attr;
            }

        private:
            // Shared pointer to Mesh
            std::shared_ptr<ParMesh> pmesh;

            // Maximum number of boundary attributes
            int max_bdr_attributes;

            // Bookkeeping for velocity dirichlet bcs (full Vector coefficient).
            std::vector<CoeffContainer> dirichlet_dbcs;

            // Bookkeeping for velocity dirichlet bcs (full Vector coefficient).
            std::vector<CoeffContainer> dirichlet_EField_dbcs;

            // Bookkeeping for Neumann bcs.
            std::vector<CoeffContainer> neumann_bcs;

            // Bookkeeping for Neumann vector bcs.
            std::vector<VecCoeffContainer> neumann_vec_bcs;

            // Bookkeeping for general Robin bcs.
            std::vector<GeneralRobinContainer> robin_bcs;

            /// Array of attributes for bcs
            Array<int> dirichlet_attr;         // Essential mesh attributes.
            Array<int> dirichlet_EField_attr;  // Essential mesh attributes.
            Array<int> neumann_attr;           // Neumann mesh attributes.
            Array<int> neumann_vec_attr;       // Neumann mesh attributes.
            Array<int> robin_attr;             // General Robin mesh attributes.

            Array<int> dirichlet_attr_tmp;     // Essential mesh attributes (temporary).
            Array<int> neumann_attr_tmp;       // Neumann mesh attributes (temporary).
            Array<int> robin_attr_tmp;         // Robin mesh attributes (temporary).

            // Verbosity
            bool verbose;
        };

    }; // namespace electrostatics

}; // namespace mfem

#endif // BCHANDLER_NS_HPP
