/**
 * \file heat_bchandler.hpp
 * \brief This file contains the definition of the BCHandler class for managing boundary conditions in the Monodomain Diffusion problem.
 *
 * The BCHandler class is designed to handle various types of boundary conditions for heat transfer simulations, and is passed to the solver object.
 *
 * Boundary Conditions:
 * - Dirichlet Boundary Conditions:  u = ud
 *
 * - Neumann Boundary Conditions:    -σ∇u•n = g      (providing scalar field g) --> Applied as BoundaryLFIntegrator (f, v)
 *
 * - Neumann Boundary Conditions:    -σ∇u•n = F • n  (providing a vector field F for the flux) --> Applied as BoundaryNormalLFIntegrator ( F • n, v)
 *
 * - Robin Boundary Conditions:      -σ∇u•n + h(u - u0) = g
 *
 * The BCs value can be set using Coefficients, functions, constant values.
 * The BCs can be applied to specific mesh attributes or to a list of mesh attributes.
 *
 * By default the CoeffContainers take ownership of the Coefficients passed to them.
 * If this is not desired, the user can pass the Coefficients with the own flag set to false
 * (e.g. if using the same coefficient for different boundaries)
 */

#pragma once

#include <mfem.hpp>
#include "../../common-ecm2/utils.hpp"

namespace mfem
{

    // Include functions from ecm2_utils namespace
    using namespace ecm2_utils;

    namespace electrophysiology // Needed since more solvers will have a class BCHandler
    {

        class BCHandler
        {

        public:
            // Constructor
            BCHandler(ParMesh *pmesh, bool verbose = true);

            /**
             * \brief Add Dirichlet BC for potential using Coefficient and list of essential mesh attributes.
             *
             * Add a Dirichlet boundary condition for potential to internal list of essential bcs passing Coefficient
             * and list of essential mesh attributes (they will be applied at setup time).
             *
             * \param coeff Pointer to Coefficient u
             * \param coeff_dudt Pointer to Coefficient for du/dt
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddDirichletBC(Coefficient *coeff, Coefficient* coeff_dudt, Array<int> &attr, bool own = true);

            /**
             * \brief Add Dirichlet BC for potential using ScalarFuncT and list of essential mesh attributes.
             *
             * Add a Dirichlet boundary condition for potential to internal list of essential bcs passing function
             * and list of essential mesh attributes (they will be applied at setup time).
             *
             * \param func Pointer to ScalarFuncT
             * \param coeff_dudt Pointer to Coefficient for du/dt
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddDirichletBC(ScalarFuncT *func, ScalarFuncT *func_dudt, Array<int> &attr);


            /**
             * \brief Add Dirichlet BC for potential using Coefficient and specific mesh attribute.
             *
             * Add a Dirichlet boundary condition for potential to internal list of essential bcs passing Coefficient,
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param coeff Pointer to Coefficient
             * \param coeff_dudt Pointer to Coefficient for du/dt
             * \param attr Boundary attribute
             *
             */
            void AddDirichletBC(Coefficient *coeff, Coefficient *coeff_dudt, int &attr, bool own = true);

            /**
             * \brief Add Dirichlet BC for potential passing ScalarFuncT and specific mesh attribute.
             *
             * Add a Dirichlet boundary condition for potential to internal list of essential bcs passing ScalarFuncT
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param func Pointer to ScalarFuncT
             * \param func_dudt Pointer to ScalarFuncT for du/dt
             * \param attr Boundary attribute
             *
             */
            void AddDirichletBC(ScalarFuncT *func, ScalarFuncT *func_dudt, int &attr);

            /**
             * \brief Add Neumann BC using Coefficient and list of essential boundaries.
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using Coefficient and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddNeumannBC(Coefficient *coeff, Array<int> &attr, bool own = true);

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
             * using Coefficient and specific mesh attribute (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Boundary attribute
             *
             */
            void AddNeumannBC(Coefficient *coeff, int &attr, bool own = true);

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
             * using real_t and specific mesh attribute (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
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
            void AddNeumannVectorBC(VecFuncT *coeff, Array<int> &attr);

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
            void AddNeumannVectorBC(VecFuncT *func, int &attr);

            /**
             * \brief Add Robin BC using two Coefficients and list of essential boundaries.
             *
             * Add a Robin boundary condition to internal list of Robin bcs,
             * using two Coefficients and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param a Pointer to Coefficient h ()
             * \param b Pointer to Coefficient T0 (Reference potential)
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddRobinBC(Coefficient *h_coeff, Coefficient *U0_coeff, Array<int> &attr, bool own = true);

            /**
             * \brief Add Robin BC using two ScalarFuncT h (heat transfer coefficient)nd list of essential boundaries.
             *
             * Add a Robin boundary condition to internal list of Robin bcs,
             * using two ScalarFuncT h (heat transfer coefficient)nd list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param a Pointer to ScalarFuncT h (heat transfer coefficient)
             * \param b Pointer to ScalarFuncT T0 (Reference potential)
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddRobinBC(ScalarFuncT *h_func, ScalarFuncT *U0_func, Array<int> &attr);

            /**
             * \brief Add Robin BC using two Coefficients and specific mesh attribute.
             *
             * Add a Robin boundary condition to internal list of Robin bcs,
             * using two Coefficients, and specific mesh attribute (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param a Pointer to Coefficient h (heat transfer coefficent))
             * \param b Pointer to Coefficient T0 (Reference potential)
             * \param attr Boundary attribute
             *
             */
            void AddRobinBC(Coefficient *h_coeff, Coefficient *U0_coeff, int &attr, bool own = true);

            /**
             * \brief Add Robin BC using two ScalarFuncT h (heat transfer coefficient)nd specific mesh attribute.
             *
             * Add a Robin boundary condition to internal list of Robin bcs,
             * using two ScalarFuncT h (heat transfer coefficient)nd specific mesh attribute(they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param a Pointer to ScalarFuncT h (heat transfer coefficient)
             * \param b Pointer to ScalarFuncT T0 (Reference potential)
             * \param attr Boundary attribute
             *
             */
            void AddRobinBC(ScalarFuncT *h_func, ScalarFuncT *U0_func, int &attr);


            /**
             * \brief Set the time in the BCs coefficients.
             *
             * Set the time in the BCs coefficients (Dirichlet, Neumann, Robin).
             *
             * \param time Time value.
             *
             */
            void SetTime(real_t time);

            // Getters

            // Getter for potential bcs
            std::vector<MultiCoeffContainer> &GetDirichletDbcs()
            {
                return dirichlet_dbcs;
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

            // Getter for Robin_bcs
            std::vector<RobinCoeffContainer> &GetRobinBcs()
            {
                return robin_bcs;
            }

            // Getter for potential dirichlet bcs
            Array<int> &GetDirichletAttr()
            {
                return dirichlet_attr;
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

            // Getter for Robin_attr
            Array<int> &GetRobinAttr()
            {
                return robin_attr;
            }

        private:
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
             * \brief Update the time in the pressure BCs coefficients.
             *
             * Update the time in the pressure BCs coefficients.
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

            /**
             * \brief Update the time in the general Robin BCs coefficients.
             * 
             * Update the time in the general Robin BCs coefficients.
             * 
             * \param new_time New time value.
             * 
             */
            void UpdateTimeGeneralRobinBCs(real_t new_time);


            real_t time;

            // Pointer to Mesh
            ParMesh* pmesh;      //< NOT OWNED

            // Maximum number of boundary attributes
            int max_bdr_attributes;

            // Bookkeeping for Dirichlet (potential, d potential/dt) bcs.
            std::vector<MultiCoeffContainer> dirichlet_dbcs;

            // Bookkeeping for Neumann bcs.
            std::vector<CoeffContainer> neumann_bcs;

            // Bookkeeping for Neumann vector bcs.
            std::vector<VecCoeffContainer> neumann_vec_bcs;

            // Bookkeeping for Robin bcs.
            std::vector<RobinCoeffContainer> robin_bcs;

            /// Array of attributes for parsing bcs
            Array<int> dirichlet_attr;     // Essential mesh attributes.
            Array<int> neumann_attr;       // Neumann mesh attributes.
            Array<int> neumann_vec_attr;   // Neumann vector mesh attributes.
            Array<int> robin_attr;         // Robin mesh attributes.
            Array<int> dirichlet_attr_tmp; // Essential mesh attributes (temporary).
            Array<int> neumann_attr_tmp;   // Neumann mesh attributes (temporary).
            Array<int> robin_attr_tmp;     // Robin mesh attributes (temporary).

            // Verbosity
            bool verbose;
        };

    }; // namespace electrophysiology

}; // namespace mfem


