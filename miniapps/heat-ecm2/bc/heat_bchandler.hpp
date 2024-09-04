#ifndef BCHANDLER_HEAT_HPP
#define BCHANDLER_HEAT_HPP

#include <mfem.hpp>
#include "utils.hpp"

namespace mfem
{

    // Include functions from ecm2_utils namespace
    using namespace ecm2_utils;

    namespace heat // Needed since more solvers will have a class BCHandler
    {

        class BCHandler
        {

        public:
            // Constructor
            BCHandler(std::shared_ptr<ParMesh> pmesh, bool verbose = true);

            /**
             * \brief Add Dirichlet BC for electric potential using Coefficient h (heat transfer coefficent))nd list of essential mesh attributes.
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
             * \brief Add Dirichlet BC for electric potential using Vector function and list of essential mesh attributes.
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
             * \brief Add Dirichlet BC for electric potential using double and specific mesh attribute.
             *
             * Add a Dirichlet boundary condition for electric potential to internal list of essential bcs passing double,
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param coeff_val Value of the Dirichlet BC
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddDirichletBC(double coeff_val, Array<int> &attr);

            /**
             * \brief Add Dirichlet BC for electric potential using Coefficient h (heat transfer coefficent))nd specific mesh attribute.
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
             * \brief Add Dirichlet BC for electric potential passing ScalarFuncT h (heat transfer coefficient)nd specific mesh attribute.
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
             * \brief Add Dirichlet BC for electric potential using double and specific mesh attribute.
             *
             * Add a Dirichlet boundary condition for electric potential to internal list of essential bcs passing double,
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param coeff_val Value of the Dirichlet BC
             * \param attr Boundary attribute
             *
             */
            void AddDirichletBC(double coeff_val, int &attr);

            /**
             * \brief Add Neumann BC using Coefficient h (heat transfer coefficent))nd list of essential boundaries.
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
             * \brief Add Neumann BC using ScalarFuncT h (heat transfer coefficient)nd list of essential boundaries.
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using ScalarFuncT h (heat transfer coefficient)nd list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddNeumannBC(ScalarFuncT *coeff, Array<int> &attr);

            /**
             * \brief Add Neumann BC using Coefficient h (heat transfer coefficent))nd specific mesh attribute.
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
             * \brief Add Neumann BC using ScalarFuncT h (heat transfer coefficient)nd specific mesh attribute.
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using ScalarFuncT h (heat transfer coefficient)nd specific mesh attribute(they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param func Pointer to ScalarFuncT
             * \param attr Boundary attribute
             *
             */
            void AddNeumannBC(ScalarFuncT *func, int &attr);

            /**
             * \brief Add Neumann BC using double and specific mesh attribute.
             *
             * Add a Neumann boundary condition to internal list of Neumann bcs,
             * using double and specific mesh attribute(they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param val Neumann value
             * \param attr Boundary attribute
             *
             */
            void AddNeumannBC(double val, int &attr);

            /**
             * \brief Add Robin BC using two Coefficients and list of essential boundaries.
             *
             * Add a Robin boundary condition to internal list of Robin bcs,
             * using two Coefficients and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param a Pointer to Coefficient h ()
             * \param b Pointer to Coefficient T0 (Reference temperature)
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddRobinBC(Coefficient *h_coeff, Coefficient *T0_coeff, Array<int> &attr);

            /**
             * \brief Add Robin BC using two ScalarFuncT h (heat transfer coefficient)nd list of essential boundaries.
             *
             * Add a Robin boundary condition to internal list of Robin bcs,
             * using two ScalarFuncT h (heat transfer coefficient)nd list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param a Pointer to ScalarFuncT h (heat transfer coefficient)
             * \param b Pointer to ScalarFuncT T0 (Reference temperature)
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddRobinBC(ScalarFuncT *h_func, ScalarFuncT *T0_func, Array<int> &attr);

            /**
             * \brief Add Robin BC using two Coefficients and specific mesh attribute.
             *
             * Add a Robin boundary condition to internal list of Robin bcs,
             * using two Coefficients, and specific mesh attribute (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param a Pointer to Coefficient h (heat transfer coefficent))
             * \param b Pointer to Coefficient T0 (Reference temperature)
             * \param attr Boundary attribute
             *
             */
            void AddRobinBC(Coefficient *h_coeff, Coefficient *T0_coeff, int &attr);

            /**
             * \brief Add Robin BC using two ScalarFuncT h (heat transfer coefficient)nd specific mesh attribute.
             *
             * Add a Robin boundary condition to internal list of Robin bcs,
             * using two ScalarFuncT h (heat transfer coefficient)nd specific mesh attribute(they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param a Pointer to ScalarFuncT h (heat transfer coefficient)
             * \param b Pointer to ScalarFuncT T0 (Reference temperature)
             * \param attr Boundary attribute
             *
             */
            void AddRobinBC(ScalarFuncT *h_func, ScalarFuncT *T0_func, int &attr);

            /**
             * \brief Add Robin BC using two doubles and specific mesh attribute.
             *
             * Add a Robin boundary condition to internal list of Robin bcs,
             * using two doubles and specific mesh attribute(they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param val_a Robin value h (heat transfer coefficient)
             * \param val_b Robin value T0 (Reference temperature)
             * \param attr Boundary attribute
             *
             */
            void AddRobinBC(double val_h, double val_T0, int &attr);

            /**
             * \brief Set the time in the BCs coefficients.
             *
             * Set the time in the BCs coefficients (Dirichlet, Neumann, Robin).
             *
             * \param time Time value.
             *
             */
            void SetTime(double time);

            // Getters

            // Getter for electric potential bcs
            std::vector<CoeffContainer> &GetDirichletDbcs()
            {
                return dirichlet_dbcs;
            }

            // Getter for Neumann_bcs
            std::vector<CoeffContainer> &GetNeumannBcs()
            {
                return neumann_bcs;
            }

            // Getter for Robin_bcs
            std::vector<RobinCoeffContainer> &GetRobinBcs()
            {
                return robin_bcs;
            }

            // Getter for temperature dirichlet bcs
            Array<int> &GetDirichletAttr()
            {
                return dirichlet_attr;
            }

            // Getter for Neumann_attr
            Array<int> &GetNeumannAttr()
            {
                return neumann_attr;
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
            void UpdateTimeDirichletBCs(double new_time);

            /**
             * \brief Update the time in the pressure BCs coefficients.
             *
             * Update the time in the pressure BCs coefficients.
             *
             * \param new_time New time value.
             *
             */
            void UpdateTimeNeumannBCs(double new_time);

            /**
             * \brief Update the time in the Robin BCs coefficients.
             *
             * Update the time in the Robin BCs coefficients.
             *
             * \param new_time New time value.
             *
             */
            void UpdateTimeRobinBCs(double new_time);

            double time;

            // Shared pointer to Mesh
            std::shared_ptr<ParMesh> pmesh;

            // Bookkeeping for Dirichlet (temperature) bcs.
            std::vector<CoeffContainer> dirichlet_dbcs;

            // Bookkeeping for Neumann bcs.
            std::vector<CoeffContainer> neumann_bcs;

            // Bookkeeping for Robin bcs.
            std::vector<RobinCoeffContainer> robin_bcs;

            /// Array of attributes for parsing bcs
            Array<int> dirichlet_attr;     // Essential mesh attributes.
            Array<int> neumann_attr;       // Neumann mesh attributes.
            Array<int> robin_attr;         // Robin mesh attributes.
            Array<int> dirichlet_attr_tmp; // Essential mesh attributes (temporary).
            Array<int> neumann_attr_tmp;   // Neumann mesh attributes (temporary).
            Array<int> robin_attr_tmp;     // Robin mesh attributes (temporary).

            // Verbosity
            bool verbose;
        };

    }; // namespace heat

}; // namespace mfem

#endif // BCHANDLER_NS_HPP
