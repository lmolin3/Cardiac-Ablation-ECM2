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

        private:
            // Shared pointer to Mesh
            std::shared_ptr<ParMesh> pmesh;

            // Bookkeeping for velocity dirichlet bcs (full Vector coefficient).
            std::vector<CoeffContainer> dirichlet_dbcs;

            // Bookkeeping for velocity dirichlet bcs (full Vector coefficient).
            std::vector<CoeffContainer> dirichlet_EField_dbcs;

            // Bookkeeping for Neumann bcs.
            std::vector<CoeffContainer> neumann_bcs;

            /// Array of attributes for bcs
            Array<int> dirichlet_attr;         // Essential mesh attributes.
            Array<int> dirichlet_EField_attr;  // Essential mesh attributes.
            Array<int> neumann_attr;           // Neumann mesh attributes.
            Array<int> dirichlet_attr_tmp;     // Essential mesh attributes (temporary).
            Array<int> neumann_attr_tmp;       // Neumann mesh attributes (temporary).

            // Verbosity
            bool verbose;
        };

    }; // namespace electrostatics

}; // namespace mfem

#endif // BCHANDLER_NS_HPP
