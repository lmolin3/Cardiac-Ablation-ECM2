#ifndef BCHANDLER_NS_HPP
#define BCHANDLER_NS_HPP

#include <mfem.hpp>
#include "utils.hpp"

namespace mfem
{

    // Include functions from ecm2_utils namespace
    using namespace ecm2_utils;

    namespace navier // Needed since more solvers will have a class BCHandler
    {

        class BCHandler
        {

        public:
            // Constructor
            BCHandler(std::shared_ptr<ParMesh> pmesh, bool verbose = true);

            /**
             * \brief Add Dirichlet velocity BC using VectorCoefficient and list of essential mesh attributes.
             *
             * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing VectorCoefficient
             * and list of essential mesh attributes (they will be applied at setup time).
             *
             * \param coeff Pointer to VectorCoefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr, bool own = true);

            /**
             * \brief Add Dirichlet velocity BC using Vector function and list of essential mesh attributes.
             *
             * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing Vector function
             * and list of essential mesh attributes (they will be applied at setup time).
             *
             * \param func Pointer to VecFuncT
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddVelDirichletBC(VecFuncT *func, Array<int> &attr, bool own = true);

            /**
             * \brief Add Dirichlet velocity BC componentwise using Coefficient and list of active mesh boundaries.
             *
             * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing
             * Coefficient, list of essential mesh attributes, and constrained component (they will be applied at setup time).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             * \param dir Component of bc constrained (0=x, 1=y, 2=z)
             *
             */
            void AddVelDirichletBC(Coefficient *coeff, Array<int> &attr, int &dir, bool own = true);

            /**
             * \brief Add Dirichlet velocity BC using VectorCoefficient and specific mesh attribute.
             *
             * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing VectorCoefficient,
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param coeff Pointer to VectorCoefficient
             * \param attr Boundary attribute
             *
             */
            void AddVelDirichletBC(VectorCoefficient *coeff, int &attr, bool own = true);

            /**
             * \brief Add Dirichlet velocity BC passing VecFuncT and specific mesh attribute.
             *
             * Add a Dirichlet velocity boundary condition to internal list of essential bcs passing VecFuncT
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param func Pointer to VecFuncT
             * \param attr Boundary attribute
             *
             */
            void AddVelDirichletBC(VecFuncT *func, int &attr, bool own = true);

            /**
             * \brief Add Dirichlet velocity BC componentwise passing coefficient and specific mesh attribute.
             *
             * Add a Dirichlet velocity boundary condition to internal list of essential bcs, passing
             * Coefficient, specific mesh attribute, and constrained component (they will be applied at setup time).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Boundary attribute
             * \param dir Component of bc constrained (0=x, 1=y, 2=z)
             *
             * \note dir=2 only if mesh is three dimensional.
             *
             */
            void AddVelDirichletBC(Coefficient *coeff, int &attr, int &dir);

            /**
             * \brief Add Dirichlet pressure BC using Coefficient and list of essential mesh attributes.
             *
             * Add a Dirichlet pressure boundary condition to internal list of essential bcs passing Coefficient
             * and list of essential mesh attributes (they will be applied at setup time).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddPresDirichletBC(Coefficient *coeff, Array<int> &attr, bool own = true);

            /**
             * \brief Add Dirichlet pressure BC using Scalar function and list of essential mesh attributes.
             *
             * Add a Dirichlet pressure boundary condition to internal list of essential bcs passing Vector function
             * and list of essential mesh attributes (they will be applied at setup time).
             *
             * \param func Pointer to ScalarFuncT
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddPresDirichletBC(ScalarFuncT *func, Array<int> &attr, bool own = true);

            /**
             * \brief Add Dirichlet pressure BC using Coefficient and specific mesh attribute.
             *
             * Add a Dirichlet pressure boundary condition to internal list of essential bcs passing Coefficient,
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param coeff Pointer to Coefficient
             * \param attr Boundary attribute
             *
             */
            void AddPresDirichletBC(Coefficient *coeff, int &attr, bool own = true);

            /**
             * \brief Add Dirichlet pressure BC passing ScalarFuncT and specific mesh attribute.
             *
             * Add a Dirichlet pressure boundary condition to internal list of essential bcs passing ScalarFuncT
             * and integer for specific mesh attribute (they will be applied at setup time).
             *
             * \param func Pointer to ScalarFuncT
             * \param attr Boundary attribute
             *
             */
            void AddPresDirichletBC(ScalarFuncT *func, int &attr, bool own = true);

            /**
             * \brief Add Traction (Neumann) BC using VectorCoefficient and list of essential boundaries.
             *
             * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
             * using VectorCoefficient, and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param coeff Pointer to VectorCoefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddTractionBC(VectorCoefficient *coeff, Array<int> &attr, bool own = true);

            /**
             * \brief Add Traction (Neumann) BC using VecFuncT and list of essential boundaries.
             *
             * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
             * using VecFuncT and list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param coeff Pointer to VectorCoefficient
             * \param attr Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
             *
             */
            void AddTractionBC(VecFuncT *coeff, Array<int> &attr, bool own = true);

            /**
             * \brief Add Traction (Neumann) BC using VectorCoefficient and specific mesh attribute.
             *
             * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
             * using VectorCoefficient, and specific mesh attribute (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param coeff Pointer to VectorCoefficient
             * \param attr Boundary attribute
             *
             */
            void AddTractionBC(VectorCoefficient *coeff, int &attr, bool own = true);

            /**
             * \brief Add Traction (Neumann) BC using VecFuncT and specific mesh attribute.
             *
             * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
             * using VecFuncT and specific mesh attribute(they will be applied at setup time by adding BoundaryIntegrators to the rhs).
             *
             * \param func Pointer to VecFuncT
             * \param attr Boundary attribute
             *
             */
            void AddTractionBC(VecFuncT *func, int &attr, bool own = true);


            /**
            * \brief Add Traction (Neumann) BC computed from vector field u and scalar field p, and list of essential boundaries.
            *        Neumann: (alpha n.grad(u) + beta p.n, v)
            * 
            *
            * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
            * using list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
            * The boundary contribution is computed with a VectorBoundaryLFIntegrator (psi,v), with vector computed as
            * 
            *     psi = (alpha n.grad(u) + beta p.n, v)
            *
            * \param alpha Coefficient multiplying vector field term
            * \param u     ParGridFunction for vector field
            * \param beta  Coefficient multiplying scalard field term
            * \param p     ParGridFunction for scalar field
            * \param attr  Array of boundary attributes (0 or 1=marked bdry, size of pmesh->attributes.Max())
            *
            */
            void AddCustomTractionBC(Coefficient *alpha, ParGridFunction *u, Coefficient *beta, ParGridFunction *p, Array<int> &attr, bool own = true);

            /**
            * \brief Add Traction (Neumann) BC computed from vector field u and scalar field p, and specific mesh attribute.
            *        Neumann: (alpha n.grad(u) + beta p.n, v)
            * 
            *
            * Add a Traction (Neumann) boundary condition to internal list of traction bcs,
            * using list of active mesh boundaries (they will be applied at setup time by adding BoundaryIntegrators to the rhs).
            * The boundary contribution is computed with a VectorBoundaryLFIntegrator (psi,v), with vector computed as
            * 
            *     psi = (alpha n.grad(u) + beta p.n, v)
            *
            * \param alpha Coefficient multiplying vector field term
            * \param u     ParGridFunction for vector field
            * \param beta  Coefficient multiplying scalard field term
            * \param p     ParGridFunction for scalar field
            * \param attr  Mesh attribute
            *
            */
            void AddCustomTractionBC(Coefficient *alpha, ParGridFunction *u, Coefficient *beta, ParGridFunction *p, int &attr, bool own = true);

            /**
             * \brief Update the time in the velocity BCs coefficients.
             *
             * Update the time in the velocity BCs coefficients.
             *
             * \param new_time New time value.
             *
             */
            void UpdateTimeVelocityBCs(double new_time);

            /**
             * \brief Update the time in the pressure BCs coefficients.
             * 
             * Update the time in the pressure BCs coefficients.
             * 
             * \param new_time New time value.
             * 
             */
            void UpdateTimePressureBCs(double new_time);


            /**
             * \brief Update the time in the traction BCs coefficients.
             * 
             * Update the time in the traction BCs coefficients.
             * 
             * \param new_time New time value.
             * 
             */
            void UpdateTimeTractionBCs(double new_time);

            /**
             * \brief Update the time in the Custom Traction BCs coefficients.
             * 
             * Update the time in the Custom Traction BCs coefficients.
             * 
             * \param new_time New time value.
             * 
             */
            void UpdateTimeCustomTractionBCs(double new_time);


            /**
             * \brief Check if problem is fully dirichlet (velocity).
             *        Check if all velocity dofs are constrained by Dirichlet BCs.
             */
            bool IsFullyDirichlet();

            // Getter for vel_dbcs
            std::vector<VecCoeffContainer> &GetVelDbcs() 
            {
                return vel_dbcs;
            }

            // Getter for vel_dbcs_xyz
            std::vector<CompCoeffContainer> &GetVelDbcsXYZ() 
            {
                return vel_dbcs_xyz;
            }

            // Getter for pres_dbcs
            std::vector<CoeffContainer> &GetPresDbcs() 
            {
                return pres_dbcs;
            }

            // Getter for traction_bcs
            std::vector<VecCoeffContainer> &GetTractionBcs() 
            {
                return traction_bcs;
            }

            // Getter for custom_traction_bcs
            std::vector<CustomNeumannContainer> &GetCustomTractionBcs() 
            {
                return custom_traction_bcs;
            }

            // Getter for vel_ess_attr
            Array<int> &GetVelEssAttr() 
            {
                return vel_ess_attr;
            }

            // Getter for vel_ess_attr_x
            Array<int> &GetVelEssAttrX() 
            {
                return vel_ess_attr_x;
            }

            // Getter for vel_ess_attr_y
            Array<int> &GetVelEssAttrY() 
            {
                return vel_ess_attr_y;
            }

            // Getter for vel_ess_attr_z
            Array<int> &GetVelEssAttrZ() 
            {
                return vel_ess_attr_z;
            }

            // Getter for pres_ess_attr
            Array<int> &GetPresEssAttr() 
            {
                return pres_ess_attr;
            }

            // Getter for traction_attr
            Array<int> &GetTractionAttr() 
            {
                return traction_attr;
            }

            // Getter for custom_traction_attr
            Array<int> &GetCustomTractionAttr() 
            {
                return custom_traction_attr;
            }

        private:
            // Shared pointer to Mesh
            std::shared_ptr<ParMesh> pmesh;
            int max_bdr_attributes;

            // Bookkeeping for velocity dirichlet bcs (full Vector coefficient).
            std::vector<VecCoeffContainer> vel_dbcs;

            // Bookkeeping for velocity dirichlet bcs (componentwise).
            std::string dir_string; // string for direction name for printing output
            std::vector<CompCoeffContainer> vel_dbcs_xyz;

            // Bookkeeping for pressure dirichlet bcs (scalar coefficient).
            std::vector<CoeffContainer> pres_dbcs;

            // Bookkeeping for traction (neumann) bcs.
            std::vector<VecCoeffContainer> traction_bcs;

            // Bookkeeping for custom traction (neumann) bcs.
            std::vector<CustomNeumannContainer> custom_traction_bcs;

            /// Array of attributes for bcs
            Array<int> vel_ess_attr;   // Essential mesh attributes (full velocity applied).
            Array<int> vel_ess_attr_x; // Essential mesh attributes (x component applied).
            Array<int> vel_ess_attr_y; // Essential mesh attributes (y component applied).
            Array<int> vel_ess_attr_z; // Essential mesh attributes (z component applied).
            Array<int> pres_ess_attr;  // Essential mesh attributes (pressure applied).
            Array<int> traction_attr;  // Traction mesh attributes.
            Array<int> custom_traction_attr; // Custom traction mesh attributes.
            Array<int> ess_attr_tmp;   // Temporary variable for essential mesh attributes.
            Array<int> trac_attr_tmp;  // Temporary variable for traction mesh attributes.

            // Check if problem is fully dirichlet (velocity).
            bool is_fully_dirichlet;
            bool cached_is_fully_dirichlet = false;

            // Verbosity
            bool verbose;
        };

    }; // namespace navier

}; // namespace mfem

#endif // BCHANDLER_NS_HPP