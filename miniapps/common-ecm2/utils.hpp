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

            VecCoeffContainer(VecCoeffContainer &&obj) noexcept
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

            CustomNeumannContainer(CustomNeumannContainer &&obj) noexcept
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

            CoeffContainer(CoeffContainer &&obj) noexcept
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
            CompCoeffContainer(CompCoeffContainer &&obj) noexcept
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

            RobinCoeffContainer(RobinCoeffContainer &&obj) noexcept
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
                delete hT0_coeff; // Deleted regardless since it is created by RobinCoeffContainer
                hT0_coeff = nullptr;

                if (own)
                {
                    delete h_coeff;
                    delete T0_coeff;
                    h_coeff = nullptr;
                    T0_coeff = nullptr;
                }
            }

            Array<int> attr;
            Coefficient *h_coeff;
            Coefficient *T0_coeff;
            ProductCoefficient *hT0_coeff;
            bool own;
        };


        /// Container for coefficient used for General Robin bcs:
        //
        // μ1 ∇u⋅n + α1 u =  μ2 ∇u2⋅n + α2 u2 
        //
        // holding: alpha1, alpha2, mu2, grad_u2, u2
        //
        // TODO: we can add code to handle case where alpha1, alpha2, mu2 are not provided (i.e. this becomes classical Robin, or even Neumann)
        // We can set these are nullptr and check for nullptr in the specific Solver class.
        //
        class GeneralRobinContainer
        {
        public:
            GeneralRobinContainer(Array<int> attr, Coefficient *alpha1_, Coefficient *alpha2_, Coefficient *u2_, VectorCoefficient *grad_u2_, Coefficient *mu2_, bool own = true)
                : attr(attr), alpha1(alpha1_), alpha2(alpha2_), mu2(mu2_), grad_u2(grad_u2_), u2(u2_), own(own)
            {
                alpha2_u2 = new ProductCoefficient(*alpha2, *u2);
                mu2_grad_u2 = new ScalarVectorProductCoefficient(*mu2, *grad_u2);
            }

            GeneralRobinContainer(Array<int> attr, Coefficient *alpha1_, Coefficient *alpha2_, Coefficient *u2_, VectorCoefficient *mu2_grad_u2_, bool own = true)
                : attr(attr), alpha1(alpha1_), alpha2(alpha2_), grad_u2(mu2_grad_u2_), mu2(nullptr), u2(u2_), own(own)
            {
                alpha2_u2 = new ProductCoefficient(*alpha2, *u2);
                mu2_grad_u2 = new ScalarVectorProductCoefficient(1.0, *grad_u2);
            }       

            GeneralRobinContainer(GeneralRobinContainer &&obj) noexcept 
            {
                // Deep copy the attribute and direction
                this->attr = obj.attr;
                this->own = obj.own;

                // Move the coefficient pointer
                this->alpha1 = obj.alpha1;
                this->alpha2 = obj.alpha2;
                this->mu2 = obj.mu2;
                this->grad_u2 = obj.grad_u2;
                this->u2 = obj.u2;
                this->alpha2_u2 = obj.alpha2_u2;
                this->mu2_grad_u2 = obj.mu2_grad_u2;
                obj.alpha1 = nullptr;
                obj.alpha2 = nullptr;
                obj.mu2 = nullptr;
                obj.grad_u2 = nullptr;
                obj.u2 = nullptr;
                obj.alpha2_u2 = nullptr;
                obj.mu2_grad_u2 = nullptr;
            }

            ~GeneralRobinContainer()
            {
                delete alpha2_u2; // Deleted regardless since it is created by GeneralRobinContainer
                alpha2_u2 = nullptr;
                delete mu2_grad_u2; // Deleted regardless since it is created by GeneralRobinContainer
                mu2_grad_u2 = nullptr;
                
                if (own)
                {
                    delete alpha1;
                    delete alpha2;
                    delete grad_u2;
                    delete mu2;
                    delete u2;
                    alpha1 = nullptr;
                    alpha2 = nullptr;
                    mu2 = nullptr;
                    grad_u2 = nullptr;
                    u2 = nullptr;
                }
            }

                Array<int> attr;
                Coefficient *alpha1;                                   // May be OWNED
                Coefficient *alpha2;                                   // May be OWNED
                Coefficient *mu2;                                      // May be OWNED
                VectorCoefficient *grad_u2;                            // May be OWNED
                Coefficient *u2;                                       // May be OWNED
                ScalarVectorProductCoefficient *mu2_grad_u2 = nullptr; // OWNED
                Coefficient *alpha2_u2 = nullptr;                      // OWNED
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

        // Logical and operation between two arrays
        Array<int> operator&&(const Array<int> &a, const Array<int> &b);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                              Mesh utils                                              ///
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        void ExportMeshwithPartitioning(const std::string &outfolder, Mesh &mesh, const int *partitioning_);

    } // namespace ecm2_utils

} // namespace mfem

#endif