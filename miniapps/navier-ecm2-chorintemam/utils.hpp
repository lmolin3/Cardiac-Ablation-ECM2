#include "mfem.hpp"

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

    // Struct to pass slver parameters
    struct SolverParams
    {
        double rtol = 1e-6;
        double atol = 1e-10;
        int maxIter = 1000;
        int      pl = 0;

        SolverParams(double rtol_ = 1e-6, double atol_ = 1e-10, int maxIter_ = 1000, int pl_ = 0)
            : rtol(rtol_), atol(atol_), maxIter(maxIter_), pl(pl_) {}
    };

    /// Container for vector coefficient holding: coeff and mesh attribute (useful for BCs and forcing terms).
    class VecCoeffContainer
    {
    public:
        VecCoeffContainer(Array<int> attr, VectorCoefficient *coeff_)
            : attr(attr)
        {
            this->coeff = coeff_;
        }

        VecCoeffContainer(VecCoeffContainer &&obj)
        {
            // Deep copy the attribute array
            this->attr = obj.attr;

            // Move the coefficient pointer
            this->coeff = obj.coeff;
            obj.coeff = nullptr;
        }

        ~VecCoeffContainer()
        {
            coeff=nullptr;
        }

        Array<int> attr;
        VectorCoefficient *coeff = nullptr;
    };

    /// Container for vector coefficient holding: coeff and mesh attribute (useful for BCs and forcing terms).
    class CustomNeumannContainer
    {
    public:
        CustomNeumannContainer(Array<int> attr, Coefficient *alpha_, ParGridFunction *u_, Coefficient *beta_, ParGridFunction *p_)
            : attr(attr)
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

            // Move the coefficient pointer
            this->u = obj.u;
            this->p = obj.p;
            this->alpha = obj.alpha;
            this->beta = obj.beta;
            obj.u = nullptr; obj.p = nullptr;
            obj.alpha = nullptr; obj.beta = nullptr;
        }

        ~CustomNeumannContainer()
        {
            u=nullptr;
            p=nullptr;
            alpha=nullptr;
            beta=nullptr;
        }

        Array<int> attr;
        ParGridFunction *u = nullptr;
        ParGridFunction *p = nullptr;
        Coefficient *alpha = nullptr;
        Coefficient *beta = nullptr;
    };

    /// Container for coefficient holding: coeff, mesh attribute id (i.e. not the full array)
    class CoeffContainer
    {
    public:
        CoeffContainer(Array<int> attr, Coefficient *coeff)
            : attr(attr), coeff(coeff)
        {}

        CoeffContainer(CoeffContainer &&obj)
        {
            // Deep copy the attribute and direction
            this->attr = obj.attr;

            // Move the coefficient pointer
            this->coeff = obj.coeff;
            obj.coeff = nullptr;
        }

        ~CoeffContainer()
        {
            delete coeff;
            coeff=nullptr;
        }

        Array<int> attr;
        Coefficient *coeff;
    };

    /// Container for componentwise coefficient holding: coeff, mesh attribute id (i.e. not the full array) and direction (x,y,z) (useful for componentwise BCs).
    class CompCoeffContainer : public CoeffContainer
    {
    public:
        // Constructor for CompCoeffContainer
        CompCoeffContainer(Array<int> attr, Coefficient *coeff, int dir)
            : CoeffContainer(attr, coeff), dir(dir)
        {}

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

    /** @brief Matrix vector multiplication with the original uneliminated
       matrix.  The original matrix is \f$ mat + mat_e \f$ so we have:
       \f$ y = mat x + mat_e x \f$ */
    void FullMult(HypreParMatrix* mat, HypreParMatrix* mat_e, Vector &x, Vector &y);

    /** @brief Addition of matrix vector multiplication with the original uneliminated
       matrix.  The original matrix is \f$ mat + mat_e \f$ so we have:
       \f$ y += a ( mat x + mat_e x ) \f$ */
    void FullAddMult(HypreParMatrix* mat, HypreParMatrix* mat_e, Vector &x, Vector &y, double a = 1.0);


    // Class containing computation of potential quantities of interest
    class QuantitiesOfInterest
    {
    public:
    QuantitiesOfInterest(ParMesh *pmesh);

    ~QuantitiesOfInterest() { delete mass_lf; };

    // Computes K = 0.5 * int_{Omega} |u|^2
    double ComputeKineticEnergy(ParGridFunction &v);
    
    // Compute CFL = dt u / h
    double ComputeCFL(ParGridFunction &u, double dt);


    private:
    ConstantCoefficient onecoeff;
    ParLinearForm *mass_lf;
    double volume;
    };



}

}


#endif