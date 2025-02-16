#include "time_adaptivity.hpp"

using namespace mfem;
using namespace navier;

///////////////////////////////////////////////////////////////////////
///                   Time Adaptivity Manager                       ///
///////////////////////////////////////////////////////////////////////

// Implementation of TimeAdaptivityManager constructor
TimeAdaptivityManager::TimeAdaptivityManager(NavierBlockPreconditioner *nsPrec_, QuantitiesOfInterest *qoi_, ParGridFunction *u_gf_) : nsPrec(nsPrec_), qoi(qoi_), u_gf(u_gf_)
{
}

// Implementation of TimeAdaptivityManager SetParameters
void TimeAdaptivityManager::SetParameters(real_t fac_min_, real_t fac_max_, real_t dt_min_, real_t dt_max_, real_t cfl_max_, real_t cfl_min, real_t cfl_tol_)
{
    fac_min = fac_min_;
    fac_max = fac_max_;
    dt_min = dt_min_;
    dt_max = dt_max_;

    cfl_max = cfl_max_;
    cfl_min = cfl_min;
    cfl_tol = cfl_tol_;
}

// Implementation of TimeAdaptivityManager PredictTimeStep
bool TimeAdaptivityManager::PredictTimeStep(TimeAdaptivityType type, real_t dt_old, real_t &dt_new)
{
    bool accept_step = true;
    switch (static_cast<TimeAdaptivityType>(type))
    {
    case TimeAdaptivityType::NONE: // Fixed time step
        dt_new = dt_old;
        break;
    case TimeAdaptivityType::CFL: // CFL-based adaptivity
    {
        accept_step = PredictTimeStep_CFL(dt_old, dt_new);
        break;
    }
    case TimeAdaptivityType::HOPC: // High Order Pressure Correction adaptivity
    {
        accept_step = PredictTimeStep_HOPC(dt_old, dt_new);
        break;
    }
    default:
        MFEM_ABORT("TimeAdaptivityManager::PredictTimeStep() >> Unknown time adaptivity type.");
        break;
    }

    return accept_step;
}

// Implementation of TimeAdaptivityManager PredictTimeStep_CFL
bool TimeAdaptivityManager::PredictTimeStep_CFL(real_t dt_old, real_t &dt_new)
{
    bool accept = false;

    // Compute CFL number
    real_t cfl = qoi->ComputeCFL(*u_gf, dt_old);
    // Define error estimator
    error_est = cfl / (cfl_max + cfl_tol);
    if (error_est < 1.0) // CFL < CFL_max
        accept = true;
    // Compute new time step
    real_t fac_safety = 2.0;
    real_t eta_new = pow(1.0 / (fac_safety * error_est), 1.0 / (1.0 + 3.0));
    real_t eta = std::min(fac_max, std::max(fac_min, eta_new));
    real_t dt_tmp = dt_old * eta;
    // Check if new time step is within bounds, otherwise limit to bounds
    dt_new = std::min(std::max(dt_tmp, dt_min), dt_max);

    return accept;
}

// Implementation of TimeAdaptivityManager PredictTimeStep_HOPC
bool TimeAdaptivityManager::PredictTimeStep_HOPC(real_t dt_old, real_t &dt_new)
{
    HOYPressureCorrectedPreconditioner *hoy_prec = dynamic_cast<HOYPressureCorrectedPreconditioner *>(nsPrec);
    if (hoy_prec == nullptr)
    {
        MFEM_ABORT("TimeAdaptivityManager::PredictTimeStep_HOPC() >> Time adaptivity type 2 requires HOYPressureCorrectedPreconditioner.");
    }

    // Placeholder for type 2 adaptivity logic
    MFEM_ABORT("TimeAdaptivityManager::PredictTimeStep_HOPC() >> Time adaptivity type 2 not implemented.");
}
