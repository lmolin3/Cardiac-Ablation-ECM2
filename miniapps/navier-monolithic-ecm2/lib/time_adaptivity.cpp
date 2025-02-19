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

// Implementation of TimeAdaptivityManager SetDefaultParameters
void TimeAdaptivityManager::SetDefaultParameters(TimeAdaptivityType type)
{
    switch (static_cast<TimeAdaptivityType>(type))
    {
    case TimeAdaptivityType::NONE:
        break;
    case TimeAdaptivityType::CFL:
        SetParameters(DefaultTimeAdaptivityParameters_CFL::chi_min, DefaultTimeAdaptivityParameters_CFL::chi_max, DefaultTimeAdaptivityParameters_CFL::chi_reject, DefaultTimeAdaptivityParameters_CFL::chi_tol, DefaultTimeAdaptivityParameters_CFL::chi_safety, DefaultTimeAdaptivityParameters_CFL::dt_min, DefaultTimeAdaptivityParameters_CFL::dt_max);
        SetParameters_CFL(DefaultTimeAdaptivityParameters_CFL::cfl_max, DefaultTimeAdaptivityParameters_CFL::cfl_min, DefaultTimeAdaptivityParameters_CFL::cfl_tol);
        break;
    case TimeAdaptivityType::HOPC:
        SetParameters(DefaultTimeAdaptivityParameters_HOPC::chi_min, DefaultTimeAdaptivityParameters_HOPC::chi_max, DefaultTimeAdaptivityParameters_HOPC::chi_reject, DefaultTimeAdaptivityParameters_HOPC::chi_tol, DefaultTimeAdaptivityParameters_HOPC::chi_safety, DefaultTimeAdaptivityParameters_HOPC::dt_min, DefaultTimeAdaptivityParameters_HOPC::dt_max);
        break;
    default:
        MFEM_ABORT("TimeAdaptivityManager::SetDefaultParameters() >> Unknown time adaptivity type.");
        break;
    }
}

// Implementation of TimeAdaptivityManager SetParameters
void TimeAdaptivityManager::SetParameters(real_t chi_min_, real_t chi_max_, real_t chi_reject_, real_t chi_tol_, real_t chi_safety_, real_t dt_min_, real_t dt_max_)
{
    chi_min = chi_min_;
    chi_max = chi_max_;
    chi_reject = chi_reject_;
    chi_tol = chi_tol_;
    chi_safety = chi_safety_;
    dt_min = dt_min_;
    dt_max = dt_max_;
}

// Implementation of TimeAdaptivityManager SetParameters_CFL
void TimeAdaptivityManager::SetParameters_CFL(real_t cfl_max_, real_t cfl_min, real_t cfl_tol_)
{
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
    bool accept = true;

    // Compute CFL number
    real_t cfl = qoi->ComputeCFL(*u_gf, dt_old);

    // Define error estimator
    error_est = cfl / (cfl_max + cfl_tol);

    // Compute new factor
    chi_new = pow(1.0 / (chi_safety * error_est), 1.0 / (1.0 + 3.0));
    chi = std::min(chi_max, std::max(chi_min, chi_new));

    // Compute new time step
    dt_new = std::min(std::max(chi * dt_old, dt_min), dt_max);

    // Check if time step is accepted
    if (error_est >= 1.0 || chi <= chi_reject ) // CFL >= CFL_max   or   chi <= chi_reject
        accept = false;

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

    bool accept = true;

    // Retrieve final pressure correction
    int q = hoy_prec->GetPressureCorrectionOrder();
    const Vector &zq = hoy_prec->GetLastPressureCorrection();

    // Compute error estimate
    error_est = zq.Norml2();  // Norml2, Normlinf, Norml1, Normlp
   
    // Compute new factor
    chi_new = pow( (chi_tol * dt_old)/error_est, 1.0 / q );
    chi = std::min(chi_max, std::max(chi_min, chi_new));

    // Compute new time step
    dt_new = std::min(std::max(chi * dt_old, dt_min), dt_max);

    // Check if time step is accepted
    if ( chi <= chi_reject ) // chi <= chi_reject
        accept = false;

    return accept;
}
