/**
 * @file pressure_correction.hpp
 * @brief File containing declarations for time adaptivity manager in Navier Stokes solver.
 */

#pragma once

#ifndef TIME_ADAPTIVITY_HPP
#define TIME_ADAPTIVITY_HPP

#include <mfem.hpp>
#include "navier_types.hpp"
#include "navier_qoi.hpp"
#include "navier_preconditioners.hpp"

namespace mfem
{
   using namespace ecm2_utils;

   namespace navier
   {

      struct DefaultTimeAdaptivityParameters_CFL
      {
         static constexpr real_t chi_min = 0.1;
         static constexpr real_t chi_max = 5;
         static constexpr real_t chi_reject = 0.5;
         static constexpr real_t chi_tol = 1e-3;
         static constexpr real_t chi_safety = 1.05; // >= 1.0
         static constexpr real_t dt_min = 1e-6;
         static constexpr real_t dt_max = 1e-1;
      
         static constexpr real_t cfl_max = 0.8;
         static constexpr real_t cfl_min = 0.1;
         static constexpr real_t cfl_tol = 1e-4;
      };


      struct DefaultTimeAdaptivityParameters_HOPC
      {
         static constexpr real_t chi_min = 0.1;
         static constexpr real_t chi_max = 5;
         static constexpr real_t chi_reject = 0.5;
         static constexpr real_t chi_tol = 1e-3;
         static constexpr real_t chi_safety = 0.9;
         static constexpr real_t dt_min = 1e-6;
         static constexpr real_t dt_max = 1e-1;
      };

      class TimeAdaptivityManager
      {
      public:
         TimeAdaptivityManager(NavierBlockPreconditioner *nsPrec_, QuantitiesOfInterest *qoi_, ParGridFunction *u_gf_);

         ~TimeAdaptivityManager() {};  

         // Set parameters
         void SetDefaultParameters(TimeAdaptivityType type);
         void SetParameters(real_t chi_min_, real_t chi_max_, real_t chi_reject_, real_t chi_tol_, real_t chi_safety_, real_t dt_min_, real_t dt_max_);
         void SetParameters_CFL(real_t cfl_max_ = 0.8, real_t cfl_min = 0.1, real_t cfl_tol_ = 1e-4);

         // Interface for time adaptivity
         bool PredictTimeStep(TimeAdaptivityType type, real_t dt_old, real_t &dt_new);

      private:
         NavierBlockPreconditioner *nsPrec = nullptr; // NOT OWNED
         QuantitiesOfInterest *qoi = nullptr;           // NOT OWNED

         ParGridFunction *u_gf = nullptr; // NOT OWNED

         // Time adaptivity parameters
         real_t chi_min = 0.1;
         real_t chi_max = 5;
         real_t chi_reject = 0.5;    
         real_t chi_tol = 1e-3;
         real_t chi_safety = 0.9;

         real_t dt_min = 1e-6;
         real_t dt_max = 1e-1;
         
         real_t cfl_min = 0.1;
         real_t cfl_max = 0.8;
         real_t cfl_tol = 1e-4;

         // Variables for computing time adaptivity
         real_t error_est;
         real_t chi;
         real_t chi_new;

         // Time adaptivity methods
         bool PredictTimeStep_CFL(real_t dt_old, real_t &dt_new);
         bool PredictTimeStep_HOPC(real_t dt_old, real_t &dt_new);
      };

   } // namespace navier
} // namespace mfem

#endif // TIME_ADAPTIVITY_HPP
