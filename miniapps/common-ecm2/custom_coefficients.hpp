#pragma once

#include "mfem.hpp"
#include <functional>

namespace mfem
{

  namespace common_ecm2
  {
    /** @brief Coefficient representing the perfusion term in the bioheat equation.
    //
    // The perfusion term is given by:
    //
    //    Q_p = rho_b * c_b * w_b(Ω) * (T - T_blood,core)
    //
    // where:
    // - rho_b: density of blood (kg/m^3)
    // - c_b: specific heat capacity of blood (J/kgK)
    // - w_b: **damage (Ω) dependent** perfusion rate (1/s).
    //        Can be constant (i.e. damage-independent), piecewise, or nonlinear
    // - T: local tissue temperature (K)
    // - T_blood,core: core blood temperature (K)
    //
    // If the user provides a GridFunction for Ω and type, then the perfusion rate
    // w_b is damage-dependent; otherwise, it is constant.
    //
    // Default values for perfusion parameters, taken from:
    //
    // [1] Zhu, L. Heat Transfer Applications in Biological Systems. in Biomedical Engineering
    // and Design Handbook, Volume 1 233–267 (McGraw-Hill Education, 2009).
    //
    // [2] Nickander, J. et al. The relative contributions of myocardial perfusion, blood
    // volume and extracellular volume to native T1 and native T2 at rest and during
    // adenosine stress in normal physiology. Journal of Cardiovascular Magnetic
    // Resonance 21 (2019).
    */

    enum class PerfusionRateType : int
    {
      CONSTANT = 0,
      PIECEWISE = 1,
      NONLINEAR = 2
    };

    class PerfusionCoefficient : public Coefficient
    {
    protected:
      real_t w_b; //< Perfusion rate (1/s)

      //<--- Blood properties and core temperature
      GridFunction *T_gf = nullptr;  //< Temperature GridFunction, NOT OWNED
      real_t rho_b;                  //< Density of blood
      real_t c_b;                    //< Specific heat capacity of blood
      real_t T_blood_core;           //< Core blood temperature
      real_t w_b_baseline = 0.0371;  //< Baseline perfusion rate
      real_t correction_coeff = 0.8; //< Correction term (includes effects like venous rewarming)
      real_t alpha;                  //< Pre-computed term: rho_b * c_b * correction_coeff

      //<--- Optional damage (Ω) GridFunction and related parameters
      GridFunction *Damage_gf = nullptr; //< Damage (Ω) gf, NOT OWNED
      real_t damage_threshold;           //< Damage threshold for damage-dependent perfusion

      // Generalized perfusion rate function
      std::function<real_t(real_t)> w_b_func;

      PerfusionRateType rate_type;

    public:
      // Constructor for constant perfusion rate
      PerfusionCoefficient(GridFunction *T_gf_,
                           real_t rho_b_,
                           real_t c_b_,
                           real_t T_blood_core_ = 310.15,
                           real_t w_b_baseline_ = 0.0371,
                           real_t correction_coeff_ = 0.8)
          : w_b(0.0),
            T_gf(T_gf_),
            rho_b(rho_b_),
            c_b(c_b_),
            T_blood_core(T_blood_core_),
            w_b_baseline(w_b_baseline_),
            correction_coeff(correction_coeff_),
            alpha(0.0),
            Damage_gf(nullptr),
            damage_threshold(0.0),
            w_b_func(),
            rate_type(PerfusionRateType::CONSTANT)
      {
        MFEM_ASSERT(T_gf != nullptr, "Temperature GridFunction must be provided for PerfusionCoefficient.");
        alpha = rho_b * c_b * correction_coeff;
        w_b_func = [this](real_t /*damage*/)
        { return w_b_baseline; };
      }

      // Constructor for damage-dependent perfusion rate
      PerfusionCoefficient(GridFunction *T_gf_,
                           real_t rho_b_,
                           real_t c_b_,
                           GridFunction *Damage_gf_,
                           PerfusionRateType rate_type_,
                           real_t T_blood_core_ = 310.15,
                           real_t w_b_baseline_ = 0.0371,
                           real_t correction_coeff_ = 0.8)
          : w_b(0.0),
            T_gf(T_gf_),
            rho_b(rho_b_),
            c_b(c_b_),
            T_blood_core(T_blood_core_),
            w_b_baseline(w_b_baseline_),
            correction_coeff(correction_coeff_),
            alpha(0.0),
            Damage_gf(Damage_gf_),
            damage_threshold(0.0),
            w_b_func(),
            rate_type(rate_type_)
      {
        MFEM_ASSERT(T_gf != nullptr, "Temperature GridFunction must be provided for PerfusionCoefficient.");
        MFEM_ASSERT(Damage_gf != nullptr, "Damage GridFunction must be provided for damage-dependent perfusion.");
        alpha = rho_b * c_b * correction_coeff;
        damage_threshold = rate_type == PerfusionRateType::PIECEWISE ? 0.2 : 0.1;

        switch (rate_type)
        {
        case PerfusionRateType::CONSTANT:
          w_b_func = [this](real_t /*damage*/)
          { return w_b_baseline; };
          break;
        case PerfusionRateType::PIECEWISE:
          w_b_func = [this](real_t damage)
          {
            return (damage > damage_threshold) ? 0.0 : w_b_baseline;
          };
          break;
        case PerfusionRateType::NONLINEAR:
          w_b_func = [this](real_t damage)
          {
            // Exponent 4.6 gives ~1% of baseline at damage = 1.0
            // This is consistent with the original model which used damage
            // from the Arrhenius equation (Ω = 4.6 corresponds to ~1% viability)
            if (damage <= 0.0)
              return w_b_baseline;
            else if (damage < damage_threshold)
              return w_b_baseline * (1.0 + 25.0 * damage - 260.0 * damage * damage);
            else
              return w_b_baseline * exp(-4.6*damage);
          };
          break;
        }
      }

      // Perfusion term
      // > 0 if T < T_blood_core (heat source)
      // < 0 if T > T_blood_core (heat sink)
      real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
      {
        real_t T_local = T_gf->GetValue(T, ip);
        real_t damage = Damage_gf ? Damage_gf->GetValue(T, ip) : 0.0;
        w_b = w_b_func(damage);
        return -alpha * w_b * (T_local - T_blood_core);
      }
    };

    /** @brief Generic Temperature-Dependent Coefficient
     *  Useful for defining material properties (e.g., thermal conductivity, specific heat, electrical conductivity) that vary with temperature.
     *
     * The user provides a function that defines how the coefficient varies with temperature.
     *
     * Example usage:
     *
     * ```cpp
     * auto thermal_conductivity_func = [](real_t T) {
     *     if (T < 310.15) // Below 37C
     *         return 0.5; // W/mK
     *     else
     *         return 0.4; // W/mK
     * };
     *
     * TemperatureDependentCoefficient k_coeff(&T_gf, thermal_conductivity_func);
     * ```
     */
    class TemperatureDependentCoefficient : public Coefficient
    {
    protected:
      GridFunction *T_gf;                       //< Temperature GridFunction, NOT OWNED
      std::function<real_t(real_t)> coeff_func; //< User-defined function for temperature dependence
    public:
      TemperatureDependentCoefficient(GridFunction *T_gf_,
                                      std::function<real_t(real_t)> coeff_func_)
          : T_gf(T_gf_), coeff_func(coeff_func_)
      {
        MFEM_ASSERT(T_gf != nullptr, "Temperature GridFunction must be provided for TemperatureDependentCoefficient.");
      }

      real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
      {
        real_t T_local = T_gf->GetValue(T, ip);
        return coeff_func(T_local);
      }
    };

  } // namespace heat
} // namespace mfem