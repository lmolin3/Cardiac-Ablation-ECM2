#include "reaction_solver.hpp"
#include "general/forall.hpp"
#include <iostream>
#include <cmath>

using namespace mfem;
using namespace mfem::electrophysiology;

namespace
{

// The scheme dispatch itself now lives in gotranx_wrapper.hpp as
// ApplyOdeScheme<K>(), shared with the contraction kernels.

// One ODE substep for every dof: read the state, override the stimulation amplitude,
// integrate, and write the new state back in place.
//
// PerDofParams=false reads NP uniform values (a broadcast load from cache); true reads
// the full NP x n array. Templated rather than a runtime flag so neither carries a branch.
template <typename IonicModelKernel, bool PerDofParams>
void ReactionSubstep(int n, int scheme, real_t t, real_t dt, real_t vrange,
                     const real_t *d_stim, const real_t *d_chi, const real_t *d_Cm,
                     real_t *d_values, const real_t *d_params)
{
   constexpr int NS = IonicModelKernel::nstates;
   constexpr int NP = IonicModelKernel::nparams;

   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      real_t s[NS], v[NS], p[NP];

      for (int k = 0; k < NS; k++) { s[k] = d_values[k * n + i]; v[k] = s[k]; }
      if (PerDofParams)
      {
         for (int k = 0; k < NP; k++) { p[k] = d_params[k * n + i]; }
      }
      else
      {
         for (int k = 0; k < NP; k++) { p[k] = d_params[k]; }
      }

      const real_t Jscaling = IonicModelKernel::dimensionless
                              ? IonicModelKernel::stim_sign * (d_chi[i] * d_Cm[i] * vrange)
                              : IonicModelKernel::stim_sign;
      p[IonicModelKernel::stim_ampl_idx] = IonicModelKernel::dimensionless ? d_stim[i] / Jscaling : d_stim[i];

      ApplyOdeScheme<IonicModelKernel>(scheme, s, t, dt, p, v);

      // Only the updated state is stored: ApplyScheme takes `s` by const pointer, so
      // writing it back to a second array would just record the previous substep.
      for (int k = 0; k < NS; k++) { d_values[k * n + i] = v[k]; }
   });
}

// Dispatch on the parameter layout, keeping the model switch in one place.
template <typename IonicModelKernel>
static inline void ReactionSubstepDispatch(
   bool per_dof_params, int n, int scheme, real_t t, real_t dt, real_t vrange,
   const real_t *d_stim, const real_t *d_chi, const real_t *d_Cm,
   real_t *d_values, const real_t *d_params)
{
   if (per_dof_params)
   {
      ReactionSubstep<IonicModelKernel, true>(
         n, scheme, t, dt, vrange, d_stim, d_chi, d_Cm, d_values, d_params);
   }
   else
   {
      ReactionSubstep<IonicModelKernel, false>(
         n, scheme, t, dt, vrange, d_stim, d_chi, d_Cm, d_values, d_params);
   }
}

// Temperature/damage dependent parameter update. gamma/eta/Q and the damaged time
// constants are recomputed per dof before the substep loop. `d_damage` holds the
// damage function f(D) already evaluated on host (it is a std::function, so it
// cannot be called from device code).
void UpdateThermalDamageParams(int n, int nparams,
                               const real_t *d_temperature, const real_t *d_damage,
                               bool have_temperature, bool have_damage,
                               real_t A, real_t B, real_t Tref, real_t Q10,
                               int eta_idx, int gamma_idx, int Q_idx,
                               int ntau, const int *d_tau_idx,
                               const real_t *d_healthy_tau, const real_t *d_delta_tau,
                               real_t *d_params)
{
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      const real_t temperature = have_temperature ? d_temperature[i] : Tref;
      const real_t damage = have_damage ? d_damage[i] : 0.0;

      // Damage effect on ionic currents: gamma = (1 - f(D))
      d_params[gamma_idx * n + i] = 1.0 - damage;

      // Moore term: eta = A * (1 + B * (T - Tref))
      const real_t dT = temperature - Tref;
      d_params[eta_idx * n + i] = A * (1.0 + B * dT);

      // Q10 power-law scaling of gating kinetics, tanh low-pass filtered above
      // Tref + 10 K where the Q10 law stops being valid.
      const real_t T_cut = Tref + 10.0;
      const real_t deltaT_tanh = 2.0;
      const real_t Q_pow = pow(Q10, dT / 10.0);
      const real_t S2 = 0.5 * (1.0 - tanh((temperature - T_cut) / deltaT_tanh));
      d_params[Q_idx * n + i] = Q_pow * S2 + 1.0 * (1.0 - S2);

      // Time constants for damage only: tau_i = tau_healthy_i * (1 + delta_tau_i * G)
      for (int k = 0; k < ntau; k++)
      {
         d_params[d_tau_idx[k] * n + i] = d_healthy_tau[k] * (1.0 + d_delta_tau[k] * damage);
      }
   });
}

} // anonymous namespace

ReactionSolver::ReactionSolver(ParFiniteElementSpace *fes_, Coefficient *chi_coeff_, Coefficient *Cm_coeff_, IonicModelType model_type_, TimeIntegrationScheme scheme_type, int ode_substeps_)
    : fes(fes_), fes_truevsize(fes_->GetTrueVSize()), model_type(model_type_), scheme(scheme_type), ode_substeps(ode_substeps_), chi_coeff(chi_coeff_), Cm_coeff(Cm_coeff_)
{
    //<--- Initialize the ionic model based on the selected type
    switch (model_type)
    {
    case IonicModelType::MITCHELL_SCHAEFFER:
        model = std::make_unique<MitchellSchaeffer>();
        break;
    case IonicModelType::FENTON_KARMA:
        model = std::make_unique<FentonKarma>();
        break;
    case IonicModelType::TENTUSSCHER_PANFILOV_EPI:
        model = std::make_unique<TP06Epi>();
        break;
    case IonicModelType::TENTUSSCHER_PANFILOV_ENDO:
        model = std::make_unique<TP06Endo>();
        break;
    case IonicModelType::MITCHELL_SCHAEFFER_TD_DEPENDENT:
    {
        model = std::make_unique<MitchellSchaefferTD>();
        int num_time_constants = dynamic_cast<GotranxODEModelWithThermalDamage*>(model.get())->GetTimeConstantsIdxs().size();
        td_delta_tau.resize(num_time_constants, 0.0);
        td_healthy_tau.resize(num_time_constants, 0.0);
    }
    break;
    default:
        mfem_error("Unsupported ionic model type");
    }

    // Check if the model has temperature/damage dependency
    has_td_dependency = dynamic_cast<GotranxODEModelWithThermalDamage*>(model.get()) != nullptr; 

    //<--- Setup grid functions and vectors for stimulation, chi, and Cm
    // Stimulation
    stimulation_gf.SetSpace(fes);
    stimulation_gf = 0.0;
    stimulation_gf.GetTrueDofs(stimulation_vec);
    // These three are read by the reaction kernel every substep; keep them on device.
    stimulation_vec.UseDevice(true);

    // Chi and Cm
    chi_gf.SetSpace(fes);
    chi_gf = 0.0;
    chi_gf.GetTrueDofs(chi_vec);

    Cm_gf.SetSpace(fes);
    Cm_gf = 0.0;
    Cm_gf.GetTrueDofs(Cm_vec);

    // Project chi and Cm coefficients once per time step
    // Outside the step, because FOR NOW we assume they are time-independent (potentially heterogeneous)
    chi_gf.ProjectCoefficient(*chi_coeff);
    chi_gf.GetTrueDofs(chi_vec);
    chi_vec.UseDevice(true);
    Cm_gf.ProjectCoefficient(*Cm_coeff);
    Cm_gf.GetTrueDofs(Cm_vec);
    Cm_vec.UseDevice(true);

    //<--- Setup grid functions and vectors for states (except potential)
    int num_states = model->GetNumStates();
    int num_non_potential_states = num_states - 1;

    // Ensure states_gfs is large enough
    if (static_cast<int>(states_gfs.size()) < num_non_potential_states)
    {
        states_gfs.resize(num_non_potential_states, nullptr);
        states_vectors.resize(num_non_potential_states, nullptr);
    }

    // Register each state variable, apart from potential
    for (int j = 0; j < num_states; j++)
    {
        if (j == model->potential_idx)
            continue; // Skip potential

        int adjusted_index = j;
        if (j > model->potential_idx)
        {
            adjusted_index -= 1;
        }

        // Create grid function and vector
        ParGridFunction *state_gf = new ParGridFunction(fes);
        Vector *state_vec = new Vector(fes_truevsize);
        states_gfs[adjusted_index] = state_gf;
        states_vectors[adjusted_index] = state_vec;
    }
}

ReactionSolver::~ReactionSolver()
{
    for (auto gf : states_gfs)
    {
        delete gf;
    }
    for (auto vec : states_vectors)
    {
        delete vec;
    }
}


void ReactionSolver::SetThermalParameters(
    ParGridFunction *temperature_gf_,
    real_t A,
    real_t B,
    real_t T_ref,
    real_t Q10)
{
    if (!has_td_dependency && Mpi::Root())
    {
        mfem_warning("ReactionSolver::SetThermalParameters(): The current ionic model does not support temperature dependency. The provided grid function will be ignored.");
    }

    // NOTE: here we assume that the gf is defined on the same fes as the EP solver
    temperature_gf = temperature_gf_;

    // Store parameters for later use
    td_A = A;
    td_B = B;
    td_Tref = T_ref;
    td_Q10 = Q10;
}


void ReactionSolver::SetDamageParameters(
    ParGridFunction *damage_gf_,
    std::function<real_t(real_t)> damage_func,
    std::vector<real_t> delta_tau)
{
    if (!has_td_dependency && Mpi::Root())
    {
        mfem_warning("ReactionSolver::SetTemperatureAndDamageGridFunctions(): The current ionic model does not support temperature/damage dependency. The provided grid functions will be ignored.");
    }

    // NOTE: here we assume that the gfs are defined on the same fes as the EP solver
    damage_gf = damage_gf_;

    // Store parameters for later use
    td_damage_func = damage_func != nullptr ? damage_func : [](real_t D) { return 1.0; }; // Default: identity function

    // For delta_tau, ensure size matches number of time constants
    int num_time_constants = dynamic_cast<GotranxODEModelWithThermalDamage*>(model.get())->GetTimeConstantsIdxs().size();
    if (static_cast<int>(delta_tau.size()) != num_time_constants)
    {
        mfem_error("ReactionSolver::SetThermalDamageParameters(): Size of delta_tau does not match number of time constants in the model.");
    }
    td_delta_tau = delta_tau;
}

void ReactionSolver::Setup(const std::vector<double> &initial_states, const std::vector<double> &params)
{
    // Check if the model has temperature/damage dependency and issue a warning if gfs are not provided
    if (has_td_dependency)
    {
        if ((temperature_gf == nullptr && damage_gf == nullptr) && Mpi::Root())
        {
            mfem_warning("ReactionSolver::Setup(): The selected ionic model has temperature/damage dependency, "
                         "but no temperature or damage grid functions were provided. "
                         "This is equivalent to having no dependency.");
        }
    }

    // Compute variables for conversion to/from dimensionless potential.
    // The range belongs to the model -- a dimensionless model defines its own
    // affine map, and a physiological one needs a clamp wide enough to leave its
    // action potential alone -- so take it from there unless the caller has
    // explicitly overridden it with SetVRange().
    if (!vrange_user_set)
    {
        Vmin = model->Vmin_default;
        Vmax = model->Vmax_default;
    }
    Vrange = Vmax - Vmin;
    invVrange = 1.0 / Vrange;

    // Get number of states and parameters from the model
    const int num_states = model->GetNumStates();
    const int num_param = model->GetNumParameters();
    const int n = fes_truevsize;

    // Pre-allocate the flat SoA arrays: entry k of dof i lives at [k*n + i].
    // Read()/Write()/ReadWrite() would set the device flag on first use anyway, but
    // setting it here means the initial allocation already has a device backing and
    // any vector algebra on these takes the device path from the start.
    values.SetSize(num_states * n);     values.UseDevice(true);

    // Either the caller asked for per-dof parameters, or the model writes them itself
    // (thermal/damage). Must not clobber the caller's request.
    per_dof_params = per_dof_params || has_td_dependency;
    uniform_params.SetSize(num_param);  uniform_params.UseDevice(true);
    if (per_dof_params)
    {
        parameters.SetSize(num_param * n);  parameters.UseDevice(true);
    }
    else
    {
        parameters.SetSize(0);
    }

    // Get default values once to avoid repeated function calls
    std::vector<double> default_states(num_states);
    std::vector<double> default_params(num_param);

    bool use_provided_states = !initial_states.empty() && initial_states.size() == num_states;
    bool use_provided_params = !params.empty() && params.size() == num_param;

    if (!use_provided_states) {
        model->init_state_values(default_states.data());
    }

    // Initialize default parameters (will be used later in case Update is called)
    // For values/states we hold them in ParGridFunctions/Vectors anyway
    parameters_default.resize(num_param);
    if (!use_provided_params) {
        model->init_parameter_values(default_params.data());
        std::copy(default_params.begin(), default_params.end(), parameters_default.begin());
    }
    else {
        std::copy(params.begin(), params.end(), parameters_default.begin());
    }

    // Every dof starts from the same states and parameters, so fill on host once
    // (this is setup, not a hot path) and let the memory manager move it to device
    // on first kernel launch.
    const std::vector<double> &s0 = use_provided_states ? initial_states : default_states;

    real_t *h_values = values.HostWrite();
    for (int k = 0; k < num_states; k++)
    {
        for (int i = 0; i < n; i++) { h_values[k * n + i] = s0[k]; }
    }

    real_t *h_uparams = uniform_params.HostWrite();
    for (int k = 0; k < num_param; k++) { h_uparams[k] = parameters_default[k]; }

    if (per_dof_params)
    {
        real_t *h_params = parameters.HostWrite();
        for (int k = 0; k < num_param; k++)
        {
            for (int i = 0; i < n; i++) { h_params[k * n + i] = parameters_default[k]; }
        }
    }

    // Store time constants for undamaged tissue (use provided params or default)
    if (has_td_dependency)
    {
        auto time_constants_idxs = dynamic_cast<GotranxODEModelWithThermalDamage*>(model.get())->GetTimeConstantsIdxs();
        for (size_t idx = 0; idx < time_constants_idxs.size(); idx++)
        {
            int param_idx = time_constants_idxs[idx];
            td_healthy_tau[idx] = parameters_default[param_idx];
        }
    }

    // Initialize stimulation vector once, outside the loop
    stimulation_vec = parameters_default[model->stim_ampl_idx];

    //<--- Contraction state, only when a contraction model has been registered
    if (contraction_model)
    {
        const int num_states_c = contraction_model->GetNumStates();
        const int num_param_c = contraction_model->GetNumParameters();

        contraction_values.SetSize(num_states_c * n); contraction_values.UseDevice(true);
        contraction_params.SetSize(num_param_c);      contraction_params.UseDevice(true);
        Ta_tvector_.SetSize(n);                       Ta_tvector_.UseDevice(true);
        Ta_tvector_ = 0.0;

        std::vector<real_t> c_states(num_states_c), c_params(num_param_c);
        contraction_model->init_state_values(c_states.data());
        contraction_model->init_parameter_values(c_params.data());

        real_t *h_cv = contraction_values.HostWrite();
        for (int k = 0; k < num_states_c; k++)
        {
            for (int i = 0; i < n; i++) { h_cv[k * n + i] = c_states[k]; }
        }

        real_t *h_cp = contraction_params.HostWrite();
        for (int k = 0; k < num_param_c; k++) { h_cp[k] = c_params[k]; }

        // Nothing has been integrated yet, so the zero-initialised tension above
        // is already the correct answer for t = 0.
        tension_updated_ = true;
    }
}


void ReactionSolver::RegisterModels(std::unique_ptr<EPModelBase> ep_model,
                                    std::unique_ptr<ContractionModelBase> contraction_model_)
{
    MFEM_VERIFY(ep_model != nullptr, "ReactionSolver::RegisterModels(): ep_model must not be null.");

    //<--- Verification: fail fast on an incompatible pairing.
    if (contraction_model_ && contraction_model_->RequiresCalcium() && !ep_model->HasCalcium())
    {
        throw std::runtime_error(
            "Incompatible pairing: ContractionModel requires [Ca2+]i, but EPModel does not provide it. "
            "(contraction model '" + contraction_model_->GetName() +
            "', EP model '" + ep_model->GetName() + "')");
    }

    //<--- Resolve the dispatch key: the reaction kernel is templated on the
    // model's Kernel struct, so the concrete type has to be known statically.
    if (dynamic_cast<MitchellSchaefferTD *>(ep_model.get()))
    {
        model_type = IonicModelType::MITCHELL_SCHAEFFER_TD_DEPENDENT;
    }
    else if (dynamic_cast<MitchellSchaeffer *>(ep_model.get()))
    {
        model_type = IonicModelType::MITCHELL_SCHAEFFER;
    }
    else if (dynamic_cast<FentonKarma *>(ep_model.get()))
    {
        model_type = IonicModelType::FENTON_KARMA;
    }
    else if (dynamic_cast<TP06Epi *>(ep_model.get()))
    {
        model_type = IonicModelType::TENTUSSCHER_PANFILOV_EPI;
    }
    else if (dynamic_cast<TP06Endo *>(ep_model.get()))
    {
        model_type = IonicModelType::TENTUSSCHER_PANFILOV_ENDO;
    }
    else
    {
        throw std::runtime_error(
            "ReactionSolver::RegisterModels(): unknown EP model '" + ep_model->GetName() +
            "'. Add it to IonicModelType and to the dispatch switches in this file.");
    }

    if (contraction_model_)
    {
        if (dynamic_cast<Land17Model *>(contraction_model_.get()))
        {
            contraction_type = ContractionModelType::LAND_2017;
        }
        else
        {
            throw std::runtime_error(
                "ReactionSolver::RegisterModels(): unknown contraction model '" +
                contraction_model_->GetName() + "'.");
        }
    }
    else
    {
        contraction_type = ContractionModelType::NONE;
    }

    model = std::move(ep_model);
    contraction_model = std::move(contraction_model_);

    has_td_dependency = dynamic_cast<GotranxODEModelWithThermalDamage *>(model.get()) != nullptr;
    if (has_td_dependency)
    {
        const int num_time_constants =
            dynamic_cast<GotranxODEModelWithThermalDamage *>(model.get())->GetTimeConstantsIdxs().size();
        td_delta_tau.resize(num_time_constants, 0.0);
        td_healthy_tau.resize(num_time_constants, 0.0);
    }

    //<--- Resource allocation: with no contraction model, none of the
    // contraction arrays are touched, so a pure-EP run allocates nothing extra.
    if (!contraction_model)
    {
        contraction_values.SetSize(0);
        contraction_params.SetSize(0);
        Ta_tvector_.SetSize(0);
        tension_updated_ = false;
    }
}


void ReactionSolver::RegisterModels(IonicModelType ep_type,
                                    ContractionModelType contraction_type_)
{
    std::unique_ptr<EPModelBase> ep;
    switch (ep_type)
    {
    case IonicModelType::MITCHELL_SCHAEFFER:
        ep = std::make_unique<MitchellSchaeffer>(); break;
    case IonicModelType::FENTON_KARMA:
        ep = std::make_unique<FentonKarma>(); break;
    case IonicModelType::TENTUSSCHER_PANFILOV_EPI:
        ep = std::make_unique<TP06Epi>(); break;
    case IonicModelType::TENTUSSCHER_PANFILOV_ENDO:
        ep = std::make_unique<TP06Endo>(); break;
    case IonicModelType::MITCHELL_SCHAEFFER_TD_DEPENDENT:
        ep = std::make_unique<MitchellSchaefferTD>(); break;
    default:
        mfem_error("ReactionSolver::RegisterModels(): unsupported ionic model type");
    }

    std::unique_ptr<ContractionModelBase> contraction;
    switch (contraction_type_)
    {
    case ContractionModelType::NONE:
        break;
    case ContractionModelType::LAND_2017:
        contraction = std::make_unique<Land17Model>(); break;
    default:
        mfem_error("ReactionSolver::RegisterModels(): unsupported contraction model type");
    }

    RegisterModels(std::move(ep), std::move(contraction));
}


const Vector &ReactionSolver::GetActiveTension()
{
    MFEM_VERIFY(contraction_model != nullptr,
                "ReactionSolver::GetActiveTension(): no contraction model registered. "
                "Call RegisterModels() with one, or do not query the tension.");

    //<--- Cached: the contraction ODEs have already been advanced for this step.
    if (tension_updated_) { return Ta_tvector_; }

    MFEM_VERIFY(contraction_values.Size() > 0,
                "ReactionSolver::GetActiveTension(): Setup() must run before the first step.");

    const int n = fes_truevsize;
    const int ca_idx = model->GetCalciumIndex();

    // Integrate everything that has elapsed since the last query, in substeps
    // no larger than the EP step so the accuracy does not depend on how often
    // the caller asks. Calcium is held at its present value across the substeps
    // -- the same staggering approximation the outer coupling already makes, so
    // the stride must stay short against the calcium transient (tens of ms).
    const real_t elapsed = (dt_pending_ > 0.0) ? dt_pending_ : dt_;
    const real_t dt_cap = (dt_ > 0.0) ? dt_ : elapsed;
    const int nsub = std::max(1, (int)std::ceil(elapsed / dt_cap - 1e-12));

    ContractionStepArgs args;
    args.ndofs = n;
    args.dt = elapsed / nsub;
    args.scheme = (int)scheme;
    // The EP states are already in SoA layout, so the calcium row is a
    // contiguous span of `values` -- no gather and no scratch vector needed.
    args.calcium = values.Read() + static_cast<size_t>(ca_idx) * n;
    args.calcium_scale = calcium_scale_;
    args.states = contraction_values.ReadWrite();
    args.params = contraction_params.Read();
    args.per_dof_params = false;
    args.active_tension = Ta_tvector_.Write();

    for (int sub = 0; sub < nsub; sub++)
    {
        contraction_model->AdvanceODE(args);
    }

    dt_pending_ = 0.0;
    tension_updated_ = true;
    return Ta_tvector_;
}

void ReactionSolver::GetPotential(Vector &u)
{
    const int n = fes_truevsize;
    const int off = model->potential_idx * n;
    const bool dimless = model->dimensionless;
    const real_t vmin = Vmin, vrange = Vrange;

    u.SetSize(n);
    const real_t *d_values = values.Read();
    real_t *d_u = u.Write();

    mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
    {
        const real_t s = d_values[off + i];
        d_u[i] = dimless ? (s * vrange + vmin) : s;
    });
}

void ReactionSolver::SetPotential(const Vector &u)
{
    MFEM_ASSERT(u.Size() == fes_truevsize, "Incompatible sizes in ReactionSolver::SetPotential");
    const int n = fes_truevsize;
    const int off = model->potential_idx * n;
    const bool dimless = model->dimensionless;
    const real_t vmin = Vmin, invvrange = invVrange;
    const bool limit = limit_voltage;

    const real_t *d_u = u.Read();
    real_t *d_values = values.ReadWrite();

    mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
    {
        const real_t v = (d_u[i] - vmin) * invvrange;
        d_values[off + i] = dimless ? (limit ? fabs(v) : v) : d_u[i];
    });
    states_gfs_stale = true;
}

void ReactionSolver::GetDefaultStates(std::vector<double> &default_states)
{
    int num_states = model->GetNumStates();
    default_states.resize(num_states);
    model->init_state_values(default_states.data());
}

void ReactionSolver::GetDefaultParameters(std::vector<double> &default_params)
{
    int num_params = model->GetNumParameters();
    default_params.resize(num_params);
    model->init_parameter_values(default_params.data());
}

void ReactionSolver::SetStimulation(Coefficient *stim, bool time_independent)
{
    stimulation_coeff = stim;
    stim_time_independent = time_independent;
    stim_projected = false;
}


void ReactionSolver::RegisterFields(DataCollection &dc)
{
    int num_states = model->GetNumStates();
    
    // Register each state variable, apart from potential
    for (int j = 0; j < num_states; j++)
    {
        if (j == model->potential_idx)
            continue; // Skip potential
        
        int adjusted_index = j;
        if (j > model->potential_idx) {
            adjusted_index -= 1;
        }
        
        // Use the already allocated states_gfs and states_vectors
        // Register with DataCollection
        std::string field_name = "state_" + std::to_string(j);
        dc.RegisterField(field_name.c_str(), states_gfs[adjusted_index]);
    }
}

ParGridFunction *ReactionSolver::GetStateGridFunction(int state_index)
{
    MFEM_ASSERT(state_index >= 0 && state_index < model->GetNumStates(),
                "Invalid state index in GetStateGridFunction");
    MFEM_ASSERT(state_index != model->potential_idx,
                "Cannot get potential through GetStateGridFunction, use GetPotential instead");

    // The caller is about to read this grid function, so make it current. It will go
    // stale again on the next Step(); see SyncStateGridFunctions().
    SyncStateGridFunctions();

    int adjusted_index = state_index;
    if (state_index > model->potential_idx)
    {
        adjusted_index -= 1; // Adjust index since potential is skipped
    }

    // Use the already allocated states_gfs and states_vectors
    return states_gfs[adjusted_index];
}

// Project the stimulation coefficient onto the t-dof vector for time `t_stim`,
// unless one of the opt-in strategies says it can be skipped.
//
// This is the single most expensive operation in the reaction step: it is a *host*
// sweep over every dof (~225 ns/dof, i.e. ~48 ms for 216k dofs) and, under a device
// backend, a synchronisation point too. Done naively it outweighs the actual ODE
// integration by roughly three orders of magnitude. See "Enforcing the stimulation
// efficiently" in reaction_solver.hpp for which strategy applies when.
void ReactionSolver::SetSeparableStimulation(Coefficient *mask,
                                             std::function<real_t(real_t)> amplitude)
{
    MFEM_VERIFY(mask && amplitude, "SetSeparableStimulation needs both a mask and an amplitude.");
    stim_mask_coeff = mask;
    stim_amplitude = std::move(amplitude);
    stim_mask_projected = false;
    stimulation_coeff = nullptr;   // the separable path replaces the space-time coefficient
}

void ReactionSolver::ProjectStimulation(real_t t_stim)
{
    // Separable stimulus: project the spatial mask once, then only rescale it.
    if (stim_mask_coeff)
    {
        if (!stim_mask_projected)
        {
            stimulation_gf.ProjectCoefficient(*stim_mask_coeff);
            stimulation_gf.GetTrueDofs(stim_mask_vec);
            stim_mask_vec.UseDevice(true);
            stim_mask_projected = true;
        }
        const real_t a = stim_amplitude(t_stim);
        // Skip the kernel entirely across the (common) stretches where the stimulus is
        // off and the vector is already zero.
        if (a == 0.0)
        {
            if (!stim_vec_is_zero) { stimulation_vec = 0.0; stim_vec_is_zero = true; }
        }
        else
        {
            stimulation_vec.Set(a, stim_mask_vec);
            stim_vec_is_zero = false;
        }
        return;
    }

    if (!stimulation_coeff) { return; }

    // A time-independent coefficient only has to be projected once; a time
    // dependent one only while it can be nonzero, if a window was declared.
    const bool needs_projection =
        stim_time_independent
        ? !stim_projected
        : (!has_stim_window ||
           (t_stim >= stim_window_begin && t_stim <= stim_window_end));

    if (needs_projection)
    {
        stimulation_coeff->SetTime(t_stim);
        stimulation_gf.ProjectCoefficient(*stimulation_coeff);
        stimulation_gf.GetTrueDofs(stimulation_vec);
        stim_projected = true;
        stim_vec_is_zero = false;
    }
    else if (!stim_time_independent && !stim_vec_is_zero)
    {
        // Outside the declared window the caller guarantees the stimulus is zero;
        // zero the vector once and then leave it alone.
        stimulation_vec = 0.0;
        stim_vec_is_zero = true;
    }
}

void ReactionSolver::Step(Vector &x, real_t &t, real_t &dt, bool provisional)
{
    // Inner loop time step
    const real_t dt_ode = dt / ode_substeps;
    real_t current_time = t;

    // Cache the step size for the contraction models and invalidate the active
    // tension: it is recomputed lazily, from the calcium this step produces.
    dt_ = dt;
    dt_pending_ += dt;
    tension_updated_ = false;

    // Cache frequently used values OUTSIDE the substep loop
    const int n = fes_truevsize;
    const bool use_dimensionless = model->dimensionless;
    const int potential_idx = model->potential_idx;
    const int pot_off = potential_idx * n;

    //<--- Seed the potential entry of `values` from the incoming vector x
    {
        const real_t vmin = Vmin, invvrange = invVrange;
        const bool limit = limit_voltage;
        const real_t *d_x = x.Read();
        real_t *d_values = values.ReadWrite();
        mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
        {
            d_values[pot_off + i] = use_dimensionless
                                    ? (limit ? fabs((d_x[i] - vmin) * invvrange)
                                             : (d_x[i] - vmin) * invvrange)
                                    : d_x[i];
        });
    }

    //<--- Temperature/damage dependent parameter update (once per step)
    if (has_td_dependency)
    {
        auto *td_model = dynamic_cast<GotranxODEModelWithThermalDamage *>(model.get());

        if (!td_device_data_ready)
        {
            auto idxs = td_model->GetTimeConstantsIdxs();
            const int ntau = static_cast<int>(idxs.size());
            td_tau_idx_d.SetSize(ntau);
            td_healthy_tau_d.SetSize(ntau);
            td_delta_tau_d.SetSize(ntau);
            for (int k = 0; k < ntau; k++)
            {
                td_tau_idx_d[k] = idxs[k];
                td_healthy_tau_d[k] = td_healthy_tau[k];
                td_delta_tau_d[k] = td_delta_tau[k];
            }
            td_device_data_ready = true;
        }

        if (temperature_gf) { temperature_gf->GetTrueDofs(temperature_vec); }

        // Apply the (host-only) damage function before entering the kernel.
        if (damage_gf)
        {
            damage_gf->GetTrueDofs(damage_vec);
            damage_transformed.SetSize(damage_vec.Size());
            const real_t *h_d = damage_vec.HostRead();
            real_t *h_dt = damage_transformed.HostWrite();
            for (int i = 0; i < damage_vec.Size(); i++) { h_dt[i] = td_damage_func(h_d[i]); }
        }

        const bool have_T = temperature_vec.Size() > 0;
        const bool have_D = damage_gf && damage_transformed.Size() > 0;

        UpdateThermalDamageParams(
            n, model->GetNumParameters(),
            have_T ? temperature_vec.Read() : nullptr,
            have_D ? damage_transformed.Read() : nullptr,
            have_T, have_D,
            td_A, td_B, td_Tref, td_Q10,
            td_model->eta_idx, td_model->gamma_idx, td_model->Q_idx,
            td_tau_idx_d.Size(), td_tau_idx_d.Read(),
            td_healthy_tau_d.Read(), td_delta_tau_d.Read(),
            parameters.ReadWrite());
    }

    //<--- Stimulation: by default the coefficient is sampled once for the whole
    // outer time step, at its end (t + dt). With ode_substeps == 1 that is exactly
    // the same evaluation the substep loop would do, so the default costs nothing in
    // accuracy; with ode_substeps == N it divides the projection cost by N.
    if (!substep_stim_projection)
    {
        ProjectStimulation(t + stimulus_sample_fraction * dt);
    }

    //<--- Substep loop: integrate the pointwise ODEs on device
    for (int sub = 0; sub < ode_substeps; sub++)
    {
        // Re-sample the stimulation at every substep, if the caller asked for it.
        if (substep_stim_projection)
        {
            ProjectStimulation(current_time + stimulus_sample_fraction * dt_ode);
        }

        const real_t *d_stim = stimulation_vec.Read();
        const real_t *d_chi = chi_vec.Read();
        const real_t *d_Cm = Cm_vec.Read();
        real_t *d_values = values.ReadWrite();
        const real_t *d_params = per_dof_params ? parameters.Read()
                                                : uniform_params.Read();

        switch (model_type)
        {
        case IonicModelType::MITCHELL_SCHAEFFER:
            ReactionSubstepDispatch<MitchellSchaeffer::Kernel>(
                per_dof_params, n, (int)scheme, current_time, dt_ode, Vrange,
                d_stim, d_chi, d_Cm, d_values, d_params);
            break;
        case IonicModelType::FENTON_KARMA:
            ReactionSubstepDispatch<FentonKarma::Kernel>(
                per_dof_params, n, (int)scheme, current_time, dt_ode, Vrange,
                d_stim, d_chi, d_Cm, d_values, d_params);
            break;
        case IonicModelType::TENTUSSCHER_PANFILOV_EPI:
            ReactionSubstepDispatch<TP06Epi::Kernel>(
                per_dof_params, n, (int)scheme, current_time, dt_ode, Vrange,
                d_stim, d_chi, d_Cm, d_values, d_params);
            break;
        case IonicModelType::TENTUSSCHER_PANFILOV_ENDO:
            ReactionSubstepDispatch<TP06Endo::Kernel>(
                per_dof_params, n, (int)scheme, current_time, dt_ode, Vrange,
                d_stim, d_chi, d_Cm, d_values, d_params);
            break;
        case IonicModelType::MITCHELL_SCHAEFFER_TD_DEPENDENT:
            ReactionSubstepDispatch<MitchellSchaefferTD::Kernel>(
                per_dof_params, n, (int)scheme, current_time, dt_ode, Vrange,
                d_stim, d_chi, d_Cm, d_values, d_params);
            break;
        default:
            mfem_error("Unsupported ionic model type in ReactionSolver::Step");
        }

        current_time += dt_ode;
    }

    //<--- Write the potential back into x, clamped to the physical range
    {
        const real_t vmin = Vmin, vmax = Vmax, vrange = Vrange;
        const bool limit = limit_voltage;
        const real_t *d_values = values.Read();
        real_t *d_x = x.Write();
        mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
        {
            real_t val = d_values[pot_off + i];
            if (use_dimensionless) { val = val * vrange + vmin; }
            d_x[i] = limit ? fmin(fmax(val, vmin), vmax) : val;
        });
    }

    //<--- Refreshing the state grid functions is deferred to SyncStateGridFunctions().
    states_gfs_stale = true;

    t = provisional ? t : current_time;
}

void ReactionSolver::EnableHeterogeneousParameters(bool enable)
{
    MFEM_VERIFY(values.Size() == 0,
                "EnableHeterogeneousParameters() must be called before Setup().");
    per_dof_params = enable;
}

void ReactionSolver::SetParameterField(int param_idx, const Vector &vals)
{
    const int n = fes_truevsize;
    const int np = model->GetNumParameters();
    MFEM_VERIFY(per_dof_params,
                "SetParameterField() requires EnableHeterogeneousParameters() before Setup().");
    MFEM_VERIFY(param_idx >= 0 && param_idx < np, "Invalid parameter index.");
    MFEM_VERIFY(vals.Size() == n, "Parameter field must be a true-dof vector.");
    MFEM_VERIFY(parameters.Size() == np * n, "Per-dof parameter storage not allocated.");

    const int off = param_idx * n;
    const real_t *d_v = vals.Read();
    real_t *d_p = parameters.ReadWrite();
    mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { d_p[off + i] = d_v[i]; });
}

void ReactionSolver::SyncStateGridFunctions()
{
    if (!states_gfs_stale) { return; }

    const int n = fes_truevsize;
    const int potential_idx = model->potential_idx;

    for (size_t k = 0; k < states_vectors.size(); k++)
    {
        const int state_idx = (k >= static_cast<size_t>(potential_idx)) ? (int)k + 1 : (int)k;
        const int off = state_idx * n;
        const real_t *d_values = values.Read();
        real_t *d_sv = states_vectors[k]->Write();
        mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { d_sv[i] = d_values[off + i]; });
        states_gfs[k]->SetFromTrueDofs(*states_vectors[k]);
    }

    states_gfs_stale = false;
}

void ReactionSolver::Update()
{
    // Finite element space might have changed; update truevsize, grid functions, and vectors
    // The fes space changed because the mesh changed (e.g., AMR), so we need to update our internal data structures
    // No need to update the potential as it will be provided at each Step call
    
    int old_size = fes_truevsize; 
    fes_truevsize = fes->GetTrueVSize();
    
    // Update grid functions and get their true DOFs in one pass
    stimulation_gf.Update();
    chi_gf.Update();
    Cm_gf.Update();
    stimulation_gf.GetTrueDofs(stimulation_vec);
    chi_gf.GetTrueDofs(chi_vec);
    Cm_gf.GetTrueDofs(Cm_vec);

    // Update states fields and their vectors
    for (size_t k = 0; k < states_gfs.size(); k++) {
        states_gfs[k]->Update();
        states_gfs[k]->GetTrueDofs(*states_vectors[k]);
    }

    // Cache model properties
    const int num_states = model->GetNumStates();
    const int num_param = model->GetNumParameters();
    const int potential_idx = model->potential_idx;
    const int n = fes_truevsize;

    // Resize internal data structures. The SoA stride changes with the dof count,
    // so the arrays are rebuilt rather than resized in place; the states are
    // restored below from states_vectors, which were just interpolated onto the
    // new mesh, and the parameters are reset to their defaults.
    values.SetSize(num_states * n);     values.UseDevice(true);

    uniform_params.SetSize(num_param);  uniform_params.UseDevice(true);
    real_t *h_uparams = uniform_params.HostWrite();
    for (int k = 0; k < num_param; k++) { h_uparams[k] = parameters_default[k]; }

    if (per_dof_params)
    {
        parameters.SetSize(num_param * n);  parameters.UseDevice(true);
        real_t *h_params = parameters.HostWrite();
        for (int k = 0; k < num_param; k++)
        {
            for (int i = 0; i < n; i++) { h_params[k * n + i] = parameters_default[k]; }
        }
    }
    else
    {
        parameters.SetSize(0);
    }

    // Restore all non-potential states from states_vectors
    for (size_t k = 0; k < states_vectors.size(); k++)
    {
        const int state_idx = (k >= static_cast<size_t>(potential_idx)) ? (int)k + 1 : (int)k;
        const int off = state_idx * n;
        const real_t *d_sv = states_vectors[k]->Read();
        real_t *d_values = values.ReadWrite();
        mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { d_values[off + i] = d_sv[i]; });
    }

    // states_vectors/states_gfs were just rebuilt on the new mesh and match `values`.
    states_gfs_stale = false;
}

void ReactionSolver::PrintIndexTable()
{
    mfem_error("ReactionSolver::PrintIndexTable not implemented yet");
}
