#include "reaction_solver.hpp"
#include <iostream>

using namespace mfem;
using namespace mfem::electrophysiology;

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
    Cm_gf.ProjectCoefficient(*Cm_coeff);
    Cm_gf.GetTrueDofs(Cm_vec);

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
        mfem_warning("ReactionSolver::SetTemperatureGridFunction(): The current ionic model does not support temperature dependency. The provided grid function will be ignored.");
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

    // Compute variables for conversion to/from dimensionless potential
    Vrange = Vmax - Vmin;
    invVrange = 1.0 / Vrange;

    // Get number of states and parameters from the model
    int num_states = model->GetNumStates();
    int num_param = model->GetNumParameters();

    // Pre-allocate all vectors at once
    states.resize(fes_truevsize);
    values.resize(fes_truevsize);
    parameters.resize(fes_truevsize);
    
    // Pre-allocate inner vectors to avoid repeated allocations
    for (int i = 0; i < fes_truevsize; i++)
    {
        states[i].resize(num_states);
        values[i].resize(num_states);
        parameters[i].resize(num_param);
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

    // Initialize all DOFs with optimized loop
    for (int i = 0; i < fes_truevsize; i++)
    {
        // Initialize states
        if (use_provided_states)
        {
            std::copy(initial_states.begin(), initial_states.end(), states[i].begin());
            std::copy(initial_states.begin(), initial_states.end(), values[i].begin());
        }
        else
        {
            std::copy(default_states.begin(), default_states.end(), states[i].begin());
            std::copy(default_states.begin(), default_states.end(), values[i].begin());
        }

        // Initialize parameters
        if (use_provided_params)
        {
            std::copy(params.begin(), params.end(), parameters[i].begin());
        }
        else
        {
            std::copy(default_params.begin(), default_params.end(), parameters[i].begin());
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
    if (!parameters.empty()) {
        stimulation_vec = parameters[0][model->stim_ampl_idx];
    }
}

void ReactionSolver::GetPotential(Vector &u)
{
    u.SetSize(fes_truevsize);
    for (int i = 0; i < fes_truevsize; i++)
    {
        u[i] = model->dimensionless ? FromDimensionless(states[i][model->potential_idx]) : states[i][model->potential_idx];
    }
}

void ReactionSolver::SetPotential(const Vector &u)
{
    MFEM_ASSERT(u.Size() == fes_truevsize, "Incompatible sizes in ReactionSolver::SetPotential");
    for (int i = 0; i < fes_truevsize; i++)
    {
        states[i][model->potential_idx] = model->dimensionless ? ToDimensionless(u[i]) : u[i];
    }
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

void ReactionSolver::SetStimulation(Coefficient *stim)
{
    stimulation_coeff = stim;
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

    int adjusted_index = state_index;
    if (state_index > model->potential_idx)
    {
        adjusted_index -= 1; // Adjust index since potential is skipped
    }

    // Use the already allocated states_gfs and states_vectors
    return states_gfs[adjusted_index];
}

void ReactionSolver::Step(Vector &x, real_t &t, real_t &dt, bool provisional)
{
    // Inner loop time step
    double dt_ode = dt / ode_substeps;
    double current_time = t;

    // Cache frequently used values OUTSIDE the substep loop
    int num_states = model->GetNumStates();

    bool use_dimensionless = model->dimensionless;
    int potential_idx = model->potential_idx;
    int stim_ampl_idx = model->stim_ampl_idx;

    // If has temperature/damage dependency, get the temperature and damage vectors
    if (temperature_gf)
    {
        temperature_gf->GetTrueDofs(temperature_vec);
    }
    if (damage_gf)
    {
        damage_gf->GetTrueDofs(damage_vec);
    }

    // Initialize values of x (potential) from the input vector, and update temperature/damage dependent parameters if needed
    for (int i = 0; i < fes_truevsize; i++)
    {

        values[i][potential_idx] = use_dimensionless ? ToDimensionless(x[i]) : x[i];

        // Temperature and damage dependency update
        if (has_td_dependency)
        {
            auto td_model = dynamic_cast<GotranxODEModelWithThermalDamage*>(model.get());

            real_t temperature = temperature_vec.Size() > 0 ? temperature_vec[i] : td_Tref; // Default to 37C if not provided
            real_t damage = damage_vec.Size() > 0 ? td_damage_func(damage_vec[i]) : 0.0;   // Default to no damage if not provided

            // Update parameters based on temperature and damage
            parameters[i][td_model->eta_idx] = td_A * (1.0 + td_B * (temperature - td_Tref));    // eta = A[1+B(T-Tref)]
            parameters[i][td_model->Q_idx] = std::pow(td_Q10, (temperature - td_Tref) / 10.0);   // Q = Q10^((T - Tref)/10)
            parameters[i][td_model->gamma_idx] = (1.0 - damage);                                 // gamma = (1-f(D))

            // Update time constants with damage effect
            auto time_constants_idxs = td_model->GetTimeConstantsIdxs();
            for (size_t idx = 0; idx < time_constants_idxs.size(); idx++)
            {
                int param_idx = time_constants_idxs[idx];
                parameters[i][param_idx] = td_healthy_tau[idx] * (1.0 + td_delta_tau[idx] * damage);
            }
        }
    }
    // Pre-compute stimulation scaling factor
    auto stimulation_data = stimulation_vec.GetData();

    for (int i = 0; i < ode_substeps; i++) // Loop for ODE solver inner time stepping
    {
        // Project stimulation current once per substep
        if (stimulation_coeff)
        {
            stimulation_coeff->SetTime(current_time + dt_ode);
            stimulation_gf.ProjectCoefficient(*stimulation_coeff);
            stimulation_gf.GetTrueDofs(stimulation_vec);
            stimulation_data = stimulation_vec.GetData(); // Update pointer after projection
        }

        // Main computational loop - optimize memory access patterns
        for (int j = 0; j < fes_truevsize; j++)
        {
            // Compute Jscaling based on chi and Cm
            double Jscaling = model->dimensionless ? model->stim_sign * (chi_vec[j] * Cm_vec[j] * Vrange) : model->stim_sign;

            // Copy states efficiently - avoid std::copy overhead for small vectors
            double *state_ptr = states[j].data();
            double *value_ptr = values[j].data();

            // Manual unrolled copy for small num_states (typically 2-4 for states cardiac models)
            if (num_states <= 4)
            {
                for (int k = 0; k < num_states; k++)
                {
                    state_ptr[k] = value_ptr[k];
                }
            }
            else
            {
                std::memcpy(state_ptr, value_ptr, num_states * sizeof(double));
            }

            // Update stimulation current in parameters
            parameters[j][stim_ampl_idx] = use_dimensionless ? stimulation_data[j] / Jscaling : stimulation_data[j];

            // Call the appropriate time integration scheme
            switch (scheme)
            {
            case TimeIntegrationScheme::EXPLICIT_EULER:
                model->explicit_euler(state_ptr, current_time, dt_ode,
                                      parameters[j].data(), value_ptr);
                break;
            case TimeIntegrationScheme::FORWARD_EXPLICIT_EULER:
                model->forward_explicit_euler(state_ptr, current_time, dt_ode,
                                              parameters[j].data(), value_ptr);
                break;
            case TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN:
                model->generalized_rush_larsen(state_ptr, current_time, dt_ode,
                                               parameters[j].data(), value_ptr);
                break;
            case TimeIntegrationScheme::FORWARD_GENERALIZED_RUSH_LARSEN:
                model->forward_generalized_rush_larsen(state_ptr, current_time, dt_ode,
                                                       parameters[j].data(), value_ptr);
                break;
            case TimeIntegrationScheme::HYBRID_RUSH_LARSEN:
                model->hybrid_rush_larsen(state_ptr, current_time, dt_ode,
                                          parameters[j].data(), value_ptr);
                break;
            default:
                break;
            }
        }
        current_time += dt_ode;
    }

    // Optimized final update loop - Combined loop for better cache locality
    for (int i = 0; i < fes_truevsize; i++)
    {
        // Update potential
        x[i] = use_dimensionless ? FromDimensionless(values[i][potential_idx]) : values[i][potential_idx];
        x[i] = std::clamp(x[i], Vmin, Vmax);

        // Update fields in the same loop
        for (size_t k = 0; k < states_vectors.size(); k++)
        {
            int state_idx = (k >= static_cast<size_t>(potential_idx)) ? k + 1 : k;
            (*states_vectors[k])[i] = values[i][state_idx];
        }
    }

    // Update all states ParGridFunctions from their vectors
    for (size_t k = 0; k < states_gfs.size(); k++)
    {
        states_gfs[k]->SetFromTrueDofs(*states_vectors[k]);
    }

    t = provisional ? t : current_time;
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

    // Resize internal data structures
    states.resize(fes_truevsize);
    values.resize(fes_truevsize);
    parameters.resize(fes_truevsize);

    // Cache model properties
    int num_states = model->GetNumStates();
    int num_param = model->GetNumParameters();
    int potential_idx = model->potential_idx;

    // Initialize inner vectors only for NEW elements (when mesh was refined)
    for (int i = old_size; i < fes_truevsize; i++)
    {
        states[i].resize(num_states);
        values[i].resize(num_states);
        parameters[i].resize(num_param);
        
        // Initialize parameters for new DOFs immediately
        std::copy(parameters_default.begin(), parameters_default.end(), parameters[i].begin());
    }

    // Update all DOFs from states_vectors - optimized loop order for cache locality
    for (int i = 0; i < fes_truevsize; i++)
    {
        for (size_t k = 0; k < states_vectors.size(); k++)
        {
            int state_idx = (k >= static_cast<size_t>(potential_idx)) ? k + 1 : k;
            double value = (*states_vectors[k])[i];
            states[i][state_idx] = value;
            values[i][state_idx] = value;
        }
    }
}

void ReactionSolver::PrintIndexTable()
{
    mfem_error("ReactionSolver::PrintIndexTable not implemented yet");
}