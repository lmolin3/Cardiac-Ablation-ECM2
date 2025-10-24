#include "reaction_solver.hpp"
#include <iostream>

using namespace mfem;
using namespace mfem::electrophysiology;

ReactionSolver::ReactionSolver(ParFiniteElementSpace *fes_, Coefficient *chi_coeff_, Coefficient *Cm_coeff_, IonicModelType model_type_, TimeIntegrationScheme scheme_type, int ode_substeps_)
    : fes(fes_), fes_truevsize(fes_->GetTrueVSize()), model_type(model_type_), scheme(scheme_type), ode_substeps(ode_substeps_), chi_coeff(chi_coeff_), Cm_coeff(Cm_coeff_)
{
    switch (model_type)
    {
    case IonicModelType::MITCHELL_SCHAEFFER:
        model = std::make_unique<MitchellSchaeffer>();
        break;
    case IonicModelType::FENTON_KARMA:
        model = std::make_unique<FentonKarma>();
        break;
    default:
        mfem_error("Unsupported ionic model type");
    }

    stimulation_gf.SetSpace(fes); 
    stimulation_gf = 0.0;
    stimulation_gf.GetTrueDofs(stimulation_vec);

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
}

ReactionSolver::~ReactionSolver()
{
    for (auto gf : registered_fields)
    {
        delete gf;
    }
    for (auto vec : registered_fields_vectors)
    {
        delete vec;
    }
}

void ReactionSolver::Setup(const std::vector<double> &initial_states, const std::vector<double> &params)
{
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
    if (!use_provided_params) {
        model->init_parameter_values(default_params.data());
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
    int num_non_potential_states = num_states - 1;
    
    // Ensure registered_fields is large enough
    if (static_cast<int>(registered_fields.size()) < num_non_potential_states) {
        registered_fields.resize(num_non_potential_states, nullptr);
        registered_fields_vectors.resize(num_non_potential_states, nullptr);
    }
    
    // Register each state variable, apart from potential
    for (int j = 0; j < num_states; j++)
    {
        if (j == model->potential_idx)
            continue; // Skip potential
        
        int adjusted_index = j;
        if (j > model->potential_idx) {
            adjusted_index -= 1;
        }
        
        // Only create if not already registered
        if (registered_fields[adjusted_index] == nullptr) {
            ParGridFunction *state_gf = new ParGridFunction(fes);
            Vector *state_vec = new Vector(fes_truevsize);
            registered_fields[adjusted_index] = state_gf;
            registered_fields_vectors[adjusted_index] = state_vec;
        }
        
        // Register with DataCollection
        std::string field_name = "state_" + std::to_string(j);
        dc.RegisterField(field_name.c_str(), registered_fields[adjusted_index]);
    }
}

bool ReactionSolver::GetStateGridFunction(int state_index, ParGridFunction *&state_gf)
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

    // Ensure registered_fields is large enough to hold all non-potential states
    int num_non_potential_states = model->GetNumStates() - 1;
    if (static_cast<int>(registered_fields.size()) < num_non_potential_states)
    {
        registered_fields.resize(num_non_potential_states, nullptr);
        registered_fields_vectors.resize(num_non_potential_states, nullptr);
    }

    // Check if the field is already registered (non-null)
    if (registered_fields[adjusted_index] != nullptr)
    {
        state_gf = registered_fields[adjusted_index];
        return false; // Caller does NOT own the pointer
    }

    // Field not registered yet - create and register it now
    state_gf = new ParGridFunction(fes);
    Vector *state_vec = new Vector(fes_truevsize);

    // Populate with current state values
    for (int i = 0; i < fes_truevsize; i++)
    {
        (*state_vec)[i] = values[i][state_index];
    }
    state_gf->SetFromTrueDofs(*state_vec);

    // Store in the sparse array
    registered_fields[adjusted_index] = state_gf;
    registered_fields_vectors[adjusted_index] = state_vec;

    return false; // Caller does NOT own the pointer (we manage it)
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

    // Initialize values of x (potential) from the input vector - vectorized
    for (int i = 0; i < fes_truevsize; i++) {
        values[i][potential_idx] = use_dimensionless ? ToDimensionless(x[i]) : x[i];
    }

    // Pre-compute stimulation scaling factor
    auto stimulation_data = stimulation_vec.GetData();

    for (int i = 0; i < ode_substeps; i++) // Loop for ODE solver inner time stepping
    {
        // Project stimulation current once per substep
        if (stimulation_coeff) {
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
            double* state_ptr = states[j].data();
            double* value_ptr = values[j].data();

            // Manual unrolled copy for small num_states (typically 2-4 for states cardiac models)
            if (num_states <= 4) {
                for (int k = 0; k < num_states; k++) {
                    state_ptr[k] = value_ptr[k];
                }
            } else {
                std::memcpy(state_ptr, value_ptr, num_states * sizeof(double));
            }

            // Update stimulation current in parameters
            parameters[j][stim_ampl_idx] = use_dimensionless ? 
                stimulation_data[j] / Jscaling : stimulation_data[j];

            // Call the appropriate time integration scheme
            switch (scheme) {
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

    // Optimized final update loop - combine with field updates for better cache locality
    if (registered_fields_vectors.empty()) {
        // Fast path when no registered fields
        if (use_dimensionless) {
            for (int i = 0; i < fes_truevsize; i++) {
                x[i] = std::clamp(FromDimensionless(values[i][potential_idx]), Vmin, Vmax);
            }
        } else {
            for (int i = 0; i < fes_truevsize; i++) {
                x[i] = std::clamp(values[i][potential_idx], Vmin, Vmax);
            }
        }
    } else {
        // Combined loop for better cache locality (like StepOld)
        for (int i = 0; i < fes_truevsize; i++) {
            // Update potential
            x[i] = use_dimensionless ? FromDimensionless(values[i][potential_idx]) : values[i][potential_idx];
            x[i] = std::clamp(x[i], Vmin, Vmax);
            
            // Update registered fields in the same loop (skip null entries)
            for (size_t k = 0; k < registered_fields_vectors.size(); k++) {
                if (registered_fields_vectors[k] != nullptr) {
                    int state_idx = (k >= static_cast<size_t>(potential_idx)) ? k + 1 : k;
                    (*registered_fields_vectors[k])[i] = values[i][state_idx];
                }
            }
        }
    }

    // Update all registered ParGridFunctions from their vectors (skip null entries)
    for (size_t k = 0; k < registered_fields.size(); k++) {
        if (registered_fields[k] != nullptr) {
            registered_fields[k]->SetFromTrueDofs(*registered_fields_vectors[k]);
        }
    }

    t = provisional ? t : current_time;
}

void ReactionSolver::PrintIndexTable()
{
    mfem_error("ReactionSolver::PrintIndexTable not implemented yet");
}