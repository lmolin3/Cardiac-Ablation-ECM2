#include "reaction_solver.hpp"
#include <iostream>

using namespace mfem;
using namespace mfem::electrophysiology;

ReactionSolver::ReactionSolver(ParFiniteElementSpace *fes_, IonicModelType model_type_, TimeIntegrationScheme scheme_type, int ode_substeps_)
    : fes(fes_), fes_truevsize(fes_->GetTrueVSize()), model_type(model_type_), scheme(scheme_type), ode_substeps(ode_substeps_)
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
    // For each state variable, apart from potential, register a field
    int num_states = model->GetNumStates();
    for (int j = 0; j < num_states; j++)
    {
        if (j == model->potential_idx)
            continue; // Skip potential, already registered elsewhere

        // Create a ParGridFunction and its vector for update for the state variable and keep track of it in registered_fields
        std::string field_name = "state_" + std::to_string(j);
        ParGridFunction *state_gf = new ParGridFunction(fes);
        Vector *state_vec = new Vector(fes_truevsize);
        registered_fields.push_back(state_gf);
        registered_fields_vectors.push_back(state_vec);

        // Register the field with the DataCollection
        dc.RegisterField(field_name.c_str(), state_gf);
    }

}

void ReactionSolver::Step(Vector &x, real_t &t, real_t &dt, bool provisional)
{
    // Inner loop time step
    double dt_ode = dt / ode_substeps;
    double current_time = t;
    
    // Cache frequently used values OUTSIDE the substep loop (moved from StepOld)
    int num_states = model->GetNumStates();
    const double chi = 2e3;
    const double Cm = 1e-3;
    double Jscaling = model->dimensionless ? model->stim_sign * (chi * Cm * Vrange) : model->stim_sign;
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
            // Copy states efficiently - avoid std::copy overhead for small vectors
            double* state_ptr = states[j].data();
            double* value_ptr = values[j].data();
            
            // Manual unrolled copy for small num_states (typically 2-4 for cardiac models)
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
            
            // Update registered fields in the same loop
            for (size_t k = 0; k < registered_fields_vectors.size(); k++) {
                int state_idx = (k >= potential_idx) ? k + 1 : k;
                (*registered_fields_vectors[k])[i] = values[i][state_idx];
            }
        }
    }

    // Update all registered ParGridFunctions from their vectors
    for (size_t k = 0; k < registered_fields.size(); k++) {
        registered_fields[k]->SetFromTrueDofs(*registered_fields_vectors[k]);
    }

    t = provisional ? t : current_time;
}

void ReactionSolver::StepOld(Vector &x, real_t &t, real_t &dt, bool provisional)
{
    // Inner loop time step
    double dt_ode = dt / ode_substeps;
    double current_time = t;

    // Initialize values of x (potential) from the input vector
    for (int i = 0; i < fes_truevsize; i++)
    {
        values[i][model->potential_idx] = model->dimensionless ? ToDimensionless(x[i]) : x[i];
    }

    // Solve ODE on each FE dof
    // NOTE: for parallelization usign device we can use MFEM's forall
    //       We need to: (i) replace for with forall
    //                   (ii) put outer loop for ode_substeps inside the inner loop (so the outer loop is on fes_truevsize)
    //                   (iii) make sure that device data is used (ReadWrite())
    //                   (iv) re-organize how parameters and states are passed. we need SoA (Structure of Arrays) instead of AoS (Array of Structures)
    //                        so that each thread can access its own data without conflicts.
    //                        so that e.g. parameters[j] is a Vector on device
    //
    //      We might organize multi-vector variables as one large vector, and then use offsets to access each dof's data.
    //      e.g.
    //      Vector states_flat;      // Size: fes_truevsize * num_states
    //      Vector values_flat;      // Size: fes_truevsize * num_states
    //      Vector parameters_flat;  // Size: fes_truevsize * num_params
    //      auto d_states = states_flat.ReadWrite();
    //      auto d_values = values_flat.ReadWrite();
    //      auto d_parameters = parameters_flat.ReadWrite();
    //      mfem::forall(fes_truevsize, [=] MFEM_HOST_DEVICE (int i) {
    //            double *state_i = &d_states[i * num_states];
    //            double *param_i = &d_parameters[i * num_params];
    //            ...
    //      });

    for (int i = 0; i < ode_substeps; i++) // Loop for ODE solver inner time stepping
    {
        // Project stimulation current
        if (stimulation_coeff)
        {
            stimulation_coeff->SetTime(current_time+dt_ode);
            stimulation_gf.ProjectCoefficient(*stimulation_coeff);
            stimulation_gf.GetTrueDofs(stimulation_vec);
        }

        // Placeholder for physical constants
        const double chi = 2e3;
        const double Cm = 1e-3;
        double Jscaling = model->dimensionless ? model->stim_sign * (chi * Cm * Vrange) : model->stim_sign;
        int num_states = model->GetNumStates();

        for (int j = 0; j < fes_truevsize; j++) // Loop on FE dofs
        {
            // Update states from current values
            for (size_t k = 0; k < num_states; k++)
            {
                states[j][k] = values[j][k];
            }

            // Update stimulation current in parameters
            parameters[j][model->stim_ampl_idx] = model->dimensionless ? stimulation_vec[j] / Jscaling : stimulation_vec[j];

            // Call the appropriate time integration scheme
            switch (scheme)
            {
            case TimeIntegrationScheme::EXPLICIT_EULER:
                model->explicit_euler(states[j].data(), current_time, dt_ode, parameters[j].data(), values[j].data());
                break;
            case TimeIntegrationScheme::FORWARD_EXPLICIT_EULER:
                model->forward_explicit_euler(states[j].data(), current_time, dt_ode, parameters[j].data(), values[j].data());
                break;
            case TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN:
                model->generalized_rush_larsen(states[j].data(), current_time, dt_ode, parameters[j].data(), values[j].data());
                break;
            case TimeIntegrationScheme::FORWARD_GENERALIZED_RUSH_LARSEN:
                model->forward_generalized_rush_larsen(states[j].data(), current_time, dt_ode, parameters[j].data(), values[j].data());
                break;
            case TimeIntegrationScheme::HYBRID_RUSH_LARSEN:
                model->hybrid_rush_larsen(states[j].data(), current_time, dt_ode, parameters[j].data(), values[j].data());
                break;
            default:
                break;
            }
        }
        current_time += dt_ode;
    }

    // Update potential (MFEM Vector) from ODE solution, and all registered fields
    // If uses dimensionless potential, convert back to [Vmin, Vmax] range
    // @note: this can be optimized by combining with the previous loop,
    //       but we'd require a vector for each registered field
    for (int i = 0; i < fes_truevsize; i++)
    {
        x[i] = model->dimensionless ? FromDimensionless(values[i][model->potential_idx]) : values[i][model->potential_idx];
        x[i] = std::clamp(x[i], Vmin, Vmax);
        //  Update vector for each registered field
        for (size_t k = 0; k < registered_fields_vectors.size(); k++)
        {
            // Find the corresponding state index (accounting for skipped potential)
            int state_idx = (k >= model->potential_idx) ? k + 1 : k;
            (*registered_fields_vectors[k])[i] = values[i][state_idx];
        }
    }

    // Update all registered ParGridFunctions from their vectors
    for (size_t k = 0; k < registered_fields.size(); k++)
    {
        registered_fields[k]->SetFromTrueDofs(*registered_fields_vectors[k]);
    }

    t = provisional ? t : current_time;
}

void ReactionSolver::PrintIndexTable()
{
    mfem_error("ReactionSolver::PrintIndexTable not implemented yet");
}