#include "IonicModel.hpp"
#include <iostream>

IonicModel::IonicModel(int dofs, int model_type, int solver_type, int dt_ode) :
dofs(dofs), model_type(model_type), solver_type(solver_type), dt_ode(dt_ode)
{
    switch (model_type)
    {
    case 0:
        model = MitchellSchaeffer();
        num_param = 10;
        num_states = 2;
        potential_idx = 1;
        ionic_idx = 1;
        break;
    case 1:
        model = FentonKarma();
        num_param = 21;
        num_states = 3;
        potential_idx = 2;
        ionic_idx = 2;
        break;    
    default:
        break;
    }

}

IonicModel::~IonicModel() {
    delete model;
}

void IonicModel::Init() {

    // Initialize structure for states and parameters (1 vector per dof)
    states = std::vector<std::vector<double>>(dofs, std::vector<double>(num_states, 0.0));
    parameters = std::vector<std::vector<double>>(dofs, std::vector<double>(num_param, 0.0));

    // Initialize parameters and states
    for (int i = 0; i < dofs; i++) {
        model->init_parameters_values(parameters[i].data());
        model->init_state_values(states[i].data());        
    }

}

void IonicModel::PrintIndexTable() {
    std::cout << "Printing index: " << index << std::endl;
}

void IonicModel::ComputeIonicCurrent(double time) {

    // Compute Ionic current for each dof and store it in MFEM GridFunction
    for (int i = 0; i < dofs; i++) {
        model->rhs(states[i].data(), time, parameters[i].data(), values);
        ionic_current[i] = values[ionic_idx];   
    }    

}

void IonicModel::Solve(double time, double dt, Vector* potential) {

    // Inner loop time step
    dt_ode = dt/ode_steps;

    // Solve ODE on each FE dof
    for (int i = 0; i < ode_steps; i++)    // Loop for ODE solver inner time stepping
    {
        #pragma omp parallel for        
        for (int j = 0; j < dofs; j++)     // Loop on FE dofs (OpenMP parallelization)
        {
            switch (solver_type)
            {
            case 0: // Explicit Euler
                model->forward_explicit_euler(states[i].data(), time, dt_ode, parameters[i].data());
                break;
            case 1: // Rush Larsen
                model->forward_rush_larsen(states[i].data(), time, dt_ode, parameters[i].data());
                break;    
            case 2: // Generalized Rush Larsen
                model->forward_generalized_rush_larsen(states[i].data(), time, dt_ode, parameters[i].data());
                break;   
            case 3: // Hybrid Generalized Rush Larsen
                model->forward_hybrid_generalized_rush_larsen(states[i].data(), time, dt_ode, parameters[i].data());
                break;    
            case 4: // Simplified Implicit Euler
                model->forward_simplified_implicit_euler(states[i].data(), time, dt_ode, parameters[i].data());
                break;    
            default:
                break;
            }            
        }
        time += dt_ode;
    }

    // Update potential (MFEM Vector) from ODE solution
    for (int i = 0; i < dofs; i++)
    {
        potential[i] = states[i][potential_idx];
    }
}


void UpdatePotential(Vector* new_potential){
        
    for (int i = 0; i < dofs; i++)
    {
        states[i][potential_idx] = potential[i];
    }    
    
}
