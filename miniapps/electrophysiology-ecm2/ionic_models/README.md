This file presents the workflow required to integrate a new ionic model into the MFEM EP solver.
Gotranx (https://finsberg.github.io/gotranx) is a python based general ODE translator, used to convert the ODE system (stored in the gotranx compatible .ode format) into working python/C++ code. 


Note: generally we rely in ionic models stored in CellML format. However, new models (not available in CellML) or modifications to original models (e.g. to integrate temperature and damage dependency) can be directly developed using the .ode format.

Workflow to Generate a New Ionic Model
1. Download CellML file of ionic model (from repos like Physiome Project).
2. Convert to Gotran format (.ode)
▶ gotranx cellml2ode model.cellml -o model.ode
3. Generate code in Python or C++
▶ gotranx gotran2py model.ode –scheme generalized_rush_larsen -o model.py
▶ gotranx gotran2c model.ode –scheme generalized_rush_larsen -o model.h

Once the .h file is generated, some minimal modifications are required to integrate it in the MFEM API.

1. Remove any include, and include the gotranx wrapper class and wrap the code in the correct namespace

#pragma once
#include "gotranx_wrapper.hpp"

namespace mfem
{
    namespace electrophysiology
    {
        # GENERATED CODE
    }
}

2. Create a C++ class for the model with its constructor, defining some model-specific information
   * number of states, parameters, monitored vars --> used to initialize arrays for the model
   * indices of potential and stimulus related variables --> used to retrieve potential data and set stimuli (note this might vary across models)
   * if the model uses dimensionless membrane potential (will trigger scaling from the ReactionSolver before passing data)
   for indices you can use both the number directly, or retrieve it with state_index()/parameter_index() providing the specific name
   * The scaling of the current (e.g. FK model required negative sign)

        class MyIonicModel : public GotranxODEModel
        {
        public:
            // Constructor to initialize the base class metadata
            MyIonicModel() : GotranxODEModel()
            {
                NUM_STATES = ;
                NUM_PARAMS = ;
                NUM_MONITORED = ;

                potential_idx = state_index("my-potential-name");   // e.g. 1
                stim_ampl_idx = parameter_index("IstimAmplitude");
                stim_duration_idx = parameter_index("IstimPulseDuration");
                stim_start_idx = parameter_index("IstimStart");
                stim_end_idx = parameter_index("IstimEnd");
                stim_period_idx = parameter_index("IstimPeriod");

                stim_sign = 1; // 


                dimensionless = true; // true if model uses dimensionless potential
            }
        }


3. Add the new ionic model to the IonicModelType enum

        enum class IonicModelType : int
        {
            MITCHELL_SCHAEFFER = 0,
            FENTON_KARMA = 1,
            MyMODEL = 2
        };

4. Add the new ionic model in the constructor of ReactionSolver

    //<--- Initialize the ionic model based on the selected type
    switch (model_type)
    {
    case IonicModelType::MITCHELL_SCHAEFFER:
        model = std::make_unique<MitchellSchaeffer>();
        break;
    case IonicModelType::FENTON_KARMA:
        model = std::make_unique<FentonKarma>();
        break;
    case IonicModelType::MyMODEL:
        model = std::make_unique<MyMODEL>();
        break;
    default:
        mfem_error("Unsupported ionic model type");
    }



- How to introduce damage and temperature dependency
In our case we are interested in including the effect of temperature and damage into the ionic model.
Given a generic ionic model
Cm dudt + Iion = Iapp
dwdt = R(u,w)

The temperature/damage dependent version reads
Cm dudt + eta(T) * gamma(D) * Iion = Iapp
dwdt = Q(T) * R(u,w)

Additionally, we assume that time constant increase with damage
tau = tau_0 (1 + delta_tau * D)

We assume here that the parameters value is set from the user outside the ionic model (this ensures that the modification is less invasive, and provides flexibility in choosing different dependencies of scaling factors).


To do so we include minimal modifications to the specific model .ode file, and then proceed as shown above for a generic ionic model.

1. Define the necessary parameters for the scaling factors
parameters(eta=ScalarParam(1, unit="1", description="Moore's term"))
parameters(Q=ScalarParam(1, unit="1", description="Q10 factor"))
parameters(gamma=ScalarParam(1, unit="1", description="Damage dependent factor"))

2. Include them in the potential and gating variable equations (e.g. for Mitchell Schaeffer)
expressions("membrane")
dVm_dt = J_stim_J_stim + eta*gamma*(J_in_J_in + J_out_J_out)

expressions("J_in_h_gate")
dh_dt = Conditional(Gt(V_gate, Vm), Q*(1.0 - h)/tau_open, Q*(-h)/tau_close)

3. Generate the .h as shown in the previous section

In the .h
4. Define the ionic model as a GotranxODEModelWithThermalDamage class
This automatically retrieves the indices
                eta_idx = parameter_index("eta");
                gamma_idx = parameter_index("gamma");
                Q_idx = parameter_index("Q");
but they can also be overridden if needed (e.g. if the name of the variables has been changed).

5. In the constructor define the vector of indices associated with the time constants that will need to be 
   updated for damage dependency (this is because each time constant might have a different delta_tau value)

Therefore in a general simulation, if one wants to use a temperature/damage model needs to simply choose the appropriate TD-dependent model whe initializing the ReactionSolver, and provide a GridFunction for Temperature and Damage. 
if nullptr is provided then dependency is not considered and the associated scaling factors do not vary (if both are nullptr, then the model acts as the original ionic model). 



