#pragma once

#include "gotranx_wrapper.hpp"

namespace mfem
{
    namespace electrophysiology
    {

        // Gotran generated C/C++ code for the "mitchell_schaeffer_2003" model (converted from CellML)
        class MitchellSchaeffer : public GotranxODEModel
        {
        public:
            // Constructor to initialize the base class metadata
            MitchellSchaeffer() : GotranxODEModel()
            {
                NUM_STATES = Kernel::nstates;
                NUM_PARAMS = Kernel::nparams;
                NUM_MONITORED = Kernel::nmonitored;

                potential_idx = state_index("Vm");
                stim_ampl_idx = parameter_index("IstimAmplitude");
                stim_duration_idx = parameter_index("IstimPulseDuration");
                stim_start_idx = parameter_index("IstimStart");
                stim_end_idx = parameter_index("IstimEnd");
                stim_period_idx = parameter_index("IstimPeriod");

                // The device kernel hard-codes these as compile-time constants; keep
                // the name-lookup path and the constants from drifting apart.
                MFEM_ASSERT(potential_idx == Kernel::potential_idx, "potential_idx mismatch");
                MFEM_ASSERT(stim_ampl_idx == Kernel::stim_ampl_idx, "stim_ampl_idx mismatch");

                dimensionless = Kernel::dimensionless; // Mitchell-Schaeffer model uses dimensionless potential
                stim_sign = Kernel::stim_sign;
            }

            // Set stimulation parameters - accepts variable number of parameters
            /*template <typename... Args>
            void SetStimulationParameters(Args... args) TODO: add param array
            {
                static_assert(sizeof...(args) == 4, "MitchellSchaeffer::SetStimulationParameters() "
                "requires exactly 4 arguments: IstimEnd, IstimPeriod, IstimPulseDuration, IstimStart");

                // Unpack the arguments into a tuple
                double values[] = {static_cast<double>(args)...};
                param_array[parameter_index("IstimEnd")] = values[0];
                param_array[parameter_index("IstimPeriod")] = values[1];
                param_array[parameter_index("IstimPulseDuration")] = values[2];
                param_array[parameter_index("IstimStart")] = values[3];
            }*/

            // Parameter index
            int parameter_index(const char name[])
            {

                if (strcmp(name, "IstimAmplitude") == 0)
                {
                    return 0;
                }

                else if (strcmp(name, "IstimEnd") == 0)
                {
                    return 1;
                }

                else if (strcmp(name, "IstimPeriod") == 0)
                {
                    return 2;
                }

                else if (strcmp(name, "IstimPulseDuration") == 0)
                {
                    return 3;
                }

                else if (strcmp(name, "IstimStart") == 0)
                {
                    return 4;
                }

                else if (strcmp(name, "V_gate") == 0)
                {
                    return 5;
                }

                else if (strcmp(name, "tau_close") == 0)
                {
                    return 6;
                }

                else if (strcmp(name, "tau_in") == 0)
                {
                    return 7;
                }

                else if (strcmp(name, "tau_open") == 0)
                {
                    return 8;
                }

                else if (strcmp(name, "tau_out") == 0)
                {
                    return 9;
                }

                return -1;
            }
            // State index
            int state_index(const char name[])
            {

                if (strcmp(name, "h") == 0)
                {
                    return 0;
                }

                else if (strcmp(name, "Vm") == 0)
                {
                    return 1;
                }

                return -1;
            }
            // Monitor index
            int monitor_index(const char name[])
            {

                if (strcmp(name, "J_in_J_in") == 0)
                {
                    return 0;
                }

                else if (strcmp(name, "J_out_J_out") == 0)
                {
                    return 1;
                }

                else if (strcmp(name, "J_stim_J_stim") == 0)
                {
                    return 2;
                }

                else if (strcmp(name, "dh_dt") == 0)
                {
                    return 3;
                }

                else if (strcmp(name, "dVm_dt") == 0)
                {
                    return 4;
                }

                return -1;
            }

            void init_parameter_values(double *parameters)
            {
                /*
                IstimAmplitude=0.2, IstimEnd=50000.0, IstimPeriod=500.0, IstimPulseDuration=1.0, IstimStart=0.0, V_gate=0.13,
                tau_close=150.0, tau_in=0.3, tau_open=120.0, tau_out=6.0
                */
                parameters[0] = 0.2;
                parameters[1] = 50000.0;
                parameters[2] = 500.0;
                parameters[3] = 1.0;
                parameters[4] = 0.0;
                parameters[5] = 0.13;
                parameters[6] = 150.0;
                parameters[7] = 0.3;
                parameters[8] = 120.0;
                parameters[9] = 6.0;
            }

            void init_state_values(double *states)
            {
                /*
                h=0.8789655121804799, Vm=8.20413566106744e-06
                */
                states[0] = 0.8789655121804799;
                states[1] = 8.20413566106744e-06;
            }

            /**
             * @brief Device-callable kernel for this model.
             *
             * Holds the model math as MFEM_HOST_DEVICE *static* functions and the
             * model metadata as compile-time constants, so that ReactionSolver can
             * instantiate a templated mfem::forall kernel over it with no virtual
             * dispatch and full inlining. The virtual methods of the enclosing class
             * forward here, so the math has a single source of truth.
             */
            struct Kernel : IonicKernelDefaults
            {
                static constexpr int nstates = 2;
                static constexpr int nparams = 10;
                static constexpr int nmonitored = 5;

                static constexpr int potential_idx = 1;   // "Vm"
                static constexpr int stim_ampl_idx = 0;   // "IstimAmplitude"

                static constexpr bool dimensionless = true;
                static constexpr real_t stim_sign = 1.0;

            MFEM_HOST_DEVICE static void rhs(const double t, const double *__restrict states, const double *__restrict parameters, double *values)
            {

                // Assign states
                const double h = states[0];
                const double Vm = states[1];

                // Assign parameters
                const double IstimAmplitude = parameters[0];
                const double IstimEnd = parameters[1];
                const double IstimPeriod = parameters[2];
                const double IstimPulseDuration = parameters[3];
                const double IstimStart = parameters[4];
                const double V_gate = parameters[5];
                const double tau_close = parameters[6];
                const double tau_in = parameters[7];
                const double tau_open = parameters[8];
                const double tau_out = parameters[9];

                // Assign expressions
                const double J_in_J_in = (h * (pow(Vm, 2.0) * (1.0 - Vm))) / tau_in;
                const double J_out_J_out = (-Vm) / tau_out;
                const double J_stim_J_stim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double dh_dt = (V_gate > Vm) ? (1.0 - h) / tau_open : (-h) / tau_close;
                values[0] = dh_dt;
                const double dVm_dt = J_stim_J_stim + (J_in_J_in + J_out_J_out);
                values[1] = dVm_dt;
            }

            MFEM_HOST_DEVICE static void monitor_values(const double t, const double *__restrict states, const double *__restrict parameters,
                                double *values)
            {

                // Assign states
                const double h = states[0];
                const double Vm = states[1];

                // Assign parameters
                const double IstimAmplitude = parameters[0];
                const double IstimEnd = parameters[1];
                const double IstimPeriod = parameters[2];
                const double IstimPulseDuration = parameters[3];
                const double IstimStart = parameters[4];
                const double V_gate = parameters[5];
                const double tau_close = parameters[6];
                const double tau_in = parameters[7];
                const double tau_open = parameters[8];
                const double tau_out = parameters[9];

                // Assign expressions
                const double J_in_J_in = (h * (pow(Vm, 2.0) * (1.0 - Vm))) / tau_in;
                values[0] = J_in_J_in;
                const double J_out_J_out = (-Vm) / tau_out;
                values[1] = J_out_J_out;
                const double J_stim_J_stim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                values[2] = J_stim_J_stim;
                const double dh_dt = (V_gate > Vm) ? (1.0 - h) / tau_open : (-h) / tau_close;
                values[3] = dh_dt;
                const double dVm_dt = J_stim_J_stim + (J_in_J_in + J_out_J_out);
                values[4] = dVm_dt;
            }

            MFEM_HOST_DEVICE static void explicit_euler(const double *__restrict states, const double t, const double dt,
                                const double *__restrict parameters, double *values)
            {

                // Assign states
                const double h = states[0];
                const double Vm = states[1];

                // Assign parameters
                const double IstimAmplitude = parameters[0];
                const double IstimEnd = parameters[1];
                const double IstimPeriod = parameters[2];
                const double IstimPulseDuration = parameters[3];
                const double IstimStart = parameters[4];
                const double V_gate = parameters[5];
                const double tau_close = parameters[6];
                const double tau_in = parameters[7];
                const double tau_open = parameters[8];
                const double tau_out = parameters[9];

                // Assign expressions
                const double J_in_J_in = (h * (pow(Vm, 2.0) * (1.0 - Vm))) / tau_in;
                const double J_out_J_out = (-Vm) / tau_out;
                const double J_stim_J_stim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double dh_dt = (V_gate > Vm) ? (1.0 - h) / tau_open : (-h) / tau_close;
                values[0] = dh_dt * dt + h;
                const double dVm_dt = J_stim_J_stim + (J_in_J_in + J_out_J_out);
                values[1] = Vm + dVm_dt * dt;
            }

            MFEM_HOST_DEVICE static void generalized_rush_larsen(const double *__restrict states, const double t, const double dt,
                                         const double *__restrict parameters, double *values)
            {

                // Assign states
                const double h = states[0];
                const double Vm = states[1];

                // Assign parameters
                const double IstimAmplitude = parameters[0];
                const double IstimEnd = parameters[1];
                const double IstimPeriod = parameters[2];
                const double IstimPulseDuration = parameters[3];
                const double IstimStart = parameters[4];
                const double V_gate = parameters[5];
                const double tau_close = parameters[6];
                const double tau_in = parameters[7];
                const double tau_open = parameters[8];
                const double tau_out = parameters[9];

                // Assign expressions
                const double J_in_J_in = (h * (pow(Vm, 2.0) * (1.0 - Vm))) / tau_in;
                const double J_out_J_out = (-Vm) / tau_out;
                const double J_stim_J_stim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double dh_dt = (V_gate > Vm) ? (1.0 - h) / tau_open : (-h) / tau_close;
                const double dh_dt_linearized = (V_gate > Vm) ? -1 / tau_open : -1 / tau_close;
                values[0] = h + ((fabs(dh_dt_linearized) > 1e-08) ? (dh_dt * (exp(dh_dt_linearized * dt) - 1) / dh_dt_linearized)
                                                                  : (dh_dt * dt));
                const double dVm_dt = (J_stim_J_stim + (J_in_J_in + J_out_J_out));
                values[1] = Vm + dVm_dt * dt;
            }
            }; // struct Kernel

        };

    } // namespace electrophysiology
} // namespace mfem