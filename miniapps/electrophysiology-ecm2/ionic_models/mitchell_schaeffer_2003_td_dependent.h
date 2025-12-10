#pragma once

#include "gotranx_wrapper.hpp"

namespace mfem
{
    namespace electrophysiology
    {

        // Gotran generated C/C++ code for the "mitchell_schaeffer_2003" model (converted from CellML)
        class MitchellSchaefferTD : public GotranxODEModelWithThermalDamage
        {
        public:
            // Constructor to initialize the base class metadata
            MitchellSchaefferTD() : GotranxODEModelWithThermalDamage()
            {
                NUM_STATES = 2;
                NUM_PARAMS = 13;
                NUM_MONITORED = 5;

                potential_idx = state_index("Vm");
                stim_ampl_idx = parameter_index("IstimAmplitude");
                stim_duration_idx = parameter_index("IstimPulseDuration");
                stim_start_idx = parameter_index("IstimStart");
                stim_end_idx = parameter_index("IstimEnd");
                stim_period_idx = parameter_index("IstimPeriod");

                dimensionless = true; // Mitchell-Schaeffer model uses dimensionless potential

                InitializeParameterIndices();

                // Time constants indices for temperature/damage dependency
                time_constants_idxs.push_back(parameter_index("tau_close"));
                time_constants_idxs.push_back(parameter_index("tau_in"));
                time_constants_idxs.push_back(parameter_index("tau_open"));
                time_constants_idxs.push_back(parameter_index("tau_out"));
            }

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

                else if (strcmp(name, "Q") == 0)
                {
                    return 5;
                }

                else if (strcmp(name, "V_gate") == 0)
                {
                    return 6;
                }

                else if (strcmp(name, "eta") == 0)
                {
                    return 7;
                }

                else if (strcmp(name, "gamma") == 0)
                {
                    return 8;
                }

                else if (strcmp(name, "tau_close") == 0)
                {
                    return 9;
                }

                else if (strcmp(name, "tau_in") == 0)
                {
                    return 10;
                }

                else if (strcmp(name, "tau_open") == 0)
                {
                    return 11;
                }

                else if (strcmp(name, "tau_out") == 0)
                {
                    return 12;
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
                IstimAmplitude=0.2, IstimEnd=50000.0, IstimPeriod=500.0, IstimPulseDuration=1.0, IstimStart=0.0, Q=1.0, V_gate=0.13,
                eta=1.0, gamma=1.0, tau_close=150.0, tau_in=0.3, tau_open=120.0, tau_out=6.0
                */
                parameters[0] = 0.2;
                parameters[1] = 50000.0;
                parameters[2] = 500.0;
                parameters[3] = 1.0;
                parameters[4] = 0.0;
                parameters[5] = 1.0;
                parameters[6] = 0.13;
                parameters[7] = 1.0;
                parameters[8] = 1.0;
                parameters[9] = 150.0;
                parameters[10] = 0.3;
                parameters[11] = 120.0;
                parameters[12] = 6.0;
            }

            void init_state_values(double *states)
            {
                /*
                h=0.8789655121804799, Vm=8.20413566106744e-06
                */
                states[0] = 0.8789655121804799;
                states[1] = 8.20413566106744e-06;
            }

            void rhs(const double t, const double *__restrict states, const double *__restrict parameters, double *values)
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
                const double Q = parameters[5];
                const double V_gate = parameters[6];
                const double eta = parameters[7];
                const double gamma = parameters[8];
                const double tau_close = parameters[9];
                const double tau_in = parameters[10];
                const double tau_open = parameters[11];
                const double tau_out = parameters[12];

                // Assign expressions
                const double J_in_J_in = (h * (pow(Vm, 2.0) * (1.0 - Vm))) / tau_in;
                const double J_out_J_out = (-Vm) / tau_out;
                const double J_stim_J_stim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double dh_dt = (V_gate > Vm) ? (Q * (1.0 - h)) / tau_open : (Q * (-h)) / tau_close;
                values[0] = dh_dt;
                const double dVm_dt = J_stim_J_stim + (eta * gamma) * (J_in_J_in + J_out_J_out);
                values[1] = dVm_dt;
            }

            void monitor_values(const double t, const double *__restrict states, const double *__restrict parameters,
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
                const double Q = parameters[5];
                const double V_gate = parameters[6];
                const double eta = parameters[7];
                const double gamma = parameters[8];
                const double tau_close = parameters[9];
                const double tau_in = parameters[10];
                const double tau_open = parameters[11];
                const double tau_out = parameters[12];

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
                const double dh_dt = (V_gate > Vm) ? (Q * (1.0 - h)) / tau_open : (Q * (-h)) / tau_close;
                values[3] = dh_dt;
                const double dVm_dt = J_stim_J_stim + (eta * gamma) * (J_in_J_in + J_out_J_out);
                values[4] = dVm_dt;
            }

            void explicit_euler(const double *__restrict states, const double t, const double dt,
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
                const double Q = parameters[5];
                const double V_gate = parameters[6];
                const double eta = parameters[7];
                const double gamma = parameters[8];
                const double tau_close = parameters[9];
                const double tau_in = parameters[10];
                const double tau_open = parameters[11];
                const double tau_out = parameters[12];

                // Assign expressions
                const double J_in_J_in = (h * (pow(Vm, 2.0) * (1.0 - Vm))) / tau_in;
                const double J_out_J_out = (-Vm) / tau_out;
                const double J_stim_J_stim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double dh_dt = (V_gate > Vm) ? (Q * (1.0 - h)) / tau_open : (Q * (-h)) / tau_close;
                values[0] = dh_dt * dt + h;
                const double dVm_dt = J_stim_J_stim + (eta * gamma) * (J_in_J_in + J_out_J_out);
                values[1] = Vm + dVm_dt * dt;
            }

            void generalized_rush_larsen(const double *__restrict states, const double t, const double dt,
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
                const double Q = parameters[5];
                const double V_gate = parameters[6];
                const double eta = parameters[7];
                const double gamma = parameters[8];
                const double tau_close = parameters[9];
                const double tau_in = parameters[10];
                const double tau_open = parameters[11];
                const double tau_out = parameters[12];

                // Assign expressions
                const double J_in_J_in = (h * (pow(Vm, 2.0) * (1.0 - Vm))) / tau_in;
                const double J_out_J_out = (-Vm) / tau_out;
                const double J_stim_J_stim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double dh_dt = (V_gate > Vm) ? (Q * (1.0 - h)) / tau_open : (Q * (-h)) / tau_close;
                const double dh_dt_linearized = (V_gate > Vm) ? -Q / tau_open : -Q / tau_close;
                values[0] = h + ((fabs(dh_dt_linearized) > 1e-08) ? (dh_dt * (exp(dh_dt_linearized * dt) - 1) / dh_dt_linearized)
                                                                  : (dh_dt * dt));
                const double dVm_dt = J_stim_J_stim + (eta * gamma) * (J_in_J_in + J_out_J_out);
                values[1] = Vm + dVm_dt * dt;
            }

            void forward_explicit_euler(const double *__restrict states, const double t, const double dt,
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
                const double Q = parameters[5];
                const double V_gate = parameters[6];
                const double eta = parameters[7];
                const double gamma = parameters[8];
                const double tau_close = parameters[9];
                const double tau_in = parameters[10];
                const double tau_open = parameters[11];
                const double tau_out = parameters[12];

                // Assign expressions
                const double J_in_J_in = (h * (pow(Vm, 2.0) * (1.0 - Vm))) / tau_in;
                const double J_out_J_out = (-Vm) / tau_out;
                const double J_stim_J_stim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double dh_dt = (V_gate > Vm) ? (Q * (1.0 - h)) / tau_open : (Q * (-h)) / tau_close;
                values[0] = dh_dt * dt + h;
                const double dVm_dt = J_stim_J_stim + (eta * gamma) * (J_in_J_in + J_out_J_out);
                values[1] = Vm + dVm_dt * dt;
            }

            void forward_generalized_rush_larsen(const double *__restrict states, const double t, const double dt,
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
                const double Q = parameters[5];
                const double V_gate = parameters[6];
                const double eta = parameters[7];
                const double gamma = parameters[8];
                const double tau_close = parameters[9];
                const double tau_in = parameters[10];
                const double tau_open = parameters[11];
                const double tau_out = parameters[12];

                // Assign expressions
                const double J_in_J_in = (h * (pow(Vm, 2.0) * (1.0 - Vm))) / tau_in;
                const double J_out_J_out = (-Vm) / tau_out;
                const double J_stim_J_stim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double dh_dt = (V_gate > Vm) ? (Q * (1.0 - h)) / tau_open : (Q * (-h)) / tau_close;
                const double dh_dt_linearized = (V_gate > Vm) ? -Q / tau_open : -Q / tau_close;
                values[0] = h + ((fabs(dh_dt_linearized) > 1e-08) ? (dh_dt * (exp(dh_dt_linearized * dt) - 1) / dh_dt_linearized)
                                                                  : (dh_dt * dt));
                const double dVm_dt = J_stim_J_stim + (eta * gamma) * (J_in_J_in + J_out_J_out);
                values[1] = Vm + dVm_dt * dt;
            }
        };

    } // namespace electrophysiology
} // namespace mfem