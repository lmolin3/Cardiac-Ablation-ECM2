#pragma once

// disable -Wunused-variable for compiling this code
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "gotranx_wrapper.hpp"

namespace mfem
{
    namespace electrophysiology
    {

        // Gotran generated C/C++ code for the "fenton_karma_1998" model (converted from CellML, MBR version)
        class FentonKarma : public GotranxODEModel
        {
        public:
            // Constructor to initialize the base class metadata
            FentonKarma() : GotranxODEModel()
            {
                NUM_STATES = 3;
                NUM_PARAMS = 21;
                NUM_MONITORED = 12;

                potential_idx = state_index("u");
                stim_ampl_idx = parameter_index("IstimAmplitude");
                stim_duration_idx = parameter_index("IstimPulseDuration");
                stim_start_idx = parameter_index("IstimStart");
                stim_end_idx = parameter_index("IstimEnd");
                stim_period_idx = parameter_index("IstimPeriod");

                stim_sign = -1; // Fenton-Karma model uses negative stimulation current

                dimensionless = true; // Fenton-Karma model is dimensionless
            }

            // Parameter index
            int parameter_index(const char name[])
            {

                if (strcmp(name, "Cm") == 0)
                {
                    return 0;
                }

                else if (strcmp(name, "IstimAmplitude") == 0)
                {
                    return 1;
                }

                else if (strcmp(name, "IstimEnd") == 0)
                {
                    return 2;
                }

                else if (strcmp(name, "IstimPeriod") == 0)
                {
                    return 3;
                }

                else if (strcmp(name, "IstimPulseDuration") == 0)
                {
                    return 4;
                }

                else if (strcmp(name, "IstimStart") == 0)
                {
                    return 5;
                }

                else if (strcmp(name, "V_0") == 0)
                {
                    return 6;
                }

                else if (strcmp(name, "V_fi") == 0)
                {
                    return 7;
                }

                else if (strcmp(name, "g_fi_max") == 0)
                {
                    return 8;
                }

                else if (strcmp(name, "k") == 0)
                {
                    return 9;
                }

                else if (strcmp(name, "tau_0") == 0)
                {
                    return 10;
                }

                else if (strcmp(name, "tau_r") == 0)
                {
                    return 11;
                }

                else if (strcmp(name, "tau_si") == 0)
                {
                    return 12;
                }

                else if (strcmp(name, "tau_v1_minus") == 0)
                {
                    return 13;
                }

                else if (strcmp(name, "tau_v2_minus") == 0)
                {
                    return 14;
                }

                else if (strcmp(name, "tau_v_plus") == 0)
                {
                    return 15;
                }

                else if (strcmp(name, "tau_w_minus") == 0)
                {
                    return 16;
                }

                else if (strcmp(name, "tau_w_plus") == 0)
                {
                    return 17;
                }

                else if (strcmp(name, "u_c") == 0)
                {
                    return 18;
                }

                else if (strcmp(name, "u_csi") == 0)
                {
                    return 19;
                }

                else if (strcmp(name, "u_v") == 0)
                {
                    return 20;
                }

                return -1;
            }
            // State index
            int state_index(const char name[])
            {

                if (strcmp(name, "w") == 0)
                {
                    return 0;
                }

                else if (strcmp(name, "v") == 0)
                {
                    return 1;
                }

                else if (strcmp(name, "u") == 0)
                {
                    return 2;
                }

                return -1;
            }
            // Monitor index
            int monitor_index(const char name[])
            {

                if (strcmp(name, "Istim") == 0)
                {
                    return 0;
                }

                else if (strcmp(name, "p_p") == 0)
                {
                    return 1;
                }

                else if (strcmp(name, "J_si") == 0)
                {
                    return 2;
                }

                else if (strcmp(name, "Vm") == 0)
                {
                    return 3;
                }

                else if (strcmp(name, "q_q") == 0)
                {
                    return 4;
                }

                else if (strcmp(name, "tau_d") == 0)
                {
                    return 5;
                }

                else if (strcmp(name, "J_so") == 0)
                {
                    return 6;
                }

                else if (strcmp(name, "dw_dt") == 0)
                {
                    return 7;
                }

                else if (strcmp(name, "tau_v_minus") == 0)
                {
                    return 8;
                }

                else if (strcmp(name, "J_fi") == 0)
                {
                    return 9;
                }

                else if (strcmp(name, "dv_dt") == 0)
                {
                    return 10;
                }

                else if (strcmp(name, "du_dt") == 0)
                {
                    return 11;
                }

                return -1;
            }

            void init_parameter_values(double *parameters)
            {
                /*
                Cm=1.0, IstimAmplitude=-0.2, IstimEnd=50000.0, IstimPeriod=1000.0, IstimPulseDuration=1.0, IstimStart=10.0,
                V_0=-85.0, V_fi=15.0, g_fi_max=4.0, k=10.0, tau_0=8.3, tau_r=50.0, tau_si=44.84, tau_v1_minus=1000.0,
                tau_v2_minus=19.2, tau_v_plus=3.33, tau_w_minus=11.0, tau_w_plus=667.0, u_c=0.13, u_csi=0.85, u_v=0.055
                */
                parameters[0] = 1.0;
                parameters[1] = -0.2;
                parameters[2] = 50000.0;
                parameters[3] = 1000.0;
                parameters[4] = 1.0;
                parameters[5] = 10.0;
                parameters[6] = -85.0;
                parameters[7] = 15.0;
                parameters[8] = 4.0;
                parameters[9] = 10.0;
                parameters[10] = 8.3;
                parameters[11] = 50.0;
                parameters[12] = 44.84;
                parameters[13] = 1000.0;
                parameters[14] = 19.2;
                parameters[15] = 3.33;
                parameters[16] = 11.0;
                parameters[17] = 667.0;
                parameters[18] = 0.13;
                parameters[19] = 0.85;
                parameters[20] = 0.055;
            }

            void init_state_values(double *states)
            {
                /*
                w=1, v=1, u=0
                */
                states[0] = 1;
                states[1] = 1;
                states[2] = 0;
            }

            void rhs(const double t, const double *__restrict states, const double *__restrict parameters, double *values)
            {

                // Assign states
                const double w = states[0];
                const double v = states[1];
                const double u = states[2];

                // Assign parameters
                const double Cm = parameters[0];
                const double IstimAmplitude = parameters[1];
                const double IstimEnd = parameters[2];
                const double IstimPeriod = parameters[3];
                const double IstimPulseDuration = parameters[4];
                const double IstimStart = parameters[5];
                const double V_0 = parameters[6];
                const double V_fi = parameters[7];
                const double g_fi_max = parameters[8];
                const double k = parameters[9];
                const double tau_0 = parameters[10];
                const double tau_r = parameters[11];
                const double tau_si = parameters[12];
                const double tau_v1_minus = parameters[13];
                const double tau_v2_minus = parameters[14];
                const double tau_v_plus = parameters[15];
                const double tau_w_minus = parameters[16];
                const double tau_w_plus = parameters[17];
                const double u_c = parameters[18];
                const double u_csi = parameters[19];
                const double u_v = parameters[20];

                // Assign expressions
                const double Istim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double p_p = (u < u_c) ? 0.0 : 1.0;
                const double J_si =
                    ((-w) * (1.0 + (exp(2.0 * (k * (u - u_csi))) - 1 * 1.0) / (exp(2.0 * (k * (u - u_csi))) + 1.0))) /
                    ((2.0 * tau_si));
                const double q_q = (u < u_v) ? 0.0 : 1.0;
                const double tau_d = Cm / g_fi_max;
                const double J_so = p_p / tau_r + (u * (1.0 - p_p)) / tau_0;
                const double dw_dt = ((-p_p) * w) / tau_w_plus + ((1.0 - p_p) * (1.0 - w)) / tau_w_minus;
                values[0] = dw_dt;
                const double tau_v_minus = q_q * tau_v1_minus + tau_v2_minus * (1.0 - q_q);
                const double J_fi = (((p_p * (-v)) * (1.0 - u)) * (u - u_c)) / tau_d;
                const double dv_dt = ((-p_p) * v) / tau_v_plus + ((1.0 - p_p) * (1.0 - v)) / tau_v_minus;
                values[1] = dv_dt;
                const double du_dt = -(Istim + (J_si + (J_fi + J_so)));
                values[2] = du_dt;
            }

            void monitor_values(const double t, const double *__restrict states, const double *__restrict parameters,
                                double *values)
            {

                // Assign states
                const double w = states[0];
                const double v = states[1];
                const double u = states[2];

                // Assign parameters
                const double Cm = parameters[0];
                const double IstimAmplitude = parameters[1];
                const double IstimEnd = parameters[2];
                const double IstimPeriod = parameters[3];
                const double IstimPulseDuration = parameters[4];
                const double IstimStart = parameters[5];
                const double V_0 = parameters[6];
                const double V_fi = parameters[7];
                const double g_fi_max = parameters[8];
                const double k = parameters[9];
                const double tau_0 = parameters[10];
                const double tau_r = parameters[11];
                const double tau_si = parameters[12];
                const double tau_v1_minus = parameters[13];
                const double tau_v2_minus = parameters[14];
                const double tau_v_plus = parameters[15];
                const double tau_w_minus = parameters[16];
                const double tau_w_plus = parameters[17];
                const double u_c = parameters[18];
                const double u_csi = parameters[19];
                const double u_v = parameters[20];

                // Assign expressions
                const double Istim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                values[0] = Istim;
                const double p_p = (u < u_c) ? 0.0 : 1.0;
                values[1] = p_p;
                const double J_si =
                    ((-w) * (1.0 + (exp(2.0 * (k * (u - u_csi))) - 1 * 1.0) / (exp(2.0 * (k * (u - u_csi))) + 1.0))) /
                    ((2.0 * tau_si));
                values[2] = J_si;
                const double Vm = V_0 + u * (-V_0 + V_fi);
                values[3] = Vm;
                const double q_q = (u < u_v) ? 0.0 : 1.0;
                values[4] = q_q;
                const double tau_d = Cm / g_fi_max;
                values[5] = tau_d;
                const double J_so = p_p / tau_r + (u * (1.0 - p_p)) / tau_0;
                values[6] = J_so;
                const double dw_dt = ((-p_p) * w) / tau_w_plus + ((1.0 - p_p) * (1.0 - w)) / tau_w_minus;
                values[7] = dw_dt;
                const double tau_v_minus = q_q * tau_v1_minus + tau_v2_minus * (1.0 - q_q);
                values[8] = tau_v_minus;
                const double J_fi = (((p_p * (-v)) * (1.0 - u)) * (u - u_c)) / tau_d;
                values[9] = J_fi;
                const double dv_dt = ((-p_p) * v) / tau_v_plus + ((1.0 - p_p) * (1.0 - v)) / tau_v_minus;
                values[10] = dv_dt;
                const double du_dt = -(Istim + (J_si + (J_fi + J_so)));
                values[11] = du_dt;
            }

            void explicit_euler(const double *__restrict states, const double t, const double dt,
                                const double *__restrict parameters, double *values)
            {

                // Assign states
                const double w = states[0];
                const double v = states[1];
                const double u = states[2];

                // Assign parameters
                const double Cm = parameters[0];
                const double IstimAmplitude = parameters[1];
                const double IstimEnd = parameters[2];
                const double IstimPeriod = parameters[3];
                const double IstimPulseDuration = parameters[4];
                const double IstimStart = parameters[5];
                const double V_0 = parameters[6];
                const double V_fi = parameters[7];
                const double g_fi_max = parameters[8];
                const double k = parameters[9];
                const double tau_0 = parameters[10];
                const double tau_r = parameters[11];
                const double tau_si = parameters[12];
                const double tau_v1_minus = parameters[13];
                const double tau_v2_minus = parameters[14];
                const double tau_v_plus = parameters[15];
                const double tau_w_minus = parameters[16];
                const double tau_w_plus = parameters[17];
                const double u_c = parameters[18];
                const double u_csi = parameters[19];
                const double u_v = parameters[20];

                // Assign expressions
                const double Istim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double p_p = (u < u_c) ? 0.0 : 1.0;
                const double J_si =
                    ((-w) * (1.0 + (exp(2.0 * (k * (u - u_csi))) - 1 * 1.0) / (exp(2.0 * (k * (u - u_csi))) + 1.0))) /
                    ((2.0 * tau_si));
                const double q_q = (u < u_v) ? 0.0 : 1.0;
                const double tau_d = Cm / g_fi_max;
                const double J_so = p_p / tau_r + (u * (1.0 - p_p)) / tau_0;
                const double dw_dt = ((-p_p) * w) / tau_w_plus + ((1.0 - p_p) * (1.0 - w)) / tau_w_minus;
                values[0] = dt * dw_dt + w;
                const double tau_v_minus = q_q * tau_v1_minus + tau_v2_minus * (1.0 - q_q);
                const double J_fi = (((p_p * (-v)) * (1.0 - u)) * (u - u_c)) / tau_d;
                const double dv_dt = ((-p_p) * v) / tau_v_plus + ((1.0 - p_p) * (1.0 - v)) / tau_v_minus;
                values[1] = dt * dv_dt + v;
                const double du_dt = -(Istim + (J_si + (J_fi + J_so)));
                values[2] = dt * du_dt + u;
            }

            void generalized_rush_larsen(const double *__restrict states, const double t, const double dt,
                                         const double *__restrict parameters, double *values)
            {

                // Assign states
                const double w = states[0];
                const double v = states[1];
                const double u = states[2];

                // Assign parameters
                const double Cm = parameters[0];
                const double IstimAmplitude = parameters[1];
                const double IstimEnd = parameters[2];
                const double IstimPeriod = parameters[3];
                const double IstimPulseDuration = parameters[4];
                const double IstimStart = parameters[5];
                const double V_0 = parameters[6];
                const double V_fi = parameters[7];
                const double g_fi_max = parameters[8];
                const double k = parameters[9];
                const double tau_0 = parameters[10];
                const double tau_r = parameters[11];
                const double tau_si = parameters[12];
                const double tau_v1_minus = parameters[13];
                const double tau_v2_minus = parameters[14];
                const double tau_v_plus = parameters[15];
                const double tau_w_minus = parameters[16];
                const double tau_w_plus = parameters[17];
                const double u_c = parameters[18];
                const double u_csi = parameters[19];
                const double u_v = parameters[20];

                // Assign expressions
                const double Istim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double p_p = (u < u_c) ? 0.0 : 1.0;
                const double J_si =
                    ((-w) * (1.0 + (exp(2.0 * (k * (u - u_csi))) - 1 * 1.0) / (exp(2.0 * (k * (u - u_csi))) + 1.0))) /
                    ((2.0 * tau_si));
                const double q_q = (u < u_v) ? 0.0 : 1.0;
                const double tau_d = Cm / g_fi_max;
                const double J_so = p_p / tau_r + (u * (1.0 - p_p)) / tau_0;
                const double dw_dt = ((-p_p) * w) / tau_w_plus + ((1.0 - p_p) * (1.0 - w)) / tau_w_minus;
                const double dw_dt_linearized = -p_p / tau_w_plus + (p_p - 1.0) / tau_w_minus;
                values[0] = w + ((fabs(dw_dt_linearized) > 1e-08) ? (dw_dt * (exp(dt * dw_dt_linearized) - 1) / dw_dt_linearized)
                                                                  : (dt * dw_dt));
                const double tau_v_minus = q_q * tau_v1_minus + tau_v2_minus * (1.0 - q_q);
                const double J_fi = (((p_p * (-v)) * (1.0 - u)) * (u - u_c)) / tau_d;
                const double dv_dt = ((-p_p) * v) / tau_v_plus + ((1.0 - p_p) * (1.0 - v)) / tau_v_minus;
                const double dv_dt_linearized = -p_p / tau_v_plus + (p_p - 1.0) / tau_v_minus;
                values[1] = v + ((fabs(dv_dt_linearized) > 1e-08) ? (dv_dt * (exp(dt * dv_dt_linearized) - 1) / dv_dt_linearized)
                                                                  : (dt * dv_dt));
                const double du_dt = -(Istim + (J_si + (J_fi + J_so)));
                values[2] = dt * du_dt + u;
            }

            void forward_explicit_euler(const double *__restrict states, const double t, const double dt,
                                        const double *__restrict parameters, double *values)
            {

                // Assign states
                const double w = states[0];
                const double v = states[1];
                const double u = states[2];

                // Assign parameters
                const double Cm = parameters[0];
                const double IstimAmplitude = parameters[1];
                const double IstimEnd = parameters[2];
                const double IstimPeriod = parameters[3];
                const double IstimPulseDuration = parameters[4];
                const double IstimStart = parameters[5];
                const double V_0 = parameters[6];
                const double V_fi = parameters[7];
                const double g_fi_max = parameters[8];
                const double k = parameters[9];
                const double tau_0 = parameters[10];
                const double tau_r = parameters[11];
                const double tau_si = parameters[12];
                const double tau_v1_minus = parameters[13];
                const double tau_v2_minus = parameters[14];
                const double tau_v_plus = parameters[15];
                const double tau_w_minus = parameters[16];
                const double tau_w_plus = parameters[17];
                const double u_c = parameters[18];
                const double u_csi = parameters[19];
                const double u_v = parameters[20];

                // Assign expressions
                const double Istim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double p_p = (u < u_c) ? 0.0 : 1.0;
                const double J_si =
                    ((-w) * (1.0 + (exp(2.0 * (k * (u - u_csi))) - 1 * 1.0) / (exp(2.0 * (k * (u - u_csi))) + 1.0))) /
                    ((2.0 * tau_si));
                const double q_q = (u < u_v) ? 0.0 : 1.0;
                const double tau_d = Cm / g_fi_max;
                const double J_so = p_p / tau_r + (u * (1.0 - p_p)) / tau_0;
                const double dw_dt = ((-p_p) * w) / tau_w_plus + ((1.0 - p_p) * (1.0 - w)) / tau_w_minus;
                values[0] = dt * dw_dt + w;
                const double tau_v_minus = q_q * tau_v1_minus + tau_v2_minus * (1.0 - q_q);
                const double J_fi = (((p_p * (-v)) * (1.0 - u)) * (u - u_c)) / tau_d;
                const double dv_dt = ((-p_p) * v) / tau_v_plus + ((1.0 - p_p) * (1.0 - v)) / tau_v_minus;
                values[1] = dt * dv_dt + v;
                const double du_dt = -(Istim + (J_si + (J_fi + J_so)));
                values[2] = dt * du_dt + u;
            }

            void forward_generalized_rush_larsen(const double *__restrict states, const double t, const double dt,
                                                 const double *__restrict parameters, double *values)
            {

                // Assign states
                const double w = states[0];
                const double v = states[1];
                const double u = states[2];

                // Assign parameters
                const double Cm = parameters[0];
                const double IstimAmplitude = parameters[1];
                const double IstimEnd = parameters[2];
                const double IstimPeriod = parameters[3];
                const double IstimPulseDuration = parameters[4];
                const double IstimStart = parameters[5];
                const double V_0 = parameters[6];
                const double V_fi = parameters[7];
                const double g_fi_max = parameters[8];
                const double k = parameters[9];
                const double tau_0 = parameters[10];
                const double tau_r = parameters[11];
                const double tau_si = parameters[12];
                const double tau_v1_minus = parameters[13];
                const double tau_v2_minus = parameters[14];
                const double tau_v_plus = parameters[15];
                const double tau_w_minus = parameters[16];
                const double tau_w_plus = parameters[17];
                const double u_c = parameters[18];
                const double u_csi = parameters[19];
                const double u_v = parameters[20];

                // Assign expressions
                const double Istim =
                    (IstimEnd >= t &&
                     IstimPulseDuration >= t + ((-IstimPeriod) * floor((-(IstimStart - t)) / IstimPeriod) - IstimStart) &&
                     IstimStart <= t)
                        ? IstimAmplitude
                        : 0.0;
                const double p_p = (u < u_c) ? 0.0 : 1.0;
                const double J_si =
                    ((-w) * (1.0 + (exp(2.0 * (k * (u - u_csi))) - 1 * 1.0) / (exp(2.0 * (k * (u - u_csi))) + 1.0))) /
                    ((2.0 * tau_si));
                const double q_q = (u < u_v) ? 0.0 : 1.0;
                const double tau_d = Cm / g_fi_max;
                const double J_so = p_p / tau_r + (u * (1.0 - p_p)) / tau_0;
                const double dw_dt = ((-p_p) * w) / tau_w_plus + ((1.0 - p_p) * (1.0 - w)) / tau_w_minus;
                const double dw_dt_linearized = -p_p / tau_w_plus + (p_p - 1.0) / tau_w_minus;
                values[0] = w + ((fabs(dw_dt_linearized) > 1e-08) ? (dw_dt * (exp(dt * dw_dt_linearized) - 1) / dw_dt_linearized)
                                                                  : (dt * dw_dt));
                const double tau_v_minus = q_q * tau_v1_minus + tau_v2_minus * (1.0 - q_q);
                const double J_fi = (((p_p * (-v)) * (1.0 - u)) * (u - u_c)) / tau_d;
                const double dv_dt = ((-p_p) * v) / tau_v_plus + ((1.0 - p_p) * (1.0 - v)) / tau_v_minus;
                const double dv_dt_linearized = -p_p / tau_v_plus + (p_p - 1.0) / tau_v_minus;
                values[1] = v + ((fabs(dv_dt_linearized) > 1e-08) ? (dv_dt * (exp(dt * dv_dt_linearized) - 1) / dv_dt_linearized)
                                                                  : (dt * dv_dt));
                const double du_dt = -(Istim + (J_si + (J_fi + J_so)));
                values[2] = dt * du_dt + u;
            }

        }; // class FentonKarma

    } // namespace electrophysiology
} // namespace mfem