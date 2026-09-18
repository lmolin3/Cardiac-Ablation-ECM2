#pragma once

// GENERATED FILE -- do not edit by hand.
//
// Produced by gotranx_to_mfem.py from
//     cellml/tp06_endo.ode  ->  gotranx ode2c  ->  tp06_endo_generated.h
// Regenerate with:
//     python3 gotranx_to_mfem.py --config models.json
//
// ten Tusscher-Panfilov 2006 human ventricular endocardial cell model.

#include "gotranx_wrapper.hpp"

namespace mfem
{
    namespace electrophysiology
    {

        /** @brief ten Tusscher-Panfilov 2006 human ventricular endocardial cell model. */
        class TP06Endo : public EPModelBase
        {
        public:
            TP06Endo() : EPModelBase()
            {
                NUM_STATES = Kernel::nstates;
                NUM_PARAMS = Kernel::nparams;
                NUM_MONITORED = Kernel::nmonitored;

                potential_idx = state_index("V");
                stim_ampl_idx = parameter_index("stim_amplitude");
                stim_duration_idx = parameter_index("stim_duration");
                stim_start_idx = parameter_index("stim_start");
                stim_end_idx = parameter_index("stim_end");
                stim_period_idx = parameter_index("stim_period");

                MFEM_ASSERT(potential_idx == Kernel::potential_idx, "potential_idx mismatch");
                MFEM_ASSERT(stim_ampl_idx == Kernel::stim_ampl_idx, "stim_ampl_idx mismatch");
                MFEM_ASSERT(state_index("Ca_i") == Kernel::calcium_idx, "calcium_idx mismatch");

                dimensionless = Kernel::dimensionless;
                stim_sign = Kernel::stim_sign;

                // TP06 is physiological: the ODE emits millivolts directly, so this range is
                // only ReactionSolver's blow-up guard and must never bind on a valid solution.
                // It is set just outside the Nernst potentials, which bracket what the model
                // can physically reach: E_K is about -86 mV and E_Na about +75 mV, against a
                // resting potential of -85.2 mV and an upstroke peak converging to about
                // +37 mV. Anything outside these rails is a blow-up, not an action potential.
                Vmin_default = -95.0;
                Vmax_default = 80.0;
            }

            std::string GetName() const override { return "ten Tusscher-Panfilov 2006 (endocardial)"; }

            int GetPotentialIndex() const override { return Kernel::potential_idx; }

            bool HasCalcium() const override { return true; }
            int GetCalciumIndex() const override { return Kernel::calcium_idx; }

            int parameter_index(const char name[])
            {

                if (strcmp(name, "Buf_c") == 0) {
                    return 0;
                }


                else if (strcmp(name, "Buf_sr") == 0) {
                    return 1;
                }


                else if (strcmp(name, "Buf_ss") == 0) {
                    return 2;
                }


                else if (strcmp(name, "Ca_o") == 0) {
                    return 3;
                }


                else if (strcmp(name, "Cm") == 0) {
                    return 4;
                }


                else if (strcmp(name, "EC") == 0) {
                    return 5;
                }


                else if (strcmp(name, "F") == 0) {
                    return 6;
                }


                else if (strcmp(name, "K_NaCa") == 0) {
                    return 7;
                }


                else if (strcmp(name, "K_buf_c") == 0) {
                    return 8;
                }


                else if (strcmp(name, "K_buf_sr") == 0) {
                    return 9;
                }


                else if (strcmp(name, "K_buf_ss") == 0) {
                    return 10;
                }


                else if (strcmp(name, "K_mNa") == 0) {
                    return 11;
                }


                else if (strcmp(name, "K_mk") == 0) {
                    return 12;
                }


                else if (strcmp(name, "K_o") == 0) {
                    return 13;
                }


                else if (strcmp(name, "K_pCa") == 0) {
                    return 14;
                }


                else if (strcmp(name, "K_sat") == 0) {
                    return 15;
                }


                else if (strcmp(name, "K_up") == 0) {
                    return 16;
                }


                else if (strcmp(name, "Km_Ca") == 0) {
                    return 17;
                }


                else if (strcmp(name, "Km_Nai") == 0) {
                    return 18;
                }


                else if (strcmp(name, "Na_o") == 0) {
                    return 19;
                }


                else if (strcmp(name, "P_NaK") == 0) {
                    return 20;
                }


                else if (strcmp(name, "P_kna") == 0) {
                    return 21;
                }


                else if (strcmp(name, "R") == 0) {
                    return 22;
                }


                else if (strcmp(name, "T") == 0) {
                    return 23;
                }


                else if (strcmp(name, "V_c") == 0) {
                    return 24;
                }


                else if (strcmp(name, "V_leak") == 0) {
                    return 25;
                }


                else if (strcmp(name, "V_rel") == 0) {
                    return 26;
                }


                else if (strcmp(name, "V_sr") == 0) {
                    return 27;
                }


                else if (strcmp(name, "V_ss") == 0) {
                    return 28;
                }


                else if (strcmp(name, "V_xfer") == 0) {
                    return 29;
                }


                else if (strcmp(name, "Vmax_up") == 0) {
                    return 30;
                }


                else if (strcmp(name, "alpha") == 0) {
                    return 31;
                }


                else if (strcmp(name, "g_CaL") == 0) {
                    return 32;
                }


                else if (strcmp(name, "g_K1") == 0) {
                    return 33;
                }


                else if (strcmp(name, "g_Kr") == 0) {
                    return 34;
                }


                else if (strcmp(name, "g_Ks") == 0) {
                    return 35;
                }


                else if (strcmp(name, "g_Na") == 0) {
                    return 36;
                }


                else if (strcmp(name, "g_bca") == 0) {
                    return 37;
                }


                else if (strcmp(name, "g_bna") == 0) {
                    return 38;
                }


                else if (strcmp(name, "g_pCa") == 0) {
                    return 39;
                }


                else if (strcmp(name, "g_pK") == 0) {
                    return 40;
                }


                else if (strcmp(name, "g_to") == 0) {
                    return 41;
                }


                else if (strcmp(name, "gamma_") == 0) {
                    return 42;
                }


                else if (strcmp(name, "k1_prime") == 0) {
                    return 43;
                }


                else if (strcmp(name, "k2_prime") == 0) {
                    return 44;
                }


                else if (strcmp(name, "k3") == 0) {
                    return 45;
                }


                else if (strcmp(name, "k4") == 0) {
                    return 46;
                }


                else if (strcmp(name, "max_sr") == 0) {
                    return 47;
                }


                else if (strcmp(name, "min_sr") == 0) {
                    return 48;
                }


                else if (strcmp(name, "stim_amplitude") == 0) {
                    return 49;
                }


                else if (strcmp(name, "stim_duration") == 0) {
                    return 50;
                }


                else if (strcmp(name, "stim_end") == 0) {
                    return 51;
                }


                else if (strcmp(name, "stim_period") == 0) {
                    return 52;
                }


                else if (strcmp(name, "stim_start") == 0) {
                    return 53;
                }

                return -1;
            }

            int state_index(const char name[])
            {

                if (strcmp(name, "fCass") == 0) {
                    return 0;
                }


                else if (strcmp(name, "f") == 0) {
                    return 1;
                }


                else if (strcmp(name, "f2") == 0) {
                    return 2;
                }


                else if (strcmp(name, "r") == 0) {
                    return 3;
                }


                else if (strcmp(name, "s") == 0) {
                    return 4;
                }


                else if (strcmp(name, "Ca_i") == 0) {
                    return 5;
                }


                else if (strcmp(name, "Na_i") == 0) {
                    return 6;
                }


                else if (strcmp(name, "h") == 0) {
                    return 7;
                }


                else if (strcmp(name, "j") == 0) {
                    return 8;
                }


                else if (strcmp(name, "m") == 0) {
                    return 9;
                }


                else if (strcmp(name, "Xr1") == 0) {
                    return 10;
                }


                else if (strcmp(name, "Xr2") == 0) {
                    return 11;
                }


                else if (strcmp(name, "Xs") == 0) {
                    return 12;
                }


                else if (strcmp(name, "d") == 0) {
                    return 13;
                }


                else if (strcmp(name, "R_prime") == 0) {
                    return 14;
                }


                else if (strcmp(name, "K_i") == 0) {
                    return 15;
                }


                else if (strcmp(name, "V") == 0) {
                    return 16;
                }


                else if (strcmp(name, "Ca_SR") == 0) {
                    return 17;
                }


                else if (strcmp(name, "Ca_ss") == 0) {
                    return 18;
                }

                return -1;
            }

            int monitor_index(const char name[])
            {

                if (strcmp(name, "Ca_i_bufc") == 0) {
                    return 0;
                }


                else if (strcmp(name, "Ca_sr_bufsr") == 0) {
                    return 1;
                }


                else if (strcmp(name, "fCass_inf") == 0) {
                    return 2;
                }


                else if (strcmp(name, "tau_fCass") == 0) {
                    return 3;
                }


                else if (strcmp(name, "Ca_ss_bufss") == 0) {
                    return 4;
                }


                else if (strcmp(name, "E_Ca") == 0) {
                    return 5;
                }


                else if (strcmp(name, "E_K") == 0) {
                    return 6;
                }


                else if (strcmp(name, "E_Ks") == 0) {
                    return 7;
                }


                else if (strcmp(name, "E_Na") == 0) {
                    return 8;
                }


                else if (strcmp(name, "alpha_d") == 0) {
                    return 9;
                }


                else if (strcmp(name, "alpha_h") == 0) {
                    return 10;
                }


                else if (strcmp(name, "alpha_j") == 0) {
                    return 11;
                }


                else if (strcmp(name, "alpha_m") == 0) {
                    return 12;
                }


                else if (strcmp(name, "alpha_xr1") == 0) {
                    return 13;
                }


                else if (strcmp(name, "alpha_xr2") == 0) {
                    return 14;
                }


                else if (strcmp(name, "alpha_xs") == 0) {
                    return 15;
                }


                else if (strcmp(name, "beta_d") == 0) {
                    return 16;
                }


                else if (strcmp(name, "beta_h") == 0) {
                    return 17;
                }


                else if (strcmp(name, "beta_j") == 0) {
                    return 18;
                }


                else if (strcmp(name, "beta_m") == 0) {
                    return 19;
                }


                else if (strcmp(name, "beta_xr1") == 0) {
                    return 20;
                }


                else if (strcmp(name, "beta_xr2") == 0) {
                    return 21;
                }


                else if (strcmp(name, "beta_xs") == 0) {
                    return 22;
                }


                else if (strcmp(name, "d_inf") == 0) {
                    return 23;
                }


                else if (strcmp(name, "f2_inf") == 0) {
                    return 24;
                }


                else if (strcmp(name, "f_inf") == 0) {
                    return 25;
                }


                else if (strcmp(name, "gamma_d") == 0) {
                    return 26;
                }


                else if (strcmp(name, "h_inf") == 0) {
                    return 27;
                }


                else if (strcmp(name, "j_inf") == 0) {
                    return 28;
                }


                else if (strcmp(name, "m_inf") == 0) {
                    return 29;
                }


                else if (strcmp(name, "r_inf") == 0) {
                    return 30;
                }


                else if (strcmp(name, "s_inf") == 0) {
                    return 31;
                }


                else if (strcmp(name, "tau_f") == 0) {
                    return 32;
                }


                else if (strcmp(name, "tau_f2") == 0) {
                    return 33;
                }


                else if (strcmp(name, "tau_r") == 0) {
                    return 34;
                }


                else if (strcmp(name, "tau_s") == 0) {
                    return 35;
                }


                else if (strcmp(name, "xr1_inf") == 0) {
                    return 36;
                }


                else if (strcmp(name, "xr2_inf") == 0) {
                    return 37;
                }


                else if (strcmp(name, "xs_inf") == 0) {
                    return 38;
                }


                else if (strcmp(name, "i_CaL") == 0) {
                    return 39;
                }


                else if (strcmp(name, "i_NaCa") == 0) {
                    return 40;
                }


                else if (strcmp(name, "i_NaK") == 0) {
                    return 41;
                }


                else if (strcmp(name, "i_Stim") == 0) {
                    return 42;
                }


                else if (strcmp(name, "i_leak") == 0) {
                    return 43;
                }


                else if (strcmp(name, "i_p_Ca") == 0) {
                    return 44;
                }


                else if (strcmp(name, "i_up") == 0) {
                    return 45;
                }


                else if (strcmp(name, "i_xfer") == 0) {
                    return 46;
                }


                else if (strcmp(name, "kcasr") == 0) {
                    return 47;
                }


                else if (strcmp(name, "dfCass_dt") == 0) {
                    return 48;
                }


                else if (strcmp(name, "i_b_Ca") == 0) {
                    return 49;
                }


                else if (strcmp(name, "alpha_K1") == 0) {
                    return 50;
                }


                else if (strcmp(name, "beta_K1") == 0) {
                    return 51;
                }


                else if (strcmp(name, "i_Kr") == 0) {
                    return 52;
                }


                else if (strcmp(name, "i_p_K") == 0) {
                    return 53;
                }


                else if (strcmp(name, "i_to") == 0) {
                    return 54;
                }


                else if (strcmp(name, "i_Ks") == 0) {
                    return 55;
                }


                else if (strcmp(name, "i_Na") == 0) {
                    return 56;
                }


                else if (strcmp(name, "i_b_Na") == 0) {
                    return 57;
                }


                else if (strcmp(name, "tau_h") == 0) {
                    return 58;
                }


                else if (strcmp(name, "tau_j") == 0) {
                    return 59;
                }


                else if (strcmp(name, "tau_m") == 0) {
                    return 60;
                }


                else if (strcmp(name, "tau_xr1") == 0) {
                    return 61;
                }


                else if (strcmp(name, "tau_xr2") == 0) {
                    return 62;
                }


                else if (strcmp(name, "tau_xs") == 0) {
                    return 63;
                }


                else if (strcmp(name, "tau_d") == 0) {
                    return 64;
                }


                else if (strcmp(name, "df_dt") == 0) {
                    return 65;
                }


                else if (strcmp(name, "df2_dt") == 0) {
                    return 66;
                }


                else if (strcmp(name, "dr_dt") == 0) {
                    return 67;
                }


                else if (strcmp(name, "ds_dt") == 0) {
                    return 68;
                }


                else if (strcmp(name, "k1") == 0) {
                    return 69;
                }


                else if (strcmp(name, "k2") == 0) {
                    return 70;
                }


                else if (strcmp(name, "dCa_i_dt") == 0) {
                    return 71;
                }


                else if (strcmp(name, "xK1_inf") == 0) {
                    return 72;
                }


                else if (strcmp(name, "dNa_i_dt") == 0) {
                    return 73;
                }


                else if (strcmp(name, "dh_dt") == 0) {
                    return 74;
                }


                else if (strcmp(name, "dj_dt") == 0) {
                    return 75;
                }


                else if (strcmp(name, "dm_dt") == 0) {
                    return 76;
                }


                else if (strcmp(name, "dXr1_dt") == 0) {
                    return 77;
                }


                else if (strcmp(name, "dXr2_dt") == 0) {
                    return 78;
                }


                else if (strcmp(name, "dXs_dt") == 0) {
                    return 79;
                }


                else if (strcmp(name, "dd_dt") == 0) {
                    return 80;
                }


                else if (strcmp(name, "O_") == 0) {
                    return 81;
                }


                else if (strcmp(name, "dR_prime_dt") == 0) {
                    return 82;
                }


                else if (strcmp(name, "i_K1") == 0) {
                    return 83;
                }


                else if (strcmp(name, "i_rel") == 0) {
                    return 84;
                }


                else if (strcmp(name, "dK_i_dt") == 0) {
                    return 85;
                }


                else if (strcmp(name, "dV_dt") == 0) {
                    return 86;
                }


                else if (strcmp(name, "dCa_SR_dt") == 0) {
                    return 87;
                }


                else if (strcmp(name, "dCa_ss_dt") == 0) {
                    return 88;
                }

                return -1;
            }

            void init_parameter_values(double* parameters){
                /*
                Buf_c=0.2, Buf_sr=10.0, Buf_ss=0.4, Ca_o=2.0, Cm=0.185, EC=1.5, F=96485.3415, K_NaCa=1000.0, K_buf_c=0.001, K_buf_sr=0.3, K_buf_ss=0.00025, K_mNa=40.0, K_mk=1.0, K_o=5.4, K_pCa=0.0005, K_sat=0.1, K_up=0.00025, Km_Ca=1.38, Km_Nai=87.5, Na_o=140.0, P_NaK=2.724, P_kna=0.03, R=8314.472, T=310.0, V_c=0.016404, V_leak=0.00036, V_rel=0.102, V_sr=0.001094, V_ss=5.468e-05, V_xfer=0.0038, Vmax_up=0.006375, alpha=2.5, g_CaL=3.98e-05, g_K1=5.405, g_Kr=0.153, g_Ks=0.392, g_Na=14.838, g_bca=0.000592, g_bna=0.00029, g_pCa=0.1238, g_pK=0.0146, g_to=0.073, gamma_=0.35, k1_prime=0.15, k2_prime=0.045, k3=0.06, k4=0.005, max_sr=2.5, min_sr=1.0, stim_amplitude=52.0, stim_duration=1.0, stim_end=1000000.0, stim_period=1000.0, stim_start=10.0
                */
                parameters[0] = 0.2;
                parameters[1] = 10.0;
                parameters[2] = 0.4;
                parameters[3] = 2.0;
                parameters[4] = 0.185;
                parameters[5] = 1.5;
                parameters[6] = 96485.3415;
                parameters[7] = 1000.0;
                parameters[8] = 0.001;
                parameters[9] = 0.3;
                parameters[10] = 0.00025;
                parameters[11] = 40.0;
                parameters[12] = 1.0;
                parameters[13] = 5.4;
                parameters[14] = 0.0005;
                parameters[15] = 0.1;
                parameters[16] = 0.00025;
                parameters[17] = 1.38;
                parameters[18] = 87.5;
                parameters[19] = 140.0;
                parameters[20] = 2.724;
                parameters[21] = 0.03;
                parameters[22] = 8314.472;
                parameters[23] = 310.0;
                parameters[24] = 0.016404;
                parameters[25] = 0.00036;
                parameters[26] = 0.102;
                parameters[27] = 0.001094;
                parameters[28] = 5.468e-05;
                parameters[29] = 0.0038;
                parameters[30] = 0.006375;
                parameters[31] = 2.5;
                parameters[32] = 3.98e-05;
                parameters[33] = 5.405;
                parameters[34] = 0.153;
                parameters[35] = 0.392;
                parameters[36] = 14.838;
                parameters[37] = 0.000592;
                parameters[38] = 0.00029;
                parameters[39] = 0.1238;
                parameters[40] = 0.0146;
                parameters[41] = 0.073;
                parameters[42] = 0.35;
                parameters[43] = 0.15;
                parameters[44] = 0.045;
                parameters[45] = 0.06;
                parameters[46] = 0.005;
                parameters[47] = 2.5;
                parameters[48] = 1.0;
                parameters[49] = 52.0;
                parameters[50] = 1.0;
                parameters[51] = 1000000.0;
                parameters[52] = 1000.0;
                parameters[53] = 10.0;
            }

            void init_state_values(double* states){
                /*
                fCass=0.9953, f=0.8009, f2=0.9778, r=2.235e-08, s=0.3212, Ca_i=0.00013, Na_i=10.355, h=0.7573, j=0.7225, m=0.00155, Xr1=0.00448, Xr2=0.476, Xs=0.0087, d=3.164e-05, R_prime=0.9068, K_i=138.4, V=-86.709, Ca_SR=3.715, Ca_ss=0.00036
                */
                states[0] = 0.9953;
                states[1] = 0.8009;
                states[2] = 0.9778;
                states[3] = 2.235e-08;
                states[4] = 0.3212;
                states[5] = 0.00013;
                states[6] = 10.355;
                states[7] = 0.7573;
                states[8] = 0.7225;
                states[9] = 0.00155;
                states[10] = 0.00448;
                states[11] = 0.476;
                states[12] = 0.0087;
                states[13] = 3.164e-05;
                states[14] = 0.9068;
                states[15] = 138.4;
                states[16] = -86.709;
                states[17] = 3.715;
                states[18] = 0.00036;
            }

            /**
             * @brief Device-callable kernel for this model.
             *
             * Holds the model math as MFEM_HOST_DEVICE *static* functions and the
             * model metadata as compile-time constants, so that the reaction
             * kernels can instantiate a templated mfem::forall over it with no
             * virtual dispatch and full inlining.
             */
            struct Kernel : IonicKernelDefaults
            {
                static constexpr int nstates = 19;
                static constexpr int nparams = 54;
                static constexpr int nmonitored = 89;

                static constexpr int potential_idx = 16;   // "V"
                static constexpr int calcium_idx = 5;   // "Ca_i"
                static constexpr int stim_ampl_idx = 49;   // "stim_amplitude"
                static constexpr int stim_duration_idx = 50;   // "stim_duration"
                static constexpr int stim_start_idx = 53;   // "stim_start"
                static constexpr int stim_end_idx = 51;   // "stim_end"
                static constexpr int stim_period_idx = 52;   // "stim_period"

                static constexpr bool dimensionless = false;
                static constexpr real_t stim_sign = 1.0;

                MFEM_HOST_DEVICE static void rhs(const double t, const double *__restrict states, const double *__restrict parameters, double* values){

                    // Assign states
                    const double fCass = states[0];
                    const double f = states[1];
                    const double f2 = states[2];
                    const double r = states[3];
                    const double s = states[4];
                    const double Ca_i = states[5];
                    const double Na_i = states[6];
                    const double h = states[7];
                    const double j = states[8];
                    const double m = states[9];
                    const double Xr1 = states[10];
                    const double Xr2 = states[11];
                    const double Xs = states[12];
                    const double d = states[13];
                    const double R_prime = states[14];
                    const double K_i = states[15];
                    const double V = states[16];
                    const double Ca_SR = states[17];
                    const double Ca_ss = states[18];

                    // Assign parameters
                    const double Buf_c = parameters[0];
                    const double Buf_sr = parameters[1];
                    const double Buf_ss = parameters[2];
                    const double Ca_o = parameters[3];
                    const double Cm = parameters[4];
                    const double EC = parameters[5];
                    const double F = parameters[6];
                    const double K_NaCa = parameters[7];
                    const double K_buf_c = parameters[8];
                    const double K_buf_sr = parameters[9];
                    const double K_buf_ss = parameters[10];
                    const double K_mNa = parameters[11];
                    const double K_mk = parameters[12];
                    const double K_o = parameters[13];
                    const double K_pCa = parameters[14];
                    const double K_sat = parameters[15];
                    const double K_up = parameters[16];
                    const double Km_Ca = parameters[17];
                    const double Km_Nai = parameters[18];
                    const double Na_o = parameters[19];
                    const double P_NaK = parameters[20];
                    const double P_kna = parameters[21];
                    const double R = parameters[22];
                    const double T = parameters[23];
                    const double V_c = parameters[24];
                    const double V_leak = parameters[25];
                    const double V_rel = parameters[26];
                    const double V_sr = parameters[27];
                    const double V_ss = parameters[28];
                    const double V_xfer = parameters[29];
                    const double Vmax_up = parameters[30];
                    const double alpha = parameters[31];
                    const double g_CaL = parameters[32];
                    const double g_K1 = parameters[33];
                    const double g_Kr = parameters[34];
                    const double g_Ks = parameters[35];
                    const double g_Na = parameters[36];
                    const double g_bca = parameters[37];
                    const double g_bna = parameters[38];
                    const double g_pCa = parameters[39];
                    const double g_pK = parameters[40];
                    const double g_to = parameters[41];
                    const double gamma_ = parameters[42];
                    const double k1_prime = parameters[43];
                    const double k2_prime = parameters[44];
                    const double k3 = parameters[45];
                    const double k4 = parameters[46];
                    const double max_sr = parameters[47];
                    const double min_sr = parameters[48];
                    const double stim_amplitude = parameters[49];
                    const double stim_duration = parameters[50];
                    const double stim_end = parameters[51];
                    const double stim_period = parameters[52];
                    const double stim_start = parameters[53];

                    // Assign expressions
                    const double Ca_i_bufc = 1.0/((Buf_c*K_buf_c)/pow(Ca_i + K_buf_c, 2.0) + 1.0);
                    const double Ca_sr_bufsr = 1.0/((Buf_sr*K_buf_sr)/pow(Ca_SR + K_buf_sr, 2.0) + 1.0);
                    const double fCass_inf = 0.4 + 0.6/(pow(Ca_ss/0.05, 2.0) + 1.0);
                    const double tau_fCass = 2.0 + 80.0/(pow(Ca_ss/0.05, 2.0) + 1.0);
                    const double Ca_ss_bufss = 1.0/((Buf_ss*K_buf_ss)/pow(Ca_ss + K_buf_ss, 2.0) + 1.0);
                    const double E_Ca = (((0.5*R)*T)/F)*log(Ca_o/Ca_i);
                    const double E_K = ((R*T)/F)*log(K_o/K_i);
                    const double E_Ks = ((R*T)/F)*log((K_o + Na_o*P_kna)/(K_i + Na_i*P_kna));
                    const double E_Na = ((R*T)/F)*log(Na_o/Na_i);
                    const double alpha_d = 0.25 + 1.4/(exp((-V - 1*35.0)/13.0) + 1.0);
                    const double alpha_h = (V < 40.0*(-1)) ? 0.057*exp((-(V + 80.0))/6.8) : 0.0;
                    const double alpha_j = (V < 40.0*(-1)) ? (((V + 37.78)*((25428.0*(-1))*exp(0.2444*V) - 6.948e-06*exp(V*(0.04391*(-1)))))/1.0)/(exp(0.311*(V + 79.23)) + 1.0) : 0.0;
                    const double alpha_m = 1.0/(exp((-V - 1*60.0)/5.0) + 1.0);
                    const double alpha_xr1 = 450.0/(exp((-V - 1*45.0)/10.0) + 1.0);
                    const double alpha_xr2 = 3.0/(exp((-V - 1*60.0)/20.0) + 1.0);
                    const double alpha_xs = 1400.0/sqrt(exp((5.0 - V)/6.0) + 1.0);
                    const double beta_d = 1.4/(exp((V + 5.0)/5.0) + 1.0);
                    const double beta_h = (V < 40.0*(-1)) ? 2.7*exp(0.079*V) + 310000.0*exp(0.3485*V) : 0.77/((0.13*(exp((V + 10.66)/((11.1*(-1)))) + 1.0)));
                    const double beta_j = (V < 40.0*(-1)) ? (0.02424*exp(V*(0.01052*(-1))))/(exp((0.1378*(-1))*(V + 40.14)) + 1.0) : (0.6*exp(0.057*V))/(exp((0.1*(-1))*(V + 32.0)) + 1.0);
                    const double beta_m = 0.1/(exp((V + 35.0)/5.0) + 1.0) + 0.1/(exp((V - 1*50.0)/200.0) + 1.0);
                    const double beta_xr1 = 6.0/(exp((V + 30.0)/11.5) + 1.0);
                    const double beta_xr2 = 1.12/(exp((V - 1*60.0)/20.0) + 1.0);
                    const double beta_xs = 1.0/(exp((V - 1*35.0)/15.0) + 1.0);
                    const double d_inf = 1.0/(exp((-V - 1*8.0)/7.5) + 1.0);
                    const double f2_inf = 0.33 + 0.67/(exp((V + 35.0)/7.0) + 1.0);
                    const double f_inf = 1.0/(exp((V + 20.0)/7.0) + 1.0);
                    const double gamma_d = 1.0/(exp((50.0 - V)/20.0) + 1.0);
                    const double h_inf = 1.0/pow(exp((V + 71.55)/7.43) + 1.0, 2.0);
                    const double j_inf = 1.0/pow(exp((V + 71.55)/7.43) + 1.0, 2.0);
                    const double m_inf = 1.0/pow(exp((-V - 1*56.86)/9.03) + 1.0, 2.0);
                    const double r_inf = 1.0/(exp((20.0 - V)/6.0) + 1.0);
                    const double s_inf = 1.0/(exp((V + 28.0)/5.0) + 1.0);
                    const double tau_f = ((1102.5*exp((-pow(V + 27.0, 2.0))/225.0) + 200.0/(exp((13.0 - V)/10.0) + 1.0)) + 180.0/(exp((V + 30.0)/10.0) + 1.0)) + 20.0;
                    const double tau_f2 = (562.0*exp((-pow(V + 27.0, 2.0))/240.0) + 31.0/(exp((25.0 - V)/10.0) + 1.0)) + 80.0/(exp((V + 30.0)/10.0) + 1.0);
                    const double tau_r = 9.5*exp((-pow(V + 40.0, 2.0))/1800.0) + 0.8;
                    const double tau_s = 1000.0*exp((-pow(V + 67.0, 2.0))/1000.0) + 8.0;
                    const double xr1_inf = 1.0/(exp((-V - 1*26.0)/7.0) + 1.0);
                    const double xr2_inf = 1.0/(exp((V + 88.0)/24.0) + 1.0);
                    const double xs_inf = 1.0/(exp((-V - 1*5.0)/14.0) + 1.0);
                    const double i_CaL = (((pow(F, 2.0)*((4.0*(fCass*(f2*(f*(d*g_CaL)))))*(V - 1*15.0)))/((R*T)))*(-Ca_o + (0.25*Ca_ss)*exp((F*(2.0*(V - 1*15.0)))/((R*T)))))/(exp((F*(2.0*(V - 1*15.0)))/((R*T))) - 1*1.0);
                    const double i_NaCa = (K_NaCa*(Ca_o*(pow(Na_i, 3.0)*exp((F*(V*gamma_))/((R*T)))) - Ca_i*alpha*pow(Na_o, 3.0)*exp((F*(V*(gamma_ - 1*1.0)))/((R*T)))))/((((Ca_o + Km_Ca)*(pow(Km_Nai, 3.0) + pow(Na_o, 3.0)))*(K_sat*exp((F*(V*(gamma_ - 1*1.0)))/((R*T))) + 1.0)));
                    const double i_NaK = ((Na_i*((K_o*P_NaK)/(K_mk + K_o)))/(K_mNa + Na_i))/((0.1245*exp((F*(V*(0.1*(-1))))/((R*T))) + 1.0) + 0.0353*exp((F*(-V))/((R*T))));
                    const double i_Stim = (stim_end >= t && stim_duration + stim_start >= (-stim_period)*floor(t/stim_period) + t && stim_start <= (-stim_period)*floor(t/stim_period) + t) ? -stim_amplitude : 0.0;
                    const double i_leak = V_leak*(Ca_SR - Ca_i);
                    const double i_p_Ca = (Ca_i*g_pCa)/(Ca_i + K_pCa);
                    const double i_up = Vmax_up/(1.0 + pow(K_up, 2.0)/pow(Ca_i, 2.0));
                    const double i_xfer = V_xfer*(-Ca_i + Ca_ss);
                    const double kcasr = max_sr - (max_sr - min_sr)/(pow(EC/Ca_SR, 2.0) + 1.0);
                    const double dfCass_dt = (-fCass + fCass_inf)/tau_fCass;
                    values[0] = dfCass_dt;
                    const double i_b_Ca = g_bca*(-E_Ca + V);
                    const double alpha_K1 = 0.1/(exp(0.06*((-E_K + V) - 1*200.0)) + 1.0);
                    const double beta_K1 = (3.0*exp(0.0002*((-E_K + V) + 100.0)) + exp(0.1*((-E_K + V) - 1*10.0)))/(exp((0.5*(-1))*(-E_K + V)) + 1.0);
                    const double i_Kr = (Xr2*(Xr1*((0.4303314829119352*sqrt(K_o))*g_Kr)))*(-E_K + V);
                    const double i_p_K = (g_pK*(-E_K + V))/(exp((25.0 - V)/5.98) + 1.0);
                    const double i_to = (s*(g_to*r))*(-E_K + V);
                    const double i_Ks = (pow(Xs, 2.0)*g_Ks)*(-E_Ks + V);
                    const double i_Na = (j*(h*(g_Na*pow(m, 3.0))))*(-E_Na + V);
                    const double i_b_Na = g_bna*(-E_Na + V);
                    const double tau_h = 1.0/(alpha_h + beta_h);
                    const double tau_j = 1.0/(alpha_j + beta_j);
                    const double tau_m = (1.0*alpha_m)*beta_m;
                    const double tau_xr1 = (1.0*alpha_xr1)*beta_xr1;
                    const double tau_xr2 = (1.0*alpha_xr2)*beta_xr2;
                    const double tau_xs = (1.0*alpha_xs)*beta_xs + 80.0;
                    const double tau_d = (1.0*alpha_d)*beta_d + gamma_d;
                    const double df_dt = (-f + f_inf)/tau_f;
                    values[1] = df_dt;
                    const double df2_dt = (-f2 + f2_inf)/tau_f2;
                    values[2] = df2_dt;
                    const double dr_dt = (-r + r_inf)/tau_r;
                    values[3] = dr_dt;
                    const double ds_dt = (-s + s_inf)/tau_s;
                    values[4] = ds_dt;
                    const double k1 = k1_prime/kcasr;
                    const double k2 = k2_prime*kcasr;
                    const double dCa_i_dt = Ca_i_bufc*(((-Cm)*(1.0*((-2.0)*i_NaCa + (i_b_Ca + i_p_Ca))))/((F*((1.0*2.0)*V_c))) + (i_xfer + (V_sr*(i_leak - i_up))/V_c));
                    values[5] = dCa_i_dt;
                    const double xK1_inf = alpha_K1/(alpha_K1 + beta_K1);
                    const double dNa_i_dt = Cm*(((1.0*(-1))*(3.0*i_NaCa + (3.0*i_NaK + (i_Na + i_b_Na))))/((F*(1.0*V_c))));
                    values[6] = dNa_i_dt;
                    const double dh_dt = (-h + h_inf)/tau_h;
                    values[7] = dh_dt;
                    const double dj_dt = (-j + j_inf)/tau_j;
                    values[8] = dj_dt;
                    const double dm_dt = (-m + m_inf)/tau_m;
                    values[9] = dm_dt;
                    const double dXr1_dt = (-Xr1 + xr1_inf)/tau_xr1;
                    values[10] = dXr1_dt;
                    const double dXr2_dt = (-Xr2 + xr2_inf)/tau_xr2;
                    values[11] = dXr2_dt;
                    const double dXs_dt = (-Xs + xs_inf)/tau_xs;
                    values[12] = dXs_dt;
                    const double dd_dt = (-d + d_inf)/tau_d;
                    values[13] = dd_dt;
                    const double O_ = (R_prime*(pow(Ca_ss, 2.0)*k1))/(pow(Ca_ss, 2.0)*k1 + k3);
                    const double dR_prime_dt = R_prime*(Ca_ss*(-k2)) + k4*(1.0 - R_prime);
                    values[14] = dR_prime_dt;
                    const double i_K1 = ((0.4303314829119352*sqrt(K_o))*(g_K1*xK1_inf))*(-E_K + V);
                    const double i_rel = (O_*V_rel)*(Ca_SR - Ca_ss);
                    const double dK_i_dt = Cm*(((1.0*(-1))*((-2.0)*i_NaK + (i_Stim + (i_p_K + (i_Ks + (i_Kr + (i_K1 + i_to)))))))/((F*(1.0*V_c))));
                    values[15] = dK_i_dt;
                    const double dV_dt = ((1.0*(-1))/1.0)*(i_Stim + (i_p_Ca + (i_p_K + (i_b_Ca + (i_NaCa + (i_b_Na + (i_Na + (i_NaK + (i_CaL + (i_Ks + (i_Kr + (i_K1 + i_to))))))))))));
                    values[16] = dV_dt;
                    const double dCa_SR_dt = Ca_sr_bufsr*(i_up - (i_leak + i_rel));
                    values[17] = dCa_SR_dt;
                    const double dCa_ss_dt = Ca_ss_bufss*(((Cm*(i_CaL*(1.0*(-1))))/((F*((1.0*2.0)*V_ss))) + (V_sr*i_rel)/V_ss) - (V_c*i_xfer)/V_ss);
                    values[18] = dCa_ss_dt;
                }

                MFEM_HOST_DEVICE static void monitor_values(const double t, const double *__restrict states, const double *__restrict parameters, double* values){

                    // Assign states
                    const double fCass = states[0];
                    const double f = states[1];
                    const double f2 = states[2];
                    const double r = states[3];
                    const double s = states[4];
                    const double Ca_i = states[5];
                    const double Na_i = states[6];
                    const double h = states[7];
                    const double j = states[8];
                    const double m = states[9];
                    const double Xr1 = states[10];
                    const double Xr2 = states[11];
                    const double Xs = states[12];
                    const double d = states[13];
                    const double R_prime = states[14];
                    const double K_i = states[15];
                    const double V = states[16];
                    const double Ca_SR = states[17];
                    const double Ca_ss = states[18];

                    // Assign parameters
                    const double Buf_c = parameters[0];
                    const double Buf_sr = parameters[1];
                    const double Buf_ss = parameters[2];
                    const double Ca_o = parameters[3];
                    const double Cm = parameters[4];
                    const double EC = parameters[5];
                    const double F = parameters[6];
                    const double K_NaCa = parameters[7];
                    const double K_buf_c = parameters[8];
                    const double K_buf_sr = parameters[9];
                    const double K_buf_ss = parameters[10];
                    const double K_mNa = parameters[11];
                    const double K_mk = parameters[12];
                    const double K_o = parameters[13];
                    const double K_pCa = parameters[14];
                    const double K_sat = parameters[15];
                    const double K_up = parameters[16];
                    const double Km_Ca = parameters[17];
                    const double Km_Nai = parameters[18];
                    const double Na_o = parameters[19];
                    const double P_NaK = parameters[20];
                    const double P_kna = parameters[21];
                    const double R = parameters[22];
                    const double T = parameters[23];
                    const double V_c = parameters[24];
                    const double V_leak = parameters[25];
                    const double V_rel = parameters[26];
                    const double V_sr = parameters[27];
                    const double V_ss = parameters[28];
                    const double V_xfer = parameters[29];
                    const double Vmax_up = parameters[30];
                    const double alpha = parameters[31];
                    const double g_CaL = parameters[32];
                    const double g_K1 = parameters[33];
                    const double g_Kr = parameters[34];
                    const double g_Ks = parameters[35];
                    const double g_Na = parameters[36];
                    const double g_bca = parameters[37];
                    const double g_bna = parameters[38];
                    const double g_pCa = parameters[39];
                    const double g_pK = parameters[40];
                    const double g_to = parameters[41];
                    const double gamma_ = parameters[42];
                    const double k1_prime = parameters[43];
                    const double k2_prime = parameters[44];
                    const double k3 = parameters[45];
                    const double k4 = parameters[46];
                    const double max_sr = parameters[47];
                    const double min_sr = parameters[48];
                    const double stim_amplitude = parameters[49];
                    const double stim_duration = parameters[50];
                    const double stim_end = parameters[51];
                    const double stim_period = parameters[52];
                    const double stim_start = parameters[53];

                    // Assign expressions
                    const double Ca_i_bufc = 1.0/((Buf_c*K_buf_c)/pow(Ca_i + K_buf_c, 2.0) + 1.0);
                    values[0] = Ca_i_bufc;
                    const double Ca_sr_bufsr = 1.0/((Buf_sr*K_buf_sr)/pow(Ca_SR + K_buf_sr, 2.0) + 1.0);
                    values[1] = Ca_sr_bufsr;
                    const double fCass_inf = 0.4 + 0.6/(pow(Ca_ss/0.05, 2.0) + 1.0);
                    values[2] = fCass_inf;
                    const double tau_fCass = 2.0 + 80.0/(pow(Ca_ss/0.05, 2.0) + 1.0);
                    values[3] = tau_fCass;
                    const double Ca_ss_bufss = 1.0/((Buf_ss*K_buf_ss)/pow(Ca_ss + K_buf_ss, 2.0) + 1.0);
                    values[4] = Ca_ss_bufss;
                    const double E_Ca = (((0.5*R)*T)/F)*log(Ca_o/Ca_i);
                    values[5] = E_Ca;
                    const double E_K = ((R*T)/F)*log(K_o/K_i);
                    values[6] = E_K;
                    const double E_Ks = ((R*T)/F)*log((K_o + Na_o*P_kna)/(K_i + Na_i*P_kna));
                    values[7] = E_Ks;
                    const double E_Na = ((R*T)/F)*log(Na_o/Na_i);
                    values[8] = E_Na;
                    const double alpha_d = 0.25 + 1.4/(exp((-V - 1*35.0)/13.0) + 1.0);
                    values[9] = alpha_d;
                    const double alpha_h = (V < 40.0*(-1)) ? 0.057*exp((-(V + 80.0))/6.8) : 0.0;
                    values[10] = alpha_h;
                    const double alpha_j = (V < 40.0*(-1)) ? (((V + 37.78)*((25428.0*(-1))*exp(0.2444*V) - 6.948e-06*exp(V*(0.04391*(-1)))))/1.0)/(exp(0.311*(V + 79.23)) + 1.0) : 0.0;
                    values[11] = alpha_j;
                    const double alpha_m = 1.0/(exp((-V - 1*60.0)/5.0) + 1.0);
                    values[12] = alpha_m;
                    const double alpha_xr1 = 450.0/(exp((-V - 1*45.0)/10.0) + 1.0);
                    values[13] = alpha_xr1;
                    const double alpha_xr2 = 3.0/(exp((-V - 1*60.0)/20.0) + 1.0);
                    values[14] = alpha_xr2;
                    const double alpha_xs = 1400.0/sqrt(exp((5.0 - V)/6.0) + 1.0);
                    values[15] = alpha_xs;
                    const double beta_d = 1.4/(exp((V + 5.0)/5.0) + 1.0);
                    values[16] = beta_d;
                    const double beta_h = (V < 40.0*(-1)) ? 2.7*exp(0.079*V) + 310000.0*exp(0.3485*V) : 0.77/((0.13*(exp((V + 10.66)/((11.1*(-1)))) + 1.0)));
                    values[17] = beta_h;
                    const double beta_j = (V < 40.0*(-1)) ? (0.02424*exp(V*(0.01052*(-1))))/(exp((0.1378*(-1))*(V + 40.14)) + 1.0) : (0.6*exp(0.057*V))/(exp((0.1*(-1))*(V + 32.0)) + 1.0);
                    values[18] = beta_j;
                    const double beta_m = 0.1/(exp((V + 35.0)/5.0) + 1.0) + 0.1/(exp((V - 1*50.0)/200.0) + 1.0);
                    values[19] = beta_m;
                    const double beta_xr1 = 6.0/(exp((V + 30.0)/11.5) + 1.0);
                    values[20] = beta_xr1;
                    const double beta_xr2 = 1.12/(exp((V - 1*60.0)/20.0) + 1.0);
                    values[21] = beta_xr2;
                    const double beta_xs = 1.0/(exp((V - 1*35.0)/15.0) + 1.0);
                    values[22] = beta_xs;
                    const double d_inf = 1.0/(exp((-V - 1*8.0)/7.5) + 1.0);
                    values[23] = d_inf;
                    const double f2_inf = 0.33 + 0.67/(exp((V + 35.0)/7.0) + 1.0);
                    values[24] = f2_inf;
                    const double f_inf = 1.0/(exp((V + 20.0)/7.0) + 1.0);
                    values[25] = f_inf;
                    const double gamma_d = 1.0/(exp((50.0 - V)/20.0) + 1.0);
                    values[26] = gamma_d;
                    const double h_inf = 1.0/pow(exp((V + 71.55)/7.43) + 1.0, 2.0);
                    values[27] = h_inf;
                    const double j_inf = 1.0/pow(exp((V + 71.55)/7.43) + 1.0, 2.0);
                    values[28] = j_inf;
                    const double m_inf = 1.0/pow(exp((-V - 1*56.86)/9.03) + 1.0, 2.0);
                    values[29] = m_inf;
                    const double r_inf = 1.0/(exp((20.0 - V)/6.0) + 1.0);
                    values[30] = r_inf;
                    const double s_inf = 1.0/(exp((V + 28.0)/5.0) + 1.0);
                    values[31] = s_inf;
                    const double tau_f = ((1102.5*exp((-pow(V + 27.0, 2.0))/225.0) + 200.0/(exp((13.0 - V)/10.0) + 1.0)) + 180.0/(exp((V + 30.0)/10.0) + 1.0)) + 20.0;
                    values[32] = tau_f;
                    const double tau_f2 = (562.0*exp((-pow(V + 27.0, 2.0))/240.0) + 31.0/(exp((25.0 - V)/10.0) + 1.0)) + 80.0/(exp((V + 30.0)/10.0) + 1.0);
                    values[33] = tau_f2;
                    const double tau_r = 9.5*exp((-pow(V + 40.0, 2.0))/1800.0) + 0.8;
                    values[34] = tau_r;
                    const double tau_s = 1000.0*exp((-pow(V + 67.0, 2.0))/1000.0) + 8.0;
                    values[35] = tau_s;
                    const double xr1_inf = 1.0/(exp((-V - 1*26.0)/7.0) + 1.0);
                    values[36] = xr1_inf;
                    const double xr2_inf = 1.0/(exp((V + 88.0)/24.0) + 1.0);
                    values[37] = xr2_inf;
                    const double xs_inf = 1.0/(exp((-V - 1*5.0)/14.0) + 1.0);
                    values[38] = xs_inf;
                    const double i_CaL = (((pow(F, 2.0)*((4.0*(fCass*(f2*(f*(d*g_CaL)))))*(V - 1*15.0)))/((R*T)))*(-Ca_o + (0.25*Ca_ss)*exp((F*(2.0*(V - 1*15.0)))/((R*T)))))/(exp((F*(2.0*(V - 1*15.0)))/((R*T))) - 1*1.0);
                    values[39] = i_CaL;
                    const double i_NaCa = (K_NaCa*(Ca_o*(pow(Na_i, 3.0)*exp((F*(V*gamma_))/((R*T)))) - Ca_i*alpha*pow(Na_o, 3.0)*exp((F*(V*(gamma_ - 1*1.0)))/((R*T)))))/((((Ca_o + Km_Ca)*(pow(Km_Nai, 3.0) + pow(Na_o, 3.0)))*(K_sat*exp((F*(V*(gamma_ - 1*1.0)))/((R*T))) + 1.0)));
                    values[40] = i_NaCa;
                    const double i_NaK = ((Na_i*((K_o*P_NaK)/(K_mk + K_o)))/(K_mNa + Na_i))/((0.1245*exp((F*(V*(0.1*(-1))))/((R*T))) + 1.0) + 0.0353*exp((F*(-V))/((R*T))));
                    values[41] = i_NaK;
                    const double i_Stim = (stim_end >= t && stim_duration + stim_start >= (-stim_period)*floor(t/stim_period) + t && stim_start <= (-stim_period)*floor(t/stim_period) + t) ? -stim_amplitude : 0.0;
                    values[42] = i_Stim;
                    const double i_leak = V_leak*(Ca_SR - Ca_i);
                    values[43] = i_leak;
                    const double i_p_Ca = (Ca_i*g_pCa)/(Ca_i + K_pCa);
                    values[44] = i_p_Ca;
                    const double i_up = Vmax_up/(1.0 + pow(K_up, 2.0)/pow(Ca_i, 2.0));
                    values[45] = i_up;
                    const double i_xfer = V_xfer*(-Ca_i + Ca_ss);
                    values[46] = i_xfer;
                    const double kcasr = max_sr - (max_sr - min_sr)/(pow(EC/Ca_SR, 2.0) + 1.0);
                    values[47] = kcasr;
                    const double dfCass_dt = (-fCass + fCass_inf)/tau_fCass;
                    values[48] = dfCass_dt;
                    const double i_b_Ca = g_bca*(-E_Ca + V);
                    values[49] = i_b_Ca;
                    const double alpha_K1 = 0.1/(exp(0.06*((-E_K + V) - 1*200.0)) + 1.0);
                    values[50] = alpha_K1;
                    const double beta_K1 = (3.0*exp(0.0002*((-E_K + V) + 100.0)) + exp(0.1*((-E_K + V) - 1*10.0)))/(exp((0.5*(-1))*(-E_K + V)) + 1.0);
                    values[51] = beta_K1;
                    const double i_Kr = (Xr2*(Xr1*((0.4303314829119352*sqrt(K_o))*g_Kr)))*(-E_K + V);
                    values[52] = i_Kr;
                    const double i_p_K = (g_pK*(-E_K + V))/(exp((25.0 - V)/5.98) + 1.0);
                    values[53] = i_p_K;
                    const double i_to = (s*(g_to*r))*(-E_K + V);
                    values[54] = i_to;
                    const double i_Ks = (pow(Xs, 2.0)*g_Ks)*(-E_Ks + V);
                    values[55] = i_Ks;
                    const double i_Na = (j*(h*(g_Na*pow(m, 3.0))))*(-E_Na + V);
                    values[56] = i_Na;
                    const double i_b_Na = g_bna*(-E_Na + V);
                    values[57] = i_b_Na;
                    const double tau_h = 1.0/(alpha_h + beta_h);
                    values[58] = tau_h;
                    const double tau_j = 1.0/(alpha_j + beta_j);
                    values[59] = tau_j;
                    const double tau_m = (1.0*alpha_m)*beta_m;
                    values[60] = tau_m;
                    const double tau_xr1 = (1.0*alpha_xr1)*beta_xr1;
                    values[61] = tau_xr1;
                    const double tau_xr2 = (1.0*alpha_xr2)*beta_xr2;
                    values[62] = tau_xr2;
                    const double tau_xs = (1.0*alpha_xs)*beta_xs + 80.0;
                    values[63] = tau_xs;
                    const double tau_d = (1.0*alpha_d)*beta_d + gamma_d;
                    values[64] = tau_d;
                    const double df_dt = (-f + f_inf)/tau_f;
                    values[65] = df_dt;
                    const double df2_dt = (-f2 + f2_inf)/tau_f2;
                    values[66] = df2_dt;
                    const double dr_dt = (-r + r_inf)/tau_r;
                    values[67] = dr_dt;
                    const double ds_dt = (-s + s_inf)/tau_s;
                    values[68] = ds_dt;
                    const double k1 = k1_prime/kcasr;
                    values[69] = k1;
                    const double k2 = k2_prime*kcasr;
                    values[70] = k2;
                    const double dCa_i_dt = Ca_i_bufc*(((-Cm)*(1.0*((-2.0)*i_NaCa + (i_b_Ca + i_p_Ca))))/((F*((1.0*2.0)*V_c))) + (i_xfer + (V_sr*(i_leak - i_up))/V_c));
                    values[71] = dCa_i_dt;
                    const double xK1_inf = alpha_K1/(alpha_K1 + beta_K1);
                    values[72] = xK1_inf;
                    const double dNa_i_dt = Cm*(((1.0*(-1))*(3.0*i_NaCa + (3.0*i_NaK + (i_Na + i_b_Na))))/((F*(1.0*V_c))));
                    values[73] = dNa_i_dt;
                    const double dh_dt = (-h + h_inf)/tau_h;
                    values[74] = dh_dt;
                    const double dj_dt = (-j + j_inf)/tau_j;
                    values[75] = dj_dt;
                    const double dm_dt = (-m + m_inf)/tau_m;
                    values[76] = dm_dt;
                    const double dXr1_dt = (-Xr1 + xr1_inf)/tau_xr1;
                    values[77] = dXr1_dt;
                    const double dXr2_dt = (-Xr2 + xr2_inf)/tau_xr2;
                    values[78] = dXr2_dt;
                    const double dXs_dt = (-Xs + xs_inf)/tau_xs;
                    values[79] = dXs_dt;
                    const double dd_dt = (-d + d_inf)/tau_d;
                    values[80] = dd_dt;
                    const double O_ = (R_prime*(pow(Ca_ss, 2.0)*k1))/(pow(Ca_ss, 2.0)*k1 + k3);
                    values[81] = O_;
                    const double dR_prime_dt = R_prime*(Ca_ss*(-k2)) + k4*(1.0 - R_prime);
                    values[82] = dR_prime_dt;
                    const double i_K1 = ((0.4303314829119352*sqrt(K_o))*(g_K1*xK1_inf))*(-E_K + V);
                    values[83] = i_K1;
                    const double i_rel = (O_*V_rel)*(Ca_SR - Ca_ss);
                    values[84] = i_rel;
                    const double dK_i_dt = Cm*(((1.0*(-1))*((-2.0)*i_NaK + (i_Stim + (i_p_K + (i_Ks + (i_Kr + (i_K1 + i_to)))))))/((F*(1.0*V_c))));
                    values[85] = dK_i_dt;
                    const double dV_dt = ((1.0*(-1))/1.0)*(i_Stim + (i_p_Ca + (i_p_K + (i_b_Ca + (i_NaCa + (i_b_Na + (i_Na + (i_NaK + (i_CaL + (i_Ks + (i_Kr + (i_K1 + i_to))))))))))));
                    values[86] = dV_dt;
                    const double dCa_SR_dt = Ca_sr_bufsr*(i_up - (i_leak + i_rel));
                    values[87] = dCa_SR_dt;
                    const double dCa_ss_dt = Ca_ss_bufss*(((Cm*(i_CaL*(1.0*(-1))))/((F*((1.0*2.0)*V_ss))) + (V_sr*i_rel)/V_ss) - (V_c*i_xfer)/V_ss);
                    values[88] = dCa_ss_dt;
                }

                MFEM_HOST_DEVICE static void explicit_euler(const double *__restrict states, const double t, const double dt, const double *__restrict parameters, double* values){

                    // Assign states
                    const double fCass = states[0];
                    const double f = states[1];
                    const double f2 = states[2];
                    const double r = states[3];
                    const double s = states[4];
                    const double Ca_i = states[5];
                    const double Na_i = states[6];
                    const double h = states[7];
                    const double j = states[8];
                    const double m = states[9];
                    const double Xr1 = states[10];
                    const double Xr2 = states[11];
                    const double Xs = states[12];
                    const double d = states[13];
                    const double R_prime = states[14];
                    const double K_i = states[15];
                    const double V = states[16];
                    const double Ca_SR = states[17];
                    const double Ca_ss = states[18];

                    // Assign parameters
                    const double Buf_c = parameters[0];
                    const double Buf_sr = parameters[1];
                    const double Buf_ss = parameters[2];
                    const double Ca_o = parameters[3];
                    const double Cm = parameters[4];
                    const double EC = parameters[5];
                    const double F = parameters[6];
                    const double K_NaCa = parameters[7];
                    const double K_buf_c = parameters[8];
                    const double K_buf_sr = parameters[9];
                    const double K_buf_ss = parameters[10];
                    const double K_mNa = parameters[11];
                    const double K_mk = parameters[12];
                    const double K_o = parameters[13];
                    const double K_pCa = parameters[14];
                    const double K_sat = parameters[15];
                    const double K_up = parameters[16];
                    const double Km_Ca = parameters[17];
                    const double Km_Nai = parameters[18];
                    const double Na_o = parameters[19];
                    const double P_NaK = parameters[20];
                    const double P_kna = parameters[21];
                    const double R = parameters[22];
                    const double T = parameters[23];
                    const double V_c = parameters[24];
                    const double V_leak = parameters[25];
                    const double V_rel = parameters[26];
                    const double V_sr = parameters[27];
                    const double V_ss = parameters[28];
                    const double V_xfer = parameters[29];
                    const double Vmax_up = parameters[30];
                    const double alpha = parameters[31];
                    const double g_CaL = parameters[32];
                    const double g_K1 = parameters[33];
                    const double g_Kr = parameters[34];
                    const double g_Ks = parameters[35];
                    const double g_Na = parameters[36];
                    const double g_bca = parameters[37];
                    const double g_bna = parameters[38];
                    const double g_pCa = parameters[39];
                    const double g_pK = parameters[40];
                    const double g_to = parameters[41];
                    const double gamma_ = parameters[42];
                    const double k1_prime = parameters[43];
                    const double k2_prime = parameters[44];
                    const double k3 = parameters[45];
                    const double k4 = parameters[46];
                    const double max_sr = parameters[47];
                    const double min_sr = parameters[48];
                    const double stim_amplitude = parameters[49];
                    const double stim_duration = parameters[50];
                    const double stim_end = parameters[51];
                    const double stim_period = parameters[52];
                    const double stim_start = parameters[53];

                    // Assign expressions
                    const double Ca_i_bufc = 1.0/((Buf_c*K_buf_c)/pow(Ca_i + K_buf_c, 2.0) + 1.0);
                    const double Ca_sr_bufsr = 1.0/((Buf_sr*K_buf_sr)/pow(Ca_SR + K_buf_sr, 2.0) + 1.0);
                    const double fCass_inf = 0.4 + 0.6/(pow(Ca_ss/0.05, 2.0) + 1.0);
                    const double tau_fCass = 2.0 + 80.0/(pow(Ca_ss/0.05, 2.0) + 1.0);
                    const double Ca_ss_bufss = 1.0/((Buf_ss*K_buf_ss)/pow(Ca_ss + K_buf_ss, 2.0) + 1.0);
                    const double E_Ca = (((0.5*R)*T)/F)*log(Ca_o/Ca_i);
                    const double E_K = ((R*T)/F)*log(K_o/K_i);
                    const double E_Ks = ((R*T)/F)*log((K_o + Na_o*P_kna)/(K_i + Na_i*P_kna));
                    const double E_Na = ((R*T)/F)*log(Na_o/Na_i);
                    const double alpha_d = 0.25 + 1.4/(exp((-V - 1*35.0)/13.0) + 1.0);
                    const double alpha_h = (V < 40.0*(-1)) ? 0.057*exp((-(V + 80.0))/6.8) : 0.0;
                    const double alpha_j = (V < 40.0*(-1)) ? (((V + 37.78)*((25428.0*(-1))*exp(0.2444*V) - 6.948e-06*exp(V*(0.04391*(-1)))))/1.0)/(exp(0.311*(V + 79.23)) + 1.0) : 0.0;
                    const double alpha_m = 1.0/(exp((-V - 1*60.0)/5.0) + 1.0);
                    const double alpha_xr1 = 450.0/(exp((-V - 1*45.0)/10.0) + 1.0);
                    const double alpha_xr2 = 3.0/(exp((-V - 1*60.0)/20.0) + 1.0);
                    const double alpha_xs = 1400.0/sqrt(exp((5.0 - V)/6.0) + 1.0);
                    const double beta_d = 1.4/(exp((V + 5.0)/5.0) + 1.0);
                    const double beta_h = (V < 40.0*(-1)) ? 2.7*exp(0.079*V) + 310000.0*exp(0.3485*V) : 0.77/((0.13*(exp((V + 10.66)/((11.1*(-1)))) + 1.0)));
                    const double beta_j = (V < 40.0*(-1)) ? (0.02424*exp(V*(0.01052*(-1))))/(exp((0.1378*(-1))*(V + 40.14)) + 1.0) : (0.6*exp(0.057*V))/(exp((0.1*(-1))*(V + 32.0)) + 1.0);
                    const double beta_m = 0.1/(exp((V + 35.0)/5.0) + 1.0) + 0.1/(exp((V - 1*50.0)/200.0) + 1.0);
                    const double beta_xr1 = 6.0/(exp((V + 30.0)/11.5) + 1.0);
                    const double beta_xr2 = 1.12/(exp((V - 1*60.0)/20.0) + 1.0);
                    const double beta_xs = 1.0/(exp((V - 1*35.0)/15.0) + 1.0);
                    const double d_inf = 1.0/(exp((-V - 1*8.0)/7.5) + 1.0);
                    const double f2_inf = 0.33 + 0.67/(exp((V + 35.0)/7.0) + 1.0);
                    const double f_inf = 1.0/(exp((V + 20.0)/7.0) + 1.0);
                    const double gamma_d = 1.0/(exp((50.0 - V)/20.0) + 1.0);
                    const double h_inf = 1.0/pow(exp((V + 71.55)/7.43) + 1.0, 2.0);
                    const double j_inf = 1.0/pow(exp((V + 71.55)/7.43) + 1.0, 2.0);
                    const double m_inf = 1.0/pow(exp((-V - 1*56.86)/9.03) + 1.0, 2.0);
                    const double r_inf = 1.0/(exp((20.0 - V)/6.0) + 1.0);
                    const double s_inf = 1.0/(exp((V + 28.0)/5.0) + 1.0);
                    const double tau_f = ((1102.5*exp((-pow(V + 27.0, 2.0))/225.0) + 200.0/(exp((13.0 - V)/10.0) + 1.0)) + 180.0/(exp((V + 30.0)/10.0) + 1.0)) + 20.0;
                    const double tau_f2 = (562.0*exp((-pow(V + 27.0, 2.0))/240.0) + 31.0/(exp((25.0 - V)/10.0) + 1.0)) + 80.0/(exp((V + 30.0)/10.0) + 1.0);
                    const double tau_r = 9.5*exp((-pow(V + 40.0, 2.0))/1800.0) + 0.8;
                    const double tau_s = 1000.0*exp((-pow(V + 67.0, 2.0))/1000.0) + 8.0;
                    const double xr1_inf = 1.0/(exp((-V - 1*26.0)/7.0) + 1.0);
                    const double xr2_inf = 1.0/(exp((V + 88.0)/24.0) + 1.0);
                    const double xs_inf = 1.0/(exp((-V - 1*5.0)/14.0) + 1.0);
                    const double i_CaL = (((pow(F, 2.0)*((4.0*(fCass*(f2*(f*(d*g_CaL)))))*(V - 1*15.0)))/((R*T)))*(-Ca_o + (0.25*Ca_ss)*exp((F*(2.0*(V - 1*15.0)))/((R*T)))))/(exp((F*(2.0*(V - 1*15.0)))/((R*T))) - 1*1.0);
                    const double i_NaCa = (K_NaCa*(Ca_o*(pow(Na_i, 3.0)*exp((F*(V*gamma_))/((R*T)))) - Ca_i*alpha*pow(Na_o, 3.0)*exp((F*(V*(gamma_ - 1*1.0)))/((R*T)))))/((((Ca_o + Km_Ca)*(pow(Km_Nai, 3.0) + pow(Na_o, 3.0)))*(K_sat*exp((F*(V*(gamma_ - 1*1.0)))/((R*T))) + 1.0)));
                    const double i_NaK = ((Na_i*((K_o*P_NaK)/(K_mk + K_o)))/(K_mNa + Na_i))/((0.1245*exp((F*(V*(0.1*(-1))))/((R*T))) + 1.0) + 0.0353*exp((F*(-V))/((R*T))));
                    const double i_Stim = (stim_end >= t && stim_duration + stim_start >= (-stim_period)*floor(t/stim_period) + t && stim_start <= (-stim_period)*floor(t/stim_period) + t) ? -stim_amplitude : 0.0;
                    const double i_leak = V_leak*(Ca_SR - Ca_i);
                    const double i_p_Ca = (Ca_i*g_pCa)/(Ca_i + K_pCa);
                    const double i_up = Vmax_up/(1.0 + pow(K_up, 2.0)/pow(Ca_i, 2.0));
                    const double i_xfer = V_xfer*(-Ca_i + Ca_ss);
                    const double kcasr = max_sr - (max_sr - min_sr)/(pow(EC/Ca_SR, 2.0) + 1.0);
                    const double dfCass_dt = (-fCass + fCass_inf)/tau_fCass;
                    values[0] = dfCass_dt*dt + fCass;
                    const double i_b_Ca = g_bca*(-E_Ca + V);
                    const double alpha_K1 = 0.1/(exp(0.06*((-E_K + V) - 1*200.0)) + 1.0);
                    const double beta_K1 = (3.0*exp(0.0002*((-E_K + V) + 100.0)) + exp(0.1*((-E_K + V) - 1*10.0)))/(exp((0.5*(-1))*(-E_K + V)) + 1.0);
                    const double i_Kr = (Xr2*(Xr1*((0.4303314829119352*sqrt(K_o))*g_Kr)))*(-E_K + V);
                    const double i_p_K = (g_pK*(-E_K + V))/(exp((25.0 - V)/5.98) + 1.0);
                    const double i_to = (s*(g_to*r))*(-E_K + V);
                    const double i_Ks = (pow(Xs, 2.0)*g_Ks)*(-E_Ks + V);
                    const double i_Na = (j*(h*(g_Na*pow(m, 3.0))))*(-E_Na + V);
                    const double i_b_Na = g_bna*(-E_Na + V);
                    const double tau_h = 1.0/(alpha_h + beta_h);
                    const double tau_j = 1.0/(alpha_j + beta_j);
                    const double tau_m = (1.0*alpha_m)*beta_m;
                    const double tau_xr1 = (1.0*alpha_xr1)*beta_xr1;
                    const double tau_xr2 = (1.0*alpha_xr2)*beta_xr2;
                    const double tau_xs = (1.0*alpha_xs)*beta_xs + 80.0;
                    const double tau_d = (1.0*alpha_d)*beta_d + gamma_d;
                    const double df_dt = (-f + f_inf)/tau_f;
                    values[1] = df_dt*dt + f;
                    const double df2_dt = (-f2 + f2_inf)/tau_f2;
                    values[2] = df2_dt*dt + f2;
                    const double dr_dt = (-r + r_inf)/tau_r;
                    values[3] = dr_dt*dt + r;
                    const double ds_dt = (-s + s_inf)/tau_s;
                    values[4] = ds_dt*dt + s;
                    const double k1 = k1_prime/kcasr;
                    const double k2 = k2_prime*kcasr;
                    const double dCa_i_dt = Ca_i_bufc*(((-Cm)*(1.0*((-2.0)*i_NaCa + (i_b_Ca + i_p_Ca))))/((F*((1.0*2.0)*V_c))) + (i_xfer + (V_sr*(i_leak - i_up))/V_c));
                    values[5] = Ca_i + dCa_i_dt*dt;
                    const double xK1_inf = alpha_K1/(alpha_K1 + beta_K1);
                    const double dNa_i_dt = Cm*(((1.0*(-1))*(3.0*i_NaCa + (3.0*i_NaK + (i_Na + i_b_Na))))/((F*(1.0*V_c))));
                    values[6] = Na_i + dNa_i_dt*dt;
                    const double dh_dt = (-h + h_inf)/tau_h;
                    values[7] = dh_dt*dt + h;
                    const double dj_dt = (-j + j_inf)/tau_j;
                    values[8] = dj_dt*dt + j;
                    const double dm_dt = (-m + m_inf)/tau_m;
                    values[9] = dm_dt*dt + m;
                    const double dXr1_dt = (-Xr1 + xr1_inf)/tau_xr1;
                    values[10] = Xr1 + dXr1_dt*dt;
                    const double dXr2_dt = (-Xr2 + xr2_inf)/tau_xr2;
                    values[11] = Xr2 + dXr2_dt*dt;
                    const double dXs_dt = (-Xs + xs_inf)/tau_xs;
                    values[12] = Xs + dXs_dt*dt;
                    const double dd_dt = (-d + d_inf)/tau_d;
                    values[13] = d + dd_dt*dt;
                    const double O_ = (R_prime*(pow(Ca_ss, 2.0)*k1))/(pow(Ca_ss, 2.0)*k1 + k3);
                    const double dR_prime_dt = R_prime*(Ca_ss*(-k2)) + k4*(1.0 - R_prime);
                    values[14] = R_prime + dR_prime_dt*dt;
                    const double i_K1 = ((0.4303314829119352*sqrt(K_o))*(g_K1*xK1_inf))*(-E_K + V);
                    const double i_rel = (O_*V_rel)*(Ca_SR - Ca_ss);
                    const double dK_i_dt = Cm*(((1.0*(-1))*((-2.0)*i_NaK + (i_Stim + (i_p_K + (i_Ks + (i_Kr + (i_K1 + i_to)))))))/((F*(1.0*V_c))));
                    values[15] = K_i + dK_i_dt*dt;
                    const double dV_dt = ((1.0*(-1))/1.0)*(i_Stim + (i_p_Ca + (i_p_K + (i_b_Ca + (i_NaCa + (i_b_Na + (i_Na + (i_NaK + (i_CaL + (i_Ks + (i_Kr + (i_K1 + i_to))))))))))));
                    values[16] = V + dV_dt*dt;
                    const double dCa_SR_dt = Ca_sr_bufsr*(i_up - (i_leak + i_rel));
                    values[17] = Ca_SR + dCa_SR_dt*dt;
                    const double dCa_ss_dt = Ca_ss_bufss*(((Cm*(i_CaL*(1.0*(-1))))/((F*((1.0*2.0)*V_ss))) + (V_sr*i_rel)/V_ss) - (V_c*i_xfer)/V_ss);
                    values[18] = Ca_ss + dCa_ss_dt*dt;
                }

                MFEM_HOST_DEVICE static void generalized_rush_larsen(const double *__restrict states, const double t, const double dt, const double *__restrict parameters, double* values){

                    // Assign states
                    const double fCass = states[0];
                    const double f = states[1];
                    const double f2 = states[2];
                    const double r = states[3];
                    const double s = states[4];
                    const double Ca_i = states[5];
                    const double Na_i = states[6];
                    const double h = states[7];
                    const double j = states[8];
                    const double m = states[9];
                    const double Xr1 = states[10];
                    const double Xr2 = states[11];
                    const double Xs = states[12];
                    const double d = states[13];
                    const double R_prime = states[14];
                    const double K_i = states[15];
                    const double V = states[16];
                    const double Ca_SR = states[17];
                    const double Ca_ss = states[18];

                    // Assign parameters
                    const double Buf_c = parameters[0];
                    const double Buf_sr = parameters[1];
                    const double Buf_ss = parameters[2];
                    const double Ca_o = parameters[3];
                    const double Cm = parameters[4];
                    const double EC = parameters[5];
                    const double F = parameters[6];
                    const double K_NaCa = parameters[7];
                    const double K_buf_c = parameters[8];
                    const double K_buf_sr = parameters[9];
                    const double K_buf_ss = parameters[10];
                    const double K_mNa = parameters[11];
                    const double K_mk = parameters[12];
                    const double K_o = parameters[13];
                    const double K_pCa = parameters[14];
                    const double K_sat = parameters[15];
                    const double K_up = parameters[16];
                    const double Km_Ca = parameters[17];
                    const double Km_Nai = parameters[18];
                    const double Na_o = parameters[19];
                    const double P_NaK = parameters[20];
                    const double P_kna = parameters[21];
                    const double R = parameters[22];
                    const double T = parameters[23];
                    const double V_c = parameters[24];
                    const double V_leak = parameters[25];
                    const double V_rel = parameters[26];
                    const double V_sr = parameters[27];
                    const double V_ss = parameters[28];
                    const double V_xfer = parameters[29];
                    const double Vmax_up = parameters[30];
                    const double alpha = parameters[31];
                    const double g_CaL = parameters[32];
                    const double g_K1 = parameters[33];
                    const double g_Kr = parameters[34];
                    const double g_Ks = parameters[35];
                    const double g_Na = parameters[36];
                    const double g_bca = parameters[37];
                    const double g_bna = parameters[38];
                    const double g_pCa = parameters[39];
                    const double g_pK = parameters[40];
                    const double g_to = parameters[41];
                    const double gamma_ = parameters[42];
                    const double k1_prime = parameters[43];
                    const double k2_prime = parameters[44];
                    const double k3 = parameters[45];
                    const double k4 = parameters[46];
                    const double max_sr = parameters[47];
                    const double min_sr = parameters[48];
                    const double stim_amplitude = parameters[49];
                    const double stim_duration = parameters[50];
                    const double stim_end = parameters[51];
                    const double stim_period = parameters[52];
                    const double stim_start = parameters[53];

                    // Assign expressions
                    const double Ca_i_bufc = 1.0/((Buf_c*K_buf_c)/pow(Ca_i + K_buf_c, 2.0) + 1.0);
                    const double Ca_sr_bufsr = 1.0/((Buf_sr*K_buf_sr)/pow(Ca_SR + K_buf_sr, 2.0) + 1.0);
                    const double fCass_inf = 0.4 + 0.6/(pow(Ca_ss/0.05, 2.0) + 1.0);
                    const double tau_fCass = 2.0 + 80.0/(pow(Ca_ss/0.05, 2.0) + 1.0);
                    const double Ca_ss_bufss = 1.0/((Buf_ss*K_buf_ss)/pow(Ca_ss + K_buf_ss, 2.0) + 1.0);
                    const double E_Ca = (((0.5*R)*T)/F)*log(Ca_o/Ca_i);
                    const double E_K = ((R*T)/F)*log(K_o/K_i);
                    const double E_Ks = ((R*T)/F)*log((K_o + Na_o*P_kna)/(K_i + Na_i*P_kna));
                    const double E_Na = ((R*T)/F)*log(Na_o/Na_i);
                    const double alpha_d = 0.25 + 1.4/(exp((-V - 1*35.0)/13.0) + 1.0);
                    const double alpha_h = (V < 40.0*(-1)) ? 0.057*exp((-(V + 80.0))/6.8) : 0.0;
                    const double alpha_j = (V < 40.0*(-1)) ? (((V + 37.78)*((25428.0*(-1))*exp(0.2444*V) - 6.948e-06*exp(V*(0.04391*(-1)))))/1.0)/(exp(0.311*(V + 79.23)) + 1.0) : 0.0;
                    const double alpha_m = 1.0/(exp((-V - 1*60.0)/5.0) + 1.0);
                    const double alpha_xr1 = 450.0/(exp((-V - 1*45.0)/10.0) + 1.0);
                    const double alpha_xr2 = 3.0/(exp((-V - 1*60.0)/20.0) + 1.0);
                    const double alpha_xs = 1400.0/sqrt(exp((5.0 - V)/6.0) + 1.0);
                    const double beta_d = 1.4/(exp((V + 5.0)/5.0) + 1.0);
                    const double beta_h = (V < 40.0*(-1)) ? 2.7*exp(0.079*V) + 310000.0*exp(0.3485*V) : 0.77/((0.13*(exp((V + 10.66)/((11.1*(-1)))) + 1.0)));
                    const double beta_j = (V < 40.0*(-1)) ? (0.02424*exp(V*(0.01052*(-1))))/(exp((0.1378*(-1))*(V + 40.14)) + 1.0) : (0.6*exp(0.057*V))/(exp((0.1*(-1))*(V + 32.0)) + 1.0);
                    const double beta_m = 0.1/(exp((V + 35.0)/5.0) + 1.0) + 0.1/(exp((V - 1*50.0)/200.0) + 1.0);
                    const double beta_xr1 = 6.0/(exp((V + 30.0)/11.5) + 1.0);
                    const double beta_xr2 = 1.12/(exp((V - 1*60.0)/20.0) + 1.0);
                    const double beta_xs = 1.0/(exp((V - 1*35.0)/15.0) + 1.0);
                    const double d_inf = 1.0/(exp((-V - 1*8.0)/7.5) + 1.0);
                    const double f2_inf = 0.33 + 0.67/(exp((V + 35.0)/7.0) + 1.0);
                    const double f_inf = 1.0/(exp((V + 20.0)/7.0) + 1.0);
                    const double gamma_d = 1.0/(exp((50.0 - V)/20.0) + 1.0);
                    const double h_inf = 1.0/pow(exp((V + 71.55)/7.43) + 1.0, 2.0);
                    const double j_inf = 1.0/pow(exp((V + 71.55)/7.43) + 1.0, 2.0);
                    const double m_inf = 1.0/pow(exp((-V - 1*56.86)/9.03) + 1.0, 2.0);
                    const double r_inf = 1.0/(exp((20.0 - V)/6.0) + 1.0);
                    const double s_inf = 1.0/(exp((V + 28.0)/5.0) + 1.0);
                    const double tau_f = ((1102.5*exp((-pow(V + 27.0, 2.0))/225.0) + 200.0/(exp((13.0 - V)/10.0) + 1.0)) + 180.0/(exp((V + 30.0)/10.0) + 1.0)) + 20.0;
                    const double tau_f2 = (562.0*exp((-pow(V + 27.0, 2.0))/240.0) + 31.0/(exp((25.0 - V)/10.0) + 1.0)) + 80.0/(exp((V + 30.0)/10.0) + 1.0);
                    const double tau_r = 9.5*exp((-pow(V + 40.0, 2.0))/1800.0) + 0.8;
                    const double tau_s = 1000.0*exp((-pow(V + 67.0, 2.0))/1000.0) + 8.0;
                    const double xr1_inf = 1.0/(exp((-V - 1*26.0)/7.0) + 1.0);
                    const double xr2_inf = 1.0/(exp((V + 88.0)/24.0) + 1.0);
                    const double xs_inf = 1.0/(exp((-V - 1*5.0)/14.0) + 1.0);
                    const double i_CaL = (((pow(F, 2.0)*((4.0*(fCass*(f2*(f*(d*g_CaL)))))*(V - 1*15.0)))/((R*T)))*(-Ca_o + (0.25*Ca_ss)*exp((F*(2.0*(V - 1*15.0)))/((R*T)))))/(exp((F*(2.0*(V - 1*15.0)))/((R*T))) - 1*1.0);
                    const double i_NaCa = (K_NaCa*(Ca_o*(pow(Na_i, 3.0)*exp((F*(V*gamma_))/((R*T)))) - Ca_i*alpha*pow(Na_o, 3.0)*exp((F*(V*(gamma_ - 1*1.0)))/((R*T)))))/((((Ca_o + Km_Ca)*(pow(Km_Nai, 3.0) + pow(Na_o, 3.0)))*(K_sat*exp((F*(V*(gamma_ - 1*1.0)))/((R*T))) + 1.0)));
                    const double i_NaK = ((Na_i*((K_o*P_NaK)/(K_mk + K_o)))/(K_mNa + Na_i))/((0.1245*exp((F*(V*(0.1*(-1))))/((R*T))) + 1.0) + 0.0353*exp((F*(-V))/((R*T))));
                    const double i_Stim = (stim_end >= t && stim_duration + stim_start >= (-stim_period)*floor(t/stim_period) + t && stim_start <= (-stim_period)*floor(t/stim_period) + t) ? -stim_amplitude : 0.0;
                    const double i_leak = V_leak*(Ca_SR - Ca_i);
                    const double i_p_Ca = (Ca_i*g_pCa)/(Ca_i + K_pCa);
                    const double i_up = Vmax_up/(1.0 + pow(K_up, 2.0)/pow(Ca_i, 2.0));
                    const double i_xfer = V_xfer*(-Ca_i + Ca_ss);
                    const double kcasr = max_sr - (max_sr - min_sr)/(pow(EC/Ca_SR, 2.0) + 1.0);
                    const double dfCass_dt = (-fCass + fCass_inf)/tau_fCass;
                    const double dfCass_dt_linearized = -1/tau_fCass;
                    values[0] = dfCass_dt*(exp(dfCass_dt_linearized*dt) - 1)/dfCass_dt_linearized + fCass;
                    const double i_b_Ca = g_bca*(-E_Ca + V);
                    const double alpha_K1 = 0.1/(exp(0.06*((-E_K + V) - 1*200.0)) + 1.0);
                    const double beta_K1 = (3.0*exp(0.0002*((-E_K + V) + 100.0)) + exp(0.1*((-E_K + V) - 1*10.0)))/(exp((0.5*(-1))*(-E_K + V)) + 1.0);
                    const double i_Kr = (Xr2*(Xr1*((0.4303314829119352*sqrt(K_o))*g_Kr)))*(-E_K + V);
                    const double i_p_K = (g_pK*(-E_K + V))/(exp((25.0 - V)/5.98) + 1.0);
                    const double i_to = (s*(g_to*r))*(-E_K + V);
                    const double i_Ks = (pow(Xs, 2.0)*g_Ks)*(-E_Ks + V);
                    const double i_Na = (j*(h*(g_Na*pow(m, 3.0))))*(-E_Na + V);
                    const double i_b_Na = g_bna*(-E_Na + V);
                    const double tau_h = 1.0/(alpha_h + beta_h);
                    const double tau_j = 1.0/(alpha_j + beta_j);
                    const double tau_m = (1.0*alpha_m)*beta_m;
                    const double tau_xr1 = (1.0*alpha_xr1)*beta_xr1;
                    const double tau_xr2 = (1.0*alpha_xr2)*beta_xr2;
                    const double tau_xs = (1.0*alpha_xs)*beta_xs + 80.0;
                    const double tau_d = (1.0*alpha_d)*beta_d + gamma_d;
                    const double df_dt = (-f + f_inf)/tau_f;
                    const double df_dt_linearized = -1/tau_f;
                    values[1] = df_dt*(exp(df_dt_linearized*dt) - 1)/df_dt_linearized + f;
                    const double df2_dt = (-f2 + f2_inf)/tau_f2;
                    const double df2_dt_linearized = -1/tau_f2;
                    values[2] = df2_dt*(exp(df2_dt_linearized*dt) - 1)/df2_dt_linearized + f2;
                    const double dr_dt = (-r + r_inf)/tau_r;
                    const double dr_dt_linearized = -1/tau_r;
                    values[3] = dr_dt*(exp(dr_dt_linearized*dt) - 1)/dr_dt_linearized + r;
                    const double ds_dt = (-s + s_inf)/tau_s;
                    const double ds_dt_linearized = -1/tau_s;
                    values[4] = ds_dt*(exp(ds_dt_linearized*dt) - 1)/ds_dt_linearized + s;
                    const double k1 = k1_prime/kcasr;
                    const double k2 = k2_prime*kcasr;
                    const double dCa_i_dt = Ca_i_bufc*(((-Cm)*(1.0*((-2.0)*i_NaCa + (i_b_Ca + i_p_Ca))))/((F*((1.0*2.0)*V_c))) + (i_xfer + (V_sr*(i_leak - i_up))/V_c));
                    values[5] = Ca_i + dCa_i_dt*dt;
                    const double xK1_inf = alpha_K1/(alpha_K1 + beta_K1);
                    const double dNa_i_dt = Cm*(((1.0*(-1))*(3.0*i_NaCa + (3.0*i_NaK + (i_Na + i_b_Na))))/((F*(1.0*V_c))));
                    values[6] = Na_i + dNa_i_dt*dt;
                    const double dh_dt = (-h + h_inf)/tau_h;
                    const double dh_dt_linearized = -1/tau_h;
                    values[7] = dh_dt*(exp(dh_dt_linearized*dt) - 1)/dh_dt_linearized + h;
                    const double dj_dt = (-j + j_inf)/tau_j;
                    const double dj_dt_linearized = -1/tau_j;
                    values[8] = dj_dt*(exp(dj_dt_linearized*dt) - 1)/dj_dt_linearized + j;
                    const double dm_dt = (-m + m_inf)/tau_m;
                    const double dm_dt_linearized = -1/tau_m;
                    values[9] = dm_dt*(exp(dm_dt_linearized*dt) - 1)/dm_dt_linearized + m;
                    const double dXr1_dt = (-Xr1 + xr1_inf)/tau_xr1;
                    const double dXr1_dt_linearized = -1/tau_xr1;
                    values[10] = Xr1 + dXr1_dt*(exp(dXr1_dt_linearized*dt) - 1)/dXr1_dt_linearized;
                    const double dXr2_dt = (-Xr2 + xr2_inf)/tau_xr2;
                    const double dXr2_dt_linearized = -1/tau_xr2;
                    values[11] = Xr2 + dXr2_dt*(exp(dXr2_dt_linearized*dt) - 1)/dXr2_dt_linearized;
                    const double dXs_dt = (-Xs + xs_inf)/tau_xs;
                    const double dXs_dt_linearized = -1/tau_xs;
                    values[12] = Xs + dXs_dt*(exp(dXs_dt_linearized*dt) - 1)/dXs_dt_linearized;
                    const double dd_dt = (-d + d_inf)/tau_d;
                    const double dd_dt_linearized = -1/tau_d;
                    values[13] = d + dd_dt*(exp(dd_dt_linearized*dt) - 1)/dd_dt_linearized;
                    const double O_ = (R_prime*(pow(Ca_ss, 2.0)*k1))/(pow(Ca_ss, 2.0)*k1 + k3);
                    const double dR_prime_dt = R_prime*(Ca_ss*(-k2)) + k4*(1.0 - R_prime);
                    const double dR_prime_dt_linearized = Ca_ss*(-k2) - k4;
                    values[14] = R_prime + ((fabs(dR_prime_dt_linearized) > 1e-08) ? (
                       dR_prime_dt*(exp(dR_prime_dt_linearized*dt) - 1)/dR_prime_dt_linearized
                    )
                    : (
                       dR_prime_dt*dt
                    ));
                    const double i_K1 = ((0.4303314829119352*sqrt(K_o))*(g_K1*xK1_inf))*(-E_K + V);
                    const double i_rel = (O_*V_rel)*(Ca_SR - Ca_ss);
                    const double dK_i_dt = Cm*(((1.0*(-1))*((-2.0)*i_NaK + (i_Stim + (i_p_K + (i_Ks + (i_Kr + (i_K1 + i_to)))))))/((F*(1.0*V_c))));
                    values[15] = K_i + dK_i_dt*dt;
                    const double dV_dt = ((1.0*(-1))/1.0)*(i_Stim + (i_p_Ca + (i_p_K + (i_b_Ca + (i_NaCa + (i_b_Na + (i_Na + (i_NaK + (i_CaL + (i_Ks + (i_Kr + (i_K1 + i_to))))))))))));
                    values[16] = V + dV_dt*dt;
                    const double dCa_SR_dt = Ca_sr_bufsr*(i_up - (i_leak + i_rel));
                    values[17] = Ca_SR + dCa_SR_dt*dt;
                    const double dCa_ss_dt = Ca_ss_bufss*(((Cm*(i_CaL*(1.0*(-1))))/((F*((1.0*2.0)*V_ss))) + (V_sr*i_rel)/V_ss) - (V_c*i_xfer)/V_ss);
                    values[18] = Ca_ss + dCa_ss_dt*dt;
                }
            };
        };

    } // namespace electrophysiology
} // namespace mfem
