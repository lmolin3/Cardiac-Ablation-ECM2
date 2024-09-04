#include <cmath>
#include <cstring>
#include <stdexcept>
#include "ode_model.hpp"

class ODEModel {
public:
    virtual ~ODEModel() {}

    virtual void init_state_values(double* states) = 0;
    virtual void init_parameters_values(double* parameters) = 0;
    virtual int state_index(const char name[]) = 0;
    virtual int parameter_index(const char name[]) = 0;
    virtual void rhs(const double *__restrict states, const double t, const double *__restrict parameters, double* values) = 0;
    virtual void forward_explicit_euler(double *__restrict states, const double t, const double dt, const double *__restrict parameters) = 0;
    virtual void forward_rush_larsen(double *__restrict states, const double t, const double dt, const double *__restrict parameters) = 0;
    virtual void forward_generalized_rush_larsen(double *__restrict states, const double t, const double dt, const double *__restrict parameters) = 0;
};



// Gotran generated C/C++ code for the "mitchell_schaeffer_2003" model (converted from CellML)
class MitchellSchaeffer : public ODEModel {
public:

    // Init state values
    void init_state_values(double* states) override
    {
    states[0] = 0.8789655121804799; // h;
    states[1] = 8.20413566106744e-06; // Vm;
    }

    // Default parameter values
    void init_parameters_values(double* parameters) override
    {
    parameters[0] = 0.2; // IstimAmplitude;
    parameters[1] = 50000.0; // IstimEnd;
    parameters[2] = 500.0; // IstimPeriod;
    parameters[3] = 1.0; // IstimPulseDuration;
    parameters[4] = 0.0; // IstimStart;
    parameters[5] = 0.3; // tau_in;
    parameters[6] = 0.13; // V_gate;
    parameters[7] = 150.0; // tau_close;
    parameters[8] = 120.0; // tau_open;
    parameters[9] = 6.0; // tau_out;
    }

    // State index
    int state_index(const char name[]) override
    {
    // State names
    char names[][3] = {"h", "Vm"};

    int i;
    for (i=0; i<2; i++)
    {
        if (strcmp(names[i], name)==0)
        {
        return i;
        }
    }
    return -1;
    }

    // Parameter index
    int parameter_index(const char name[]) override
    {
    // Parameter names
    char names[][19] = {"IstimAmplitude", "IstimEnd", "IstimPeriod",
        "IstimPulseDuration", "IstimStart", "tau_in", "V_gate", "tau_close",
        "tau_open", "tau_out"};

    int i;
    for (i=0; i<10; i++)
    {
        if (strcmp(names[i], name)==0)
        {
        return i;
        }
    }
    return -1;
    }

    // Compute the right hand side of the mitchell_schaeffer_2003 ODE
    void rhs(const double *__restrict states, const double t, const double
    *__restrict parameters, double* values) override
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
    const double tau_in = parameters[5];
    const double V_gate = parameters[6];
    const double tau_close = parameters[7];
    const double tau_open = parameters[8];
    const double tau_out = parameters[9];

    // Expressions for the J_stim component
    const double J_stim = (t - IstimStart - IstimPeriod*std::floor((t -
        IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
        IstimStart ? IstimAmplitude : 0.);

    // Expressions for the J_in component
    const double J_in = (Vm*Vm)*(1. - Vm)*h/tau_in;

    // Expressions for the h gate component
    values[0] = (Vm < V_gate ? (1. - h)/tau_open : -h/tau_close);

    // Expressions for the J_out component
    const double J_out = -Vm/tau_out;

    // Expressions for the Membrane component
    values[1] = J_in + J_out + J_stim;
    }

    // Compute a forward step using the explicit Euler scheme to the
    // mitchell_schaeffer_2003 ODE
    void forward_explicit_euler(double *__restrict states, const double t, const
    double dt, const double *__restrict parameters) override
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
    const double tau_in = parameters[5];
    const double V_gate = parameters[6];
    const double tau_close = parameters[7];
    const double tau_open = parameters[8];
    const double tau_out = parameters[9];

    // Expressions for the J_stim component
    const double J_stim = (t - IstimStart - IstimPeriod*std::floor((t -
        IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
        IstimStart ? IstimAmplitude : 0.);

    // Expressions for the J_in component
    const double J_in = (Vm*Vm)*(1. - Vm)*h/tau_in;

    // Expressions for the h gate component
    const double dh_dt = (Vm < V_gate ? (1. - h)/tau_open : -h/tau_close);
    states[0] = dt*dh_dt + h;

    // Expressions for the J_out component
    const double J_out = -Vm/tau_out;

    // Expressions for the Membrane component
    const double dVm_dt = J_in + J_out + J_stim;
    states[1] = dt*dVm_dt + Vm;
    }

    // Compute a forward step using the Rush-Larsen scheme to the
    // mitchell_schaeffer_2003 ODE
    void forward_rush_larsen(double *__restrict states, const double t, const
    double dt, const double *__restrict parameters) override
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
    const double tau_in = parameters[5];
    const double V_gate = parameters[6];
    const double tau_close = parameters[7];
    const double tau_open = parameters[8];
    const double tau_out = parameters[9];

    // Expressions for the J_stim component
    const double J_stim = (t - IstimStart - IstimPeriod*std::floor((t -
        IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
        IstimStart ? IstimAmplitude : 0.);

    // Expressions for the J_in component
    const double J_in = (Vm*Vm)*(1. - Vm)*h/tau_in;

    // Expressions for the h gate component
    const double dh_dt = (Vm < V_gate ? (1. - h)/tau_open : -h/tau_close);
    const double dh_dt_linearized = (Vm < V_gate ? -1./tau_open : -1./tau_close);
    states[0] = (std::fabs(dh_dt_linearized) > 1.0e-8 ?
        (std::expm1(dt*dh_dt_linearized))*dh_dt/dh_dt_linearized : dt*dh_dt) + h;

    // Expressions for the J_out component
    const double J_out = -Vm/tau_out;

    // Expressions for the Membrane component
    const double dVm_dt = J_in + J_out + J_stim;
    const double dJ_in_dVm = -(Vm*Vm)*h/tau_in + 2.*(1. - Vm)*Vm*h/tau_in;
    const double dJ_out_dVm = -1./tau_out;
    const double dVm_dt_linearized = dJ_in_dVm + dJ_out_dVm;
    states[1] = (std::fabs(dVm_dt_linearized) > 1.0e-8 ?
        (std::expm1(dt*dVm_dt_linearized))*dVm_dt/dVm_dt_linearized : dt*dVm_dt)
        + Vm;
    }

    // Compute a forward step using the generalised Rush-Larsen (GRL1) scheme to
    // the mitchell_schaeffer_2003 ODE
    void forward_generalized_rush_larsen(double *__restrict states, const double
    t, const double dt, const double *__restrict parameters) override
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
    const double tau_in = parameters[5];
    const double V_gate = parameters[6];
    const double tau_close = parameters[7];
    const double tau_open = parameters[8];
    const double tau_out = parameters[9];

    // Expressions for the J_stim component
    const double J_stim = (t - IstimStart - IstimPeriod*std::floor((t -
        IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
        IstimStart ? IstimAmplitude : 0.);

    // Expressions for the J_in component
    const double J_in = (Vm*Vm)*(1. - Vm)*h/tau_in;

    // Expressions for the h gate component
    const double dh_dt = (Vm < V_gate ? (1. - h)/tau_open : -h/tau_close);
    const double dh_dt_linearized = (Vm < V_gate ? -1./tau_open : -1./tau_close);
    states[0] = (std::fabs(dh_dt_linearized) > 1.0e-8 ?
        (std::expm1(dt*dh_dt_linearized))*dh_dt/dh_dt_linearized : dt*dh_dt) + h;

    // Expressions for the J_out component
    const double J_out = -Vm/tau_out;

    // Expressions for the Membrane component
    const double dVm_dt = J_in + J_out + J_stim;
    const double dJ_in_dVm = -(Vm*Vm)*h/tau_in + 2.*(1. - Vm)*Vm*h/tau_in;
    const double dJ_out_dVm = -1./tau_out;
    const double dVm_dt_linearized = dJ_in_dVm + dJ_out_dVm;
    states[1] = (std::fabs(dVm_dt_linearized) > 1.0e-8 ?
        (std::expm1(dt*dVm_dt_linearized))*dVm_dt/dVm_dt_linearized : dt*dVm_dt)
        + Vm;
    }

};


// Gotran generated C/C++ code for the "fenton_karma_1998_BR" model (converted from CellML)
class FentonKarma : public ODEModel {
public:

    // Init state values
    void init_state_values(double* states) override
    {
    states[0] = 1.0; // v;
    states[1] = 1.0; // w;
    states[2] = 0.0; // u;
    }

    // Default parameter values
    void init_parameters_values(double* parameters) override
    {
    parameters[0] = 0.13; // u_c;
    parameters[1] = 0.04; // u_v;
    parameters[2] = 4.0; // g_fi_max;
    parameters[3] = 1250.0; // tau_v1_minus;
    parameters[4] = 19.6; // tau_v2_minus;
    parameters[5] = 3.33; // tau_v_plus;
    parameters[6] = 12.5; // tau_0;
    parameters[7] = 33.33; // tau_r;
    parameters[8] = 10.0; // k;
    parameters[9] = 29.0; // tau_si;
    parameters[10] = 0.85; // u_csi;
    parameters[11] = 41.0; // tau_w_minus;
    parameters[12] = 870.0; // tau_w_plus;
    parameters[13] = -0.2; // IstimAmplitude;
    parameters[14] = 50000.0; // IstimEnd;
    parameters[15] = 1000.0; // IstimPeriod;
    parameters[16] = 1.0; // IstimPulseDuration;
    parameters[17] = 10.0; // IstimStart;
    parameters[18] = 1.0; // Cm;
    parameters[19] = -85.0; // V_0;
    parameters[20] = 15.0; // V_fi;
    }

    // State index
    int state_index(const char name[]) override
    {
    // State names
    char names[][2] = {"v", "w", "u"};

    int i;
    for (i=0; i<3; i++)
    {
        if (strcmp(names[i], name)==0)
        {
        return i;
        }
    }
    return -1;
    }

    // Parameter index
    int parameter_index(const char name[]) override
    {
    // Parameter names
    char names[][19] = {"u_c", "u_v", "g_fi_max", "tau_v1_minus",
        "tau_v2_minus", "tau_v_plus", "tau_0", "tau_r", "k", "tau_si", "u_csi",
        "tau_w_minus", "tau_w_plus", "IstimAmplitude", "IstimEnd", "IstimPeriod",
        "IstimPulseDuration", "IstimStart", "Cm", "V_0", "V_fi"};

    int i;
    for (i=0; i<21; i++)
    {
        if (strcmp(names[i], name)==0)
        {
        return i;
        }
    }
    return -1;
    }

    // Compute the right hand side of the fenton_karma_1998_BR ODE
    void rhs(const double *__restrict states, const double t, const double
    *__restrict parameters, double* values) override
    {

    // Assign states
    const double v = states[0];
    const double w = states[1];
    const double u = states[2];

    // Assign parameters
    const double u_c = parameters[0];
    const double u_v = parameters[1];
    const double g_fi_max = parameters[2];
    const double tau_v1_minus = parameters[3];
    const double tau_v2_minus = parameters[4];
    const double tau_v_plus = parameters[5];
    const double tau_0 = parameters[6];
    const double tau_r = parameters[7];
    const double k = parameters[8];
    const double tau_si = parameters[9];
    const double u_csi = parameters[10];
    const double tau_w_minus = parameters[11];
    const double tau_w_plus = parameters[12];
    const double IstimAmplitude = parameters[13];
    const double IstimEnd = parameters[14];
    const double IstimPeriod = parameters[15];
    const double IstimPulseDuration = parameters[16];
    const double IstimStart = parameters[17];
    const double Cm = parameters[18];

    // Expressions for the p component
    const double p = (u < u_c ? 0. : 1.);

    // Expressions for the q component
    const double q = (u < u_v ? 0. : 1.);

    // Expressions for the Fast inward current component
    const double tau_d = Cm/g_fi_max;
    const double J_fi = -(1. - u)*(-u_c + u)*p*v/tau_d;

    // Expressions for the v gate component
    const double tau_v_minus = tau_v1_minus*q + tau_v2_minus*(1. - q);
    values[0] = (1. - p)*(1. - v)/tau_v_minus - p*v/tau_v_plus;

    // Expressions for the Slow outward current component
    const double J_so = p/tau_r + (1. - p)*u/tau_0;

    // Expressions for the Slow inward current component
    const double J_si = -(1. + std::tanh(k*(-u_csi + u)))*w/(2.*tau_si);

    // Expressions for the w gate component
    values[1] = (1. - p)*(1. - w)/tau_w_minus - p*w/tau_w_plus;

    // Expressions for the Stimulus protocol component
    const double J_stim = (t - IstimStart - IstimPeriod*std::floor((t -
        IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
        IstimStart ? IstimAmplitude : 0.);

    // Expressions for the Membrane component
    values[2] = -J_fi - J_si - J_so - J_stim;
    }

    // Compute a forward step using the explicit Euler scheme to the
    // fenton_karma_1998_BR ODE
    void forward_explicit_euler(double *__restrict states, const double t, const
    double dt, const double *__restrict parameters) override
    {

    // Assign states
    const double v = states[0];
    const double w = states[1];
    const double u = states[2];

    // Assign parameters
    const double u_c = parameters[0];
    const double u_v = parameters[1];
    const double g_fi_max = parameters[2];
    const double tau_v1_minus = parameters[3];
    const double tau_v2_minus = parameters[4];
    const double tau_v_plus = parameters[5];
    const double tau_0 = parameters[6];
    const double tau_r = parameters[7];
    const double k = parameters[8];
    const double tau_si = parameters[9];
    const double u_csi = parameters[10];
    const double tau_w_minus = parameters[11];
    const double tau_w_plus = parameters[12];
    const double IstimAmplitude = parameters[13];
    const double IstimEnd = parameters[14];
    const double IstimPeriod = parameters[15];
    const double IstimPulseDuration = parameters[16];
    const double IstimStart = parameters[17];
    const double Cm = parameters[18];

    // Expressions for the p component
    const double p = (u < u_c ? 0. : 1.);

    // Expressions for the q component
    const double q = (u < u_v ? 0. : 1.);

    // Expressions for the Fast inward current component
    const double tau_d = Cm/g_fi_max;
    const double J_fi = -(1. - u)*(-u_c + u)*p*v/tau_d;

    // Expressions for the v gate component
    const double tau_v_minus = tau_v1_minus*q + tau_v2_minus*(1. - q);
    const double dv_dt = (1. - p)*(1. - v)/tau_v_minus - p*v/tau_v_plus;
    states[0] = dt*dv_dt + v;

    // Expressions for the Slow outward current component
    const double J_so = p/tau_r + (1. - p)*u/tau_0;

    // Expressions for the Slow inward current component
    const double J_si = -(1. + std::tanh(k*(-u_csi + u)))*w/(2.*tau_si);

    // Expressions for the w gate component
    const double dw_dt = (1. - p)*(1. - w)/tau_w_minus - p*w/tau_w_plus;
    states[1] = dt*dw_dt + w;

    // Expressions for the Stimulus protocol component
    const double J_stim = (t - IstimStart - IstimPeriod*std::floor((t -
        IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
        IstimStart ? IstimAmplitude : 0.);

    // Expressions for the Membrane component
    const double du_dt = -J_fi - J_si - J_so - J_stim;
    states[2] = dt*du_dt + u;
    }

    // Compute a forward step using the Rush-Larsen scheme to the
    // fenton_karma_1998_BR ODE
    void forward_rush_larsen(double *__restrict states, const double t, const
    double dt, const double *__restrict parameters) override
    {

    // Assign states
    const double v = states[0];
    const double w = states[1];
    const double u = states[2];

    // Assign parameters
    const double u_c = parameters[0];
    const double u_v = parameters[1];
    const double g_fi_max = parameters[2];
    const double tau_v1_minus = parameters[3];
    const double tau_v2_minus = parameters[4];
    const double tau_v_plus = parameters[5];
    const double tau_0 = parameters[6];
    const double tau_r = parameters[7];
    const double k = parameters[8];
    const double tau_si = parameters[9];
    const double u_csi = parameters[10];
    const double tau_w_minus = parameters[11];
    const double tau_w_plus = parameters[12];
    const double IstimAmplitude = parameters[13];
    const double IstimEnd = parameters[14];
    const double IstimPeriod = parameters[15];
    const double IstimPulseDuration = parameters[16];
    const double IstimStart = parameters[17];
    const double Cm = parameters[18];

    // Expressions for the p component
    const double p = (u < u_c ? 0. : 1.);

    // Expressions for the q component
    const double q = (u < u_v ? 0. : 1.);

    // Expressions for the Fast inward current component
    const double tau_d = Cm/g_fi_max;
    const double J_fi = -(1. - u)*(-u_c + u)*p*v/tau_d;

    // Expressions for the v gate component
    const double tau_v_minus = tau_v1_minus*q + tau_v2_minus*(1. - q);
    const double dv_dt = (1. - p)*(1. - v)/tau_v_minus - p*v/tau_v_plus;
    const double dv_dt_linearized = -p/tau_v_plus - (1. - p)/tau_v_minus;
    states[0] = (std::fabs(dv_dt_linearized) > 1.0e-8 ?
        (std::expm1(dt*dv_dt_linearized))*dv_dt/dv_dt_linearized : dt*dv_dt) + v;

    // Expressions for the Slow outward current component
    const double J_so = p/tau_r + (1. - p)*u/tau_0;

    // Expressions for the Slow inward current component
    const double J_si = -(1. + std::tanh(k*(-u_csi + u)))*w/(2.*tau_si);

    // Expressions for the w gate component
    const double dw_dt = (1. - p)*(1. - w)/tau_w_minus - p*w/tau_w_plus;
    const double dw_dt_linearized = -(1. - p)/tau_w_minus - p/tau_w_plus;
    states[1] = (std::fabs(dw_dt_linearized) > 1.0e-8 ?
        (std::expm1(dt*dw_dt_linearized))*dw_dt/dw_dt_linearized : dt*dw_dt) + w;

    // Expressions for the Stimulus protocol component
    const double J_stim = (t - IstimStart - IstimPeriod*std::floor((t -
        IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
        IstimStart ? IstimAmplitude : 0.);

    // Expressions for the Membrane component
    const double du_dt = -J_fi - J_si - J_so - J_stim;
    const double dJ_fi_du = (-u_c + u)*p*v/tau_d - (1. - u)*p*v/tau_d;
    const double dJ_si_du = -k*(1. - (std::tanh(k*(-u_csi +
        u))*std::tanh(k*(-u_csi + u))))*w/(2.*tau_si);
    const double dJ_so_du = (1. - p)/tau_0;
    const double du_dt_linearized = -dJ_fi_du - dJ_si_du - dJ_so_du;
    states[2] = (std::fabs(du_dt_linearized) > 1.0e-8 ?
        (std::expm1(dt*du_dt_linearized))*du_dt/du_dt_linearized : dt*du_dt) + u;
    }

    // Compute a forward step using the generalised Rush-Larsen (GRL1) scheme to
    // the fenton_karma_1998_BR ODE
    void forward_generalized_rush_larsen(double *__restrict states, const double
    t, const double dt, const double *__restrict parameters) override
    {

    // Assign states
    const double v = states[0];
    const double w = states[1];
    const double u = states[2];

    // Assign parameters
    const double u_c = parameters[0];
    const double u_v = parameters[1];
    const double g_fi_max = parameters[2];
    const double tau_v1_minus = parameters[3];
    const double tau_v2_minus = parameters[4];
    const double tau_v_plus = parameters[5];
    const double tau_0 = parameters[6];
    const double tau_r = parameters[7];
    const double k = parameters[8];
    const double tau_si = parameters[9];
    const double u_csi = parameters[10];
    const double tau_w_minus = parameters[11];
    const double tau_w_plus = parameters[12];
    const double IstimAmplitude = parameters[13];
    const double IstimEnd = parameters[14];
    const double IstimPeriod = parameters[15];
    const double IstimPulseDuration = parameters[16];
    const double IstimStart = parameters[17];
    const double Cm = parameters[18];

    // Expressions for the p component
    const double p = (u < u_c ? 0. : 1.);

    // Expressions for the q component
    const double q = (u < u_v ? 0. : 1.);

    // Expressions for the Fast inward current component
    const double tau_d = Cm/g_fi_max;
    const double J_fi = -(1. - u)*(-u_c + u)*p*v/tau_d;

    // Expressions for the v gate component
    const double tau_v_minus = tau_v1_minus*q + tau_v2_minus*(1. - q);
    const double dv_dt = (1. - p)*(1. - v)/tau_v_minus - p*v/tau_v_plus;
    const double dv_dt_linearized = -p/tau_v_plus - (1. - p)/tau_v_minus;
    states[0] = (std::fabs(dv_dt_linearized) > 1.0e-8 ?
        (std::expm1(dt*dv_dt_linearized))*dv_dt/dv_dt_linearized : dt*dv_dt) + v;

    // Expressions for the Slow outward current component
    const double J_so = p/tau_r + (1. - p)*u/tau_0;

    // Expressions for the Slow inward current component
    const double J_si = -(1. + std::tanh(k*(-u_csi + u)))*w/(2.*tau_si);

    // Expressions for the w gate component
    const double dw_dt = (1. - p)*(1. - w)/tau_w_minus - p*w/tau_w_plus;
    const double dw_dt_linearized = -(1. - p)/tau_w_minus - p/tau_w_plus;
    states[1] = (std::fabs(dw_dt_linearized) > 1.0e-8 ?
        (std::expm1(dt*dw_dt_linearized))*dw_dt/dw_dt_linearized : dt*dw_dt) + w;

    // Expressions for the Stimulus protocol component
    const double J_stim = (t - IstimStart - IstimPeriod*std::floor((t -
        IstimStart)/IstimPeriod) <= IstimPulseDuration && t <= IstimEnd && t >=
        IstimStart ? IstimAmplitude : 0.);

    // Expressions for the Membrane component
    const double du_dt = -J_fi - J_si - J_so - J_stim;
    const double dJ_fi_du = (-u_c + u)*p*v/tau_d - (1. - u)*p*v/tau_d;
    const double dJ_si_du = -k*(1. - (std::tanh(k*(-u_csi +
        u))*std::tanh(k*(-u_csi + u))))*w/(2.*tau_si);
    const double dJ_so_du = (1. - p)/tau_0;
    const double du_dt_linearized = -dJ_fi_du - dJ_si_du - dJ_so_du;
    states[2] = (std::fabs(du_dt_linearized) > 1.0e-8 ?
        (std::expm1(dt*du_dt_linearized))*du_dt/du_dt_linearized : dt*du_dt) + u;
    }

};
