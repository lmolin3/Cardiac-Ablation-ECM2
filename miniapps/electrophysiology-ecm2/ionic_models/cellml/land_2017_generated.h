#include <math.h>
#include <string.h>

int NUM_STATES = 7;
int NUM_PARAMS = 26;
int NUM_MONITORED = 33;
// Parameter index
int parameter_index(const char name[])
{

    if (strcmp(name, "Beta0") == 0) {
        return 0;
    }


    else if (strcmp(name, "Beta1") == 0) {
        return 1;
    }


    else if (strcmp(name, "Tot_A") == 0) {
        return 2;
    }


    else if (strcmp(name, "Tref") == 0) {
        return 3;
    }


    else if (strcmp(name, "Trpn50") == 0) {
        return 4;
    }


    else if (strcmp(name, "cai") == 0) {
        return 5;
    }


    else if (strcmp(name, "cat50_ref") == 0) {
        return 6;
    }


    else if (strcmp(name, "dLambda") == 0) {
        return 7;
    }


    else if (strcmp(name, "etal") == 0) {
        return 8;
    }


    else if (strcmp(name, "etas") == 0) {
        return 9;
    }


    else if (strcmp(name, "gammas") == 0) {
        return 10;
    }


    else if (strcmp(name, "gammaw") == 0) {
        return 11;
    }


    else if (strcmp(name, "ktrpn") == 0) {
        return 12;
    }


    else if (strcmp(name, "ku") == 0) {
        return 13;
    }


    else if (strcmp(name, "kuw") == 0) {
        return 14;
    }


    else if (strcmp(name, "kws") == 0) {
        return 15;
    }


    else if (strcmp(name, "lmbda") == 0) {
        return 16;
    }


    else if (strcmp(name, "ntm") == 0) {
        return 17;
    }


    else if (strcmp(name, "ntrpn") == 0) {
        return 18;
    }


    else if (strcmp(name, "p_a") == 0) {
        return 19;
    }


    else if (strcmp(name, "p_b") == 0) {
        return 20;
    }


    else if (strcmp(name, "p_k") == 0) {
        return 21;
    }


    else if (strcmp(name, "phi") == 0) {
        return 22;
    }


    else if (strcmp(name, "rs") == 0) {
        return 23;
    }


    else if (strcmp(name, "rw") == 0) {
        return 24;
    }


    else if (strcmp(name, "scale_HF_cat50_ref") == 0) {
        return 25;
    }

    return -1;
}
// State index
int state_index(const char name[])
{

    if (strcmp(name, "Zetaw") == 0) {
        return 0;
    }


    else if (strcmp(name, "XS") == 0) {
        return 1;
    }


    else if (strcmp(name, "XW") == 0) {
        return 2;
    }


    else if (strcmp(name, "TmB") == 0) {
        return 3;
    }


    else if (strcmp(name, "Zetas") == 0) {
        return 4;
    }


    else if (strcmp(name, "CaTrpn") == 0) {
        return 5;
    }


    else if (strcmp(name, "Cd") == 0) {
        return 6;
    }

    return -1;
}
// Monitor index
int monitor_index(const char name[])
{

    if (strcmp(name, "Aw") == 0) {
        return 0;
    }


    else if (strcmp(name, "CaTrpn_max") == 0) {
        return 1;
    }


    else if (strcmp(name, "XS_max") == 0) {
        return 2;
    }


    else if (strcmp(name, "XW_max") == 0) {
        return 3;
    }


    else if (strcmp(name, "XU") == 0) {
        return 4;
    }


    else if (strcmp(name, "ksu") == 0) {
        return 5;
    }


    else if (strcmp(name, "cs") == 0) {
        return 6;
    }


    else if (strcmp(name, "cw") == 0) {
        return 7;
    }


    else if (strcmp(name, "kwu") == 0) {
        return 8;
    }


    else if (strcmp(name, "gammasu") == 0) {
        return 9;
    }


    else if (strcmp(name, "gammawu") == 0) {
        return 10;
    }


    else if (strcmp(name, "kb") == 0) {
        return 11;
    }


    else if (strcmp(name, "lambda_min12") == 0) {
        return 12;
    }


    else if (strcmp(name, "As") == 0) {
        return 13;
    }


    else if (strcmp(name, "dZetaw_dt") == 0) {
        return 14;
    }


    else if (strcmp(name, "dXS_dt") == 0) {
        return 15;
    }


    else if (strcmp(name, "dXW_dt") == 0) {
        return 16;
    }


    else if (strcmp(name, "dTmB_dt") == 0) {
        return 17;
    }


    else if (strcmp(name, "C") == 0) {
        return 18;
    }


    else if (strcmp(name, "cat50") == 0) {
        return 19;
    }


    else if (strcmp(name, "lambda_min087") == 0) {
        return 20;
    }


    else if (strcmp(name, "dZetas_dt") == 0) {
        return 21;
    }


    else if (strcmp(name, "F1") == 0) {
        return 22;
    }


    else if (strcmp(name, "dCd") == 0) {
        return 23;
    }


    else if (strcmp(name, "dCaTrpn_dt") == 0) {
        return 24;
    }


    else if (strcmp(name, "h_lambda_prima") == 0) {
        return 25;
    }


    else if (strcmp(name, "eta") == 0) {
        return 26;
    }


    else if (strcmp(name, "h_lambda") == 0) {
        return 27;
    }


    else if (strcmp(name, "Fd") == 0) {
        return 28;
    }


    else if (strcmp(name, "dCd_dt") == 0) {
        return 29;
    }


    else if (strcmp(name, "Ta") == 0) {
        return 30;
    }


    else if (strcmp(name, "Tp") == 0) {
        return 31;
    }


    else if (strcmp(name, "Ttot") == 0) {
        return 32;
    }

    return -1;
}


void init_parameter_values(double* parameters){
    /*
    Beta0=2.3, Beta1=-2.4, Tot_A=25.0, Tref=120.0, Trpn50=0.35, cai=0.0001, cat50_ref=0.805, dLambda=0.0, etal=200.0, etas=20.0, gammas=0.0085, gammaw=0.615, ktrpn=0.1, ku=0.04, kuw=0.182, kws=0.012, lmbda=1.0, ntm=2.4, ntrpn=2.0, p_a=2.1, p_b=9.1, p_k=7.0, phi=2.23, rs=0.25, rw=0.5, scale_HF_cat50_ref=1.0
    */
    parameters[0] = 2.3;
    parameters[1] = -2.4;
    parameters[2] = 25.0;
    parameters[3] = 120.0;
    parameters[4] = 0.35;
    parameters[5] = 0.0001;
    parameters[6] = 0.805;
    parameters[7] = 0.0;
    parameters[8] = 200.0;
    parameters[9] = 20.0;
    parameters[10] = 0.0085;
    parameters[11] = 0.615;
    parameters[12] = 0.1;
    parameters[13] = 0.04;
    parameters[14] = 0.182;
    parameters[15] = 0.012;
    parameters[16] = 1.0;
    parameters[17] = 2.4;
    parameters[18] = 2.0;
    parameters[19] = 2.1;
    parameters[20] = 9.1;
    parameters[21] = 7.0;
    parameters[22] = 2.23;
    parameters[23] = 0.25;
    parameters[24] = 0.5;
    parameters[25] = 1.0;
}


void init_state_values(double* states){
    /*
    Zetaw=0.0, XS=0.0, XW=0.0, TmB=1.0, Zetas=0.0, CaTrpn=0.0152, Cd=0.0
    */
    states[0] = 0.0;
    states[1] = 0.0;
    states[2] = 0.0;
    states[3] = 1.0;
    states[4] = 0.0;
    states[5] = 0.0152;
    states[6] = 0.0;
}


void rhs(const double t, const double *__restrict states, const double *__restrict parameters, double* values){

    // Assign states
    const double Zetaw = states[0];
    const double XS = states[1];
    const double XW = states[2];
    const double TmB = states[3];
    const double Zetas = states[4];
    const double CaTrpn = states[5];
    const double Cd = states[6];

    // Assign parameters
    const double Beta0 = parameters[0];
    const double Beta1 = parameters[1];
    const double Tot_A = parameters[2];
    const double Tref = parameters[3];
    const double Trpn50 = parameters[4];
    const double cai = parameters[5];
    const double cat50_ref = parameters[6];
    const double dLambda = parameters[7];
    const double etal = parameters[8];
    const double etas = parameters[9];
    const double gammas = parameters[10];
    const double gammaw = parameters[11];
    const double ktrpn = parameters[12];
    const double ku = parameters[13];
    const double kuw = parameters[14];
    const double kws = parameters[15];
    const double lmbda = parameters[16];
    const double ntm = parameters[17];
    const double ntrpn = parameters[18];
    const double p_a = parameters[19];
    const double p_b = parameters[20];
    const double p_k = parameters[21];
    const double phi = parameters[22];
    const double rs = parameters[23];
    const double rw = parameters[24];
    const double scale_HF_cat50_ref = parameters[25];

    // Assign expressions
    const double Aw = (Tot_A*rs)/(rs + rw*(1 - rs));
    const double CaTrpn_max = (CaTrpn > 0) ? CaTrpn : 0;
    const double XS_max = (XS > 0) ? XS : 0;
    const double XW_max = (XW > 0) ? XW : 0;
    const double XU = -XW + (-XS + (1 - TmB));
    const double ksu = (kws*rw)*(-1 + 1/rs);
    const double cs = ((kws*phi)*(rw*(1 - rs)))/rs;
    const double cw = ((kuw*phi)*((1 - rs)*(1 - rw)))/((rw*(1 - rs)));
    const double kwu = kuw*(-1 + 1/rw) - kws;
    const double gammasu = gammas*((((Zetas > 0 && Zetas < -1) ? (
       Zetas > -Zetas - 1
    )
    : (
       ((Zetas > 0) ? (
          Zetas > 0
       )
       : (
          ((Zetas < -1) ? (
             Zetas > -1
          )
          : (
             0
          ))
       ))
    ))) ? (
       Zetas*((Zetas > 0) ? (
          1
       )
       : (
          0
       ))
    )
    : (
       (-Zetas - 1)*((Zetas < -1) ? (
          1
       )
       : (
          0
       ))
    ));
    const double gammawu = gammaw*fabs(Zetaw);
    const double kb = (pow(Trpn50, ntm)*ku)/(-rw*(1 - rs) + (1 - rs));
    const double lambda_min12 = (lmbda < 1.2) ? lmbda : 1.2;
    const double As = Aw;
    const double dZetaw_dt = Aw*dLambda - Zetaw*cw;
    values[0] = dZetaw_dt;
    const double dXS_dt = -XS*gammasu + (-XS*ksu + XW*kws);
    values[1] = dXS_dt;
    const double dXW_dt = -XW*gammawu + (-XW*kws + (XU*kuw - XW*kwu));
    values[2] = dXW_dt;
    const double dTmB_dt = -TmB*pow(CaTrpn, ntm/2)*ku + XU*(kb*((pow(CaTrpn, (-ntm)/2) < 100) ? (
       pow(CaTrpn, (-ntm)/2)
    )
    : (
       100
    )));
    values[3] = dTmB_dt;
    const double C = lambda_min12 - 1;
    const double cat50 = scale_HF_cat50_ref*(Beta1*(lambda_min12 - 1) + cat50_ref);
    const double lambda_min087 = (lambda_min12 < 0.87) ? lambda_min12 : 0.87;
    const double dZetas_dt = As*dLambda - Zetas*cs;
    values[4] = dZetas_dt;
    const double F1 = exp(C*p_b) - 1;
    const double dCd = C - Cd;
    const double dCaTrpn_dt = ktrpn*(-CaTrpn + pow((1000*cai)/cat50, ntrpn)*(1 - CaTrpn));
    values[5] = dCaTrpn_dt;
    const double h_lambda_prima = Beta0*((lambda_min087 + lambda_min12) - 1.87) + 1;
    const double eta = (dCd < 0) ? etas : etal;
    const double h_lambda = (h_lambda_prima > 0) ? h_lambda_prima : 0;
    const double Fd = dCd*eta;
    const double dCd_dt = (p_k*(C - Cd))/eta;
    values[6] = dCd_dt;
    const double Ta = (h_lambda*(Tref/rs))*(XS*(Zetas + 1) + XW*Zetaw);
    const double Tp = p_a*(F1 + Fd);
    const double Ttot = Ta + Tp;
}


void monitor_values(const double t, const double *__restrict states, const double *__restrict parameters, double* values){

    // Assign states
    const double Zetaw = states[0];
    const double XS = states[1];
    const double XW = states[2];
    const double TmB = states[3];
    const double Zetas = states[4];
    const double CaTrpn = states[5];
    const double Cd = states[6];

    // Assign parameters
    const double Beta0 = parameters[0];
    const double Beta1 = parameters[1];
    const double Tot_A = parameters[2];
    const double Tref = parameters[3];
    const double Trpn50 = parameters[4];
    const double cai = parameters[5];
    const double cat50_ref = parameters[6];
    const double dLambda = parameters[7];
    const double etal = parameters[8];
    const double etas = parameters[9];
    const double gammas = parameters[10];
    const double gammaw = parameters[11];
    const double ktrpn = parameters[12];
    const double ku = parameters[13];
    const double kuw = parameters[14];
    const double kws = parameters[15];
    const double lmbda = parameters[16];
    const double ntm = parameters[17];
    const double ntrpn = parameters[18];
    const double p_a = parameters[19];
    const double p_b = parameters[20];
    const double p_k = parameters[21];
    const double phi = parameters[22];
    const double rs = parameters[23];
    const double rw = parameters[24];
    const double scale_HF_cat50_ref = parameters[25];

    // Assign expressions
    const double Aw = (Tot_A*rs)/(rs + rw*(1 - rs));
    values[0] = Aw;
    const double CaTrpn_max = (CaTrpn > 0) ? CaTrpn : 0;
    values[1] = CaTrpn_max;
    const double XS_max = (XS > 0) ? XS : 0;
    values[2] = XS_max;
    const double XW_max = (XW > 0) ? XW : 0;
    values[3] = XW_max;
    const double XU = -XW + (-XS + (1 - TmB));
    values[4] = XU;
    const double ksu = (kws*rw)*(-1 + 1/rs);
    values[5] = ksu;
    const double cs = ((kws*phi)*(rw*(1 - rs)))/rs;
    values[6] = cs;
    const double cw = ((kuw*phi)*((1 - rs)*(1 - rw)))/((rw*(1 - rs)));
    values[7] = cw;
    const double kwu = kuw*(-1 + 1/rw) - kws;
    values[8] = kwu;
    const double gammasu = gammas*((((Zetas > 0 && Zetas < -1) ? (
       Zetas > -Zetas - 1
    )
    : (
       ((Zetas > 0) ? (
          Zetas > 0
       )
       : (
          ((Zetas < -1) ? (
             Zetas > -1
          )
          : (
             0
          ))
       ))
    ))) ? (
       Zetas*((Zetas > 0) ? (
          1
       )
       : (
          0
       ))
    )
    : (
       (-Zetas - 1)*((Zetas < -1) ? (
          1
       )
       : (
          0
       ))
    ));
    values[9] = gammasu;
    const double gammawu = gammaw*fabs(Zetaw);
    values[10] = gammawu;
    const double kb = (pow(Trpn50, ntm)*ku)/(-rw*(1 - rs) + (1 - rs));
    values[11] = kb;
    const double lambda_min12 = (lmbda < 1.2) ? lmbda : 1.2;
    values[12] = lambda_min12;
    const double As = Aw;
    values[13] = As;
    const double dZetaw_dt = Aw*dLambda - Zetaw*cw;
    values[14] = dZetaw_dt;
    const double dXS_dt = -XS*gammasu + (-XS*ksu + XW*kws);
    values[15] = dXS_dt;
    const double dXW_dt = -XW*gammawu + (-XW*kws + (XU*kuw - XW*kwu));
    values[16] = dXW_dt;
    const double dTmB_dt = -TmB*pow(CaTrpn, ntm/2)*ku + XU*(kb*((pow(CaTrpn, (-ntm)/2) < 100) ? (
       pow(CaTrpn, (-ntm)/2)
    )
    : (
       100
    )));
    values[17] = dTmB_dt;
    const double C = lambda_min12 - 1;
    values[18] = C;
    const double cat50 = scale_HF_cat50_ref*(Beta1*(lambda_min12 - 1) + cat50_ref);
    values[19] = cat50;
    const double lambda_min087 = (lambda_min12 < 0.87) ? lambda_min12 : 0.87;
    values[20] = lambda_min087;
    const double dZetas_dt = As*dLambda - Zetas*cs;
    values[21] = dZetas_dt;
    const double F1 = exp(C*p_b) - 1;
    values[22] = F1;
    const double dCd = C - Cd;
    values[23] = dCd;
    const double dCaTrpn_dt = ktrpn*(-CaTrpn + pow((1000*cai)/cat50, ntrpn)*(1 - CaTrpn));
    values[24] = dCaTrpn_dt;
    const double h_lambda_prima = Beta0*((lambda_min087 + lambda_min12) - 1.87) + 1;
    values[25] = h_lambda_prima;
    const double eta = (dCd < 0) ? etas : etal;
    values[26] = eta;
    const double h_lambda = (h_lambda_prima > 0) ? h_lambda_prima : 0;
    values[27] = h_lambda;
    const double Fd = dCd*eta;
    values[28] = Fd;
    const double dCd_dt = (p_k*(C - Cd))/eta;
    values[29] = dCd_dt;
    const double Ta = (h_lambda*(Tref/rs))*(XS*(Zetas + 1) + XW*Zetaw);
    values[30] = Ta;
    const double Tp = p_a*(F1 + Fd);
    values[31] = Tp;
    const double Ttot = Ta + Tp;
    values[32] = Ttot;
}



void explicit_euler(const double *__restrict states, const double t, const double dt, const double *__restrict parameters, double* values){

    // Assign states
    const double Zetaw = states[0];
    const double XS = states[1];
    const double XW = states[2];
    const double TmB = states[3];
    const double Zetas = states[4];
    const double CaTrpn = states[5];
    const double Cd = states[6];

    // Assign parameters
    const double Beta0 = parameters[0];
    const double Beta1 = parameters[1];
    const double Tot_A = parameters[2];
    const double Tref = parameters[3];
    const double Trpn50 = parameters[4];
    const double cai = parameters[5];
    const double cat50_ref = parameters[6];
    const double dLambda = parameters[7];
    const double etal = parameters[8];
    const double etas = parameters[9];
    const double gammas = parameters[10];
    const double gammaw = parameters[11];
    const double ktrpn = parameters[12];
    const double ku = parameters[13];
    const double kuw = parameters[14];
    const double kws = parameters[15];
    const double lmbda = parameters[16];
    const double ntm = parameters[17];
    const double ntrpn = parameters[18];
    const double p_a = parameters[19];
    const double p_b = parameters[20];
    const double p_k = parameters[21];
    const double phi = parameters[22];
    const double rs = parameters[23];
    const double rw = parameters[24];
    const double scale_HF_cat50_ref = parameters[25];

    // Assign expressions
    const double Aw = (Tot_A*rs)/(rs + rw*(1 - rs));
    const double CaTrpn_max = (CaTrpn > 0) ? CaTrpn : 0;
    const double XS_max = (XS > 0) ? XS : 0;
    const double XW_max = (XW > 0) ? XW : 0;
    const double XU = -XW + (-XS + (1 - TmB));
    const double ksu = (kws*rw)*(-1 + 1/rs);
    const double cs = ((kws*phi)*(rw*(1 - rs)))/rs;
    const double cw = ((kuw*phi)*((1 - rs)*(1 - rw)))/((rw*(1 - rs)));
    const double kwu = kuw*(-1 + 1/rw) - kws;
    const double gammasu = gammas*((((Zetas > 0 && Zetas < -1) ? (
       Zetas > -Zetas - 1
    )
    : (
       ((Zetas > 0) ? (
          Zetas > 0
       )
       : (
          ((Zetas < -1) ? (
             Zetas > -1
          )
          : (
             0
          ))
       ))
    ))) ? (
       Zetas*((Zetas > 0) ? (
          1
       )
       : (
          0
       ))
    )
    : (
       (-Zetas - 1)*((Zetas < -1) ? (
          1
       )
       : (
          0
       ))
    ));
    const double gammawu = gammaw*fabs(Zetaw);
    const double kb = (pow(Trpn50, ntm)*ku)/(-rw*(1 - rs) + (1 - rs));
    const double lambda_min12 = (lmbda < 1.2) ? lmbda : 1.2;
    const double As = Aw;
    const double dZetaw_dt = Aw*dLambda - Zetaw*cw;
    values[0] = Zetaw + dZetaw_dt*dt;
    const double dXS_dt = -XS*gammasu + (-XS*ksu + XW*kws);
    values[1] = XS + dXS_dt*dt;
    const double dXW_dt = -XW*gammawu + (-XW*kws + (XU*kuw - XW*kwu));
    values[2] = XW + dXW_dt*dt;
    const double dTmB_dt = -TmB*pow(CaTrpn, ntm/2)*ku + XU*(kb*((pow(CaTrpn, (-ntm)/2) < 100) ? (
       pow(CaTrpn, (-ntm)/2)
    )
    : (
       100
    )));
    values[3] = TmB + dTmB_dt*dt;
    const double C = lambda_min12 - 1;
    const double cat50 = scale_HF_cat50_ref*(Beta1*(lambda_min12 - 1) + cat50_ref);
    const double lambda_min087 = (lambda_min12 < 0.87) ? lambda_min12 : 0.87;
    const double dZetas_dt = As*dLambda - Zetas*cs;
    values[4] = Zetas + dZetas_dt*dt;
    const double F1 = exp(C*p_b) - 1;
    const double dCd = C - Cd;
    const double dCaTrpn_dt = ktrpn*(-CaTrpn + pow((1000*cai)/cat50, ntrpn)*(1 - CaTrpn));
    values[5] = CaTrpn + dCaTrpn_dt*dt;
    const double h_lambda_prima = Beta0*((lambda_min087 + lambda_min12) - 1.87) + 1;
    const double eta = (dCd < 0) ? etas : etal;
    const double h_lambda = (h_lambda_prima > 0) ? h_lambda_prima : 0;
    const double Fd = dCd*eta;
    const double dCd_dt = (p_k*(C - Cd))/eta;
    values[6] = Cd + dCd_dt*dt;
    const double Ta = (h_lambda*(Tref/rs))*(XS*(Zetas + 1) + XW*Zetaw);
    const double Tp = p_a*(F1 + Fd);
    const double Ttot = Ta + Tp;
}


void generalized_rush_larsen(const double *__restrict states, const double t, const double dt, const double *__restrict parameters, double* values){

    // Assign states
    const double Zetaw = states[0];
    const double XS = states[1];
    const double XW = states[2];
    const double TmB = states[3];
    const double Zetas = states[4];
    const double CaTrpn = states[5];
    const double Cd = states[6];

    // Assign parameters
    const double Beta0 = parameters[0];
    const double Beta1 = parameters[1];
    const double Tot_A = parameters[2];
    const double Tref = parameters[3];
    const double Trpn50 = parameters[4];
    const double cai = parameters[5];
    const double cat50_ref = parameters[6];
    const double dLambda = parameters[7];
    const double etal = parameters[8];
    const double etas = parameters[9];
    const double gammas = parameters[10];
    const double gammaw = parameters[11];
    const double ktrpn = parameters[12];
    const double ku = parameters[13];
    const double kuw = parameters[14];
    const double kws = parameters[15];
    const double lmbda = parameters[16];
    const double ntm = parameters[17];
    const double ntrpn = parameters[18];
    const double p_a = parameters[19];
    const double p_b = parameters[20];
    const double p_k = parameters[21];
    const double phi = parameters[22];
    const double rs = parameters[23];
    const double rw = parameters[24];
    const double scale_HF_cat50_ref = parameters[25];

    // Assign expressions
    const double Aw = (Tot_A*rs)/(rs + rw*(1 - rs));
    const double CaTrpn_max = (CaTrpn > 0) ? CaTrpn : 0;
    const double XS_max = (XS > 0) ? XS : 0;
    const double XW_max = (XW > 0) ? XW : 0;
    const double XU = -XW + (-XS + (1 - TmB));
    const double ksu = (kws*rw)*(-1 + 1/rs);
    const double cs = ((kws*phi)*(rw*(1 - rs)))/rs;
    const double cw = ((kuw*phi)*((1 - rs)*(1 - rw)))/((rw*(1 - rs)));
    const double kwu = kuw*(-1 + 1/rw) - kws;
    const double gammasu = gammas*((((Zetas > 0 && Zetas < -1) ? (
       Zetas > -Zetas - 1
    )
    : (
       ((Zetas > 0) ? (
          Zetas > 0
       )
       : (
          ((Zetas < -1) ? (
             Zetas > -1
          )
          : (
             0
          ))
       ))
    ))) ? (
       Zetas*((Zetas > 0) ? (
          1
       )
       : (
          0
       ))
    )
    : (
       (-Zetas - 1)*((Zetas < -1) ? (
          1
       )
       : (
          0
       ))
    ));
    const double gammawu = gammaw*fabs(Zetaw);
    const double kb = (pow(Trpn50, ntm)*ku)/(-rw*(1 - rs) + (1 - rs));
    const double lambda_min12 = (lmbda < 1.2) ? lmbda : 1.2;
    const double As = Aw;
    const double dZetaw_dt = Aw*dLambda - Zetaw*cw;
    const double dZetaw_dt_linearized = -cw;
    values[0] = Zetaw + ((fabs(dZetaw_dt_linearized) > 1e-08) ? (
       dZetaw_dt*(exp(dZetaw_dt_linearized*dt) - 1)/dZetaw_dt_linearized
    )
    : (
       dZetaw_dt*dt
    ));
    const double dXS_dt = -XS*gammasu + (-XS*ksu + XW*kws);
    const double dXS_dt_linearized = -gammasu - ksu;
    values[1] = XS + ((fabs(dXS_dt_linearized) > 1e-08) ? (
       dXS_dt*(exp(dXS_dt_linearized*dt) - 1)/dXS_dt_linearized
    )
    : (
       dXS_dt*dt
    ));
    const double dXW_dt = -XW*gammawu + (-XW*kws + (XU*kuw - XW*kwu));
    const double dXW_dt_linearized = -gammawu - kws - kwu;
    values[2] = XW + ((fabs(dXW_dt_linearized) > 1e-08) ? (
       dXW_dt*(exp(dXW_dt_linearized*dt) - 1)/dXW_dt_linearized
    )
    : (
       dXW_dt*dt
    ));
    const double dTmB_dt = -TmB*pow(CaTrpn, ntm/2)*ku + XU*(kb*((pow(CaTrpn, (-ntm)/2) < 100) ? (
       pow(CaTrpn, (-ntm)/2)
    )
    : (
       100
    )));
    const double dTmB_dt_linearized = -pow(CaTrpn, ntm/2)*ku;
    values[3] = TmB + ((fabs(dTmB_dt_linearized) > 1e-08) ? (
       dTmB_dt*(exp(dTmB_dt_linearized*dt) - 1)/dTmB_dt_linearized
    )
    : (
       dTmB_dt*dt
    ));
    const double C = lambda_min12 - 1;
    const double cat50 = scale_HF_cat50_ref*(Beta1*(lambda_min12 - 1) + cat50_ref);
    const double lambda_min087 = (lambda_min12 < 0.87) ? lambda_min12 : 0.87;
    const double dZetas_dt = As*dLambda - Zetas*cs;
    const double dZetas_dt_linearized = -cs;
    values[4] = Zetas + ((fabs(dZetas_dt_linearized) > 1e-08) ? (
       dZetas_dt*(exp(dZetas_dt_linearized*dt) - 1)/dZetas_dt_linearized
    )
    : (
       dZetas_dt*dt
    ));
    const double F1 = exp(C*p_b) - 1;
    const double dCd = C - Cd;
    const double dCaTrpn_dt = ktrpn*(-CaTrpn + pow((1000*cai)/cat50, ntrpn)*(1 - CaTrpn));
    const double dCaTrpn_dt_linearized = ktrpn*(-pow((1000*cai)/cat50, ntrpn) - 1);
    values[5] = CaTrpn + ((fabs(dCaTrpn_dt_linearized) > 1e-08) ? (
       dCaTrpn_dt*(exp(dCaTrpn_dt_linearized*dt) - 1)/dCaTrpn_dt_linearized
    )
    : (
       dCaTrpn_dt*dt
    ));
    const double h_lambda_prima = Beta0*((lambda_min087 + lambda_min12) - 1.87) + 1;
    const double eta = (dCd < 0) ? etas : etal;
    const double h_lambda = (h_lambda_prima > 0) ? h_lambda_prima : 0;
    const double Fd = dCd*eta;
    const double dCd_dt = (p_k*(C - Cd))/eta;
    const double dCd_dt_linearized = -p_k/eta;
    values[6] = Cd + ((fabs(dCd_dt_linearized) > 1e-08) ? (
       dCd_dt*(exp(dCd_dt_linearized*dt) - 1)/dCd_dt_linearized
    )
    : (
       dCd_dt*dt
    ));
    const double Ta = (h_lambda*(Tref/rs))*(XS*(Zetas + 1) + XW*Zetaw);
    const double Tp = p_a*(F1 + Fd);
    const double Ttot = Ta + Tp;
}
