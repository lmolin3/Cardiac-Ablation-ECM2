#include "ThreeStateCellDeath.h"

// Gotran generated C/C++ code for the "ThreeStateCellDeath" model

// Init state values
void init_state_values(double* states)
{
  states[0] = 1.0; // N;
  states[1] = 0.0; // U;
  states[2] = 0.0; // D;
}

// Default parameter values
void init_parameters_values(double* parameters)
{
  parameters[0] = 0.0; // k1;
  parameters[1] = 0.0; // k2;
  parameters[2] = 0.0; // k3;
}

// State index
int state_index(const char name[])
{
  // State names
  char names[][2] = {"N", "U", "D"};

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
int parameter_index(const char name[])
{
  // Parameter names
  char names[][3] = {"k1", "k2", "k3"};

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

// Compute the right hand side of the ThreeStateCellDeath ODE
void rhs(const double *__restrict states, const double t, const double
  *__restrict parameters, double* values)
{

  // Assign states
  const double N = states[0];
  const double U = states[1];
  const double D = states[2];

  // Assign parameters
  const double k1 = parameters[0];
  const double k2 = parameters[1];
  const double k3 = parameters[2];

  // Expressions for the Main component
  values[0] = k2*U - k1*N;
  values[1] = k1*N - k2*U - k3*U;
  values[2] = k3*U;
}

// Compute a forward step using the explicit Euler scheme to the
// ThreeStateCellDeath ODE
void forward_explicit_euler(double *__restrict states, const double t, const
  double dt, const double *__restrict parameters)
{

  // Assign states
  const double N = states[0];
  const double U = states[1];
  const double D = states[2];

  // Assign parameters
  const double k1 = parameters[0];
  const double k2 = parameters[1];
  const double k3 = parameters[2];

  // Expressions for the Main component
  const double dN_dt = k2*U - k1*N;
  states[0] = dt*dN_dt + N;
  const double dU_dt = k1*N - k2*U - k3*U;
  states[1] = dt*dU_dt + U;
  const double dD_dt = k3*U;
  states[2] = dt*dD_dt + D;
}

// Compute a forward step using the Rush-Larsen scheme to the
// ThreeStateCellDeath ODE
void forward_rush_larsen(double *__restrict states, const double t, const
  double dt, const double *__restrict parameters)
{

  // Assign states
  const double N = states[0];
  const double U = states[1];
  const double D = states[2];

  // Assign parameters
  const double k1 = parameters[0];
  const double k2 = parameters[1];
  const double k3 = parameters[2];

  // Expressions for the Main component
  const double dN_dt = k2*U - k1*N;
  const double dN_dt_linearized = -k1;
  states[0] = N + (std::fabs(dN_dt_linearized) > 1.0e-8 ?
    (std::expm1(dt*dN_dt_linearized))*dN_dt/dN_dt_linearized : dt*dN_dt);
  const double dU_dt = k1*N - k2*U - k3*U;
  const double dU_dt_linearized = -k2;
  states[1] = (std::fabs(dU_dt_linearized) > 1.0e-8 ?
    (std::expm1(dt*dU_dt_linearized))*dU_dt/dU_dt_linearized : dt*dU_dt) + U;
  const double dD_dt = k3*U;
  states[2] = dt*dD_dt + D;
}

// Compute a forward step using the generalised Rush-Larsen (GRL1) scheme to
// the ThreeStateCellDeath ODE
void forward_generalized_rush_larsen(double *__restrict states, const double
  t, const double dt, const double *__restrict parameters)
{

  // Assign states
  const double N = states[0];
  const double U = states[1];
  const double D = states[2];

  // Assign parameters
  const double k1 = parameters[0];
  const double k2 = parameters[1];
  const double k3 = parameters[2];

  // Expressions for the Main component
  const double dN_dt = k2*U - k1*N;
  const double dN_dt_linearized = -k1;
  states[0] = N + (std::fabs(dN_dt_linearized) > 1.0e-8 ?
    (std::expm1(dt*dN_dt_linearized))*dN_dt/dN_dt_linearized : dt*dN_dt);
  const double dU_dt = k1*N - k2*U - k3*U;
  const double dU_dt_linearized = -k2;
  states[1] = (std::fabs(dU_dt_linearized) > 1.0e-8 ?
    (std::expm1(dt*dU_dt_linearized))*dU_dt/dU_dt_linearized : dt*dU_dt) + U;
  const double dD_dt = k3*U;
  states[2] = dt*dD_dt + D;
}

// Compute a forward step using the FE / GRL1 scheme to the
// ThreeStateCellDeath ODE
void forward_hybrid_generalized_rush_larsen(double *__restrict states, const
  double t, const double dt, const double *__restrict parameters)
{

  // Assign states
  const double N = states[0];
  const double U = states[1];
  const double D = states[2];

  // Assign parameters
  const double k1 = parameters[0];
  const double k2 = parameters[1];
  const double k3 = parameters[2];

  // Expressions for the Main component
  const double dN_dt = k2*U - k1*N;
  states[0] = dt*dN_dt + N;
  const double dU_dt = k1*N - k2*U - k3*U;
  states[1] = dt*dU_dt + U;
  const double dD_dt = k3*U;
  states[2] = dt*dD_dt + D;
}

// Compute a forward step using the simplified implicit Eulerscheme to the
// ThreeStateCellDeath ODE
void forward_simplified_implicit_euler(double *__restrict states, const
  double t, const double dt, const double *__restrict parameters)
{

  // Assign states
  const double N = states[0];
  const double U = states[1];
  const double D = states[2];

  // Assign parameters
  const double k1 = parameters[0];
  const double k2 = parameters[1];
  const double k3 = parameters[2];

  // Expressions for the Main component
  const double dN_dt = k2*U - k1*N;
  const double dN_dt_diag_jac = -k1;
  const double dU_dt = k1*N - k2*U - k3*U;
  const double dU_dt_diag_jac = -k2;
  const double dD_dt = k3*U;
  states[0] = dt*dN_dt/(1. - dt*dN_dt_diag_jac) + N;
  states[1] = dt*dU_dt/(1. - dt*dU_dt_diag_jac) + U;
  states[2] = dt*dD_dt + D;
}
