#ifndef THREE_STATE_CELL_DEATH_H
#define THREE_STATE_CELL_DEATH_H

#include <cmath>
#include <cstring>
#include <stdexcept>

// Gotran generated C/C++ code for the "ThreeStateCellDeath" model

// Init state values
void init_state_values(double* states);

// Default parameter values
void init_parameters_values(double* parameters);

// State index
int state_index(const char name[]);

// Parameter index
int parameter_index(const char name[]);

// Compute the right hand side of the ThreeStateCellDeath ODE
void rhs(const double *__restrict states, const double t, const double
  *__restrict parameters, double* values);

// Compute a forward step using the explicit Euler scheme to the
// ThreeStateCellDeath ODE
void forward_explicit_euler(double *__restrict states, const double t, const
  double dt, const double *__restrict parameters);

// Compute a forward step using the Rush-Larsen scheme to the
// ThreeStateCellDeath ODE
void forward_rush_larsen(double *__restrict states, const double t, const
  double dt, const double *__restrict parameters);

// Compute a forward step using the generalised Rush-Larsen (GRL1) scheme to
// the ThreeStateCellDeath ODE
void forward_generalized_rush_larsen(double *__restrict states, const double
  t, const double dt, const double *__restrict parameters);

// Compute a forward step using the FE / GRL1 scheme to the
// ThreeStateCellDeath ODE
void forward_hybrid_generalized_rush_larsen(double *__restrict states, const
  double t, const double dt, const double *__restrict parameters);

// Compute a forward step using the simplified implicit Eulerscheme to the
// ThreeStateCellDeath ODE
void forward_simplified_implicit_euler(double *__restrict states, const
  double t, const double dt, const double *__restrict parameters);

#endif // THREE_STATE_CELL_DEATH_H    