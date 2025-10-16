#pragma once

#include "mfem.hpp"
#include <cstring>
#include <cmath>

namespace mfem
{
    namespace electrophysiology
    {

        class GotranxODEModel
        {
        public:
            // Allow ReactionSolver to access the indices
            friend class ReactionSolver;
    
            GotranxODEModel() = default;

            virtual ~GotranxODEModel() = default;

            // Set indices of variables for stimulation - accepts variable number of parameters
            /*template <typename... Args>
            void SetStimulationParameters(Args... args)
            {
                MFEM_ABORT("GotranxODEModel::SetStimulationParameters()\n"
                           "   is not implemented for this class.");
            }*/

            // Index lookup methods - must be implemented by derived classes
            virtual int parameter_index(const char name[]) = 0;
            virtual int state_index(const char name[]) = 0;
            virtual int monitor_index(const char name[]) = 0;

            // Initialization methods - must be implemented by derived classes
            virtual void init_parameter_values(double *parameters) = 0;
            virtual void init_state_values(double *states) = 0;

            // Set stimulation parameters
            /*template <typename... Args>
            void SetStimulationParameters(Args... args)
            {
                MFEM_ABORT("GotranxODEModel::SetStimulationParameters()\n"
                           "   is not implemented for this class.");
            }*/

            // Core ODE methods - must be implemented by derived classes
            virtual void rhs(const double t, const double *__restrict states,
                             const double *__restrict parameters, double *values) = 0;

            virtual void monitor_values(const double t, const double *__restrict states,
                                        const double *__restrict parameters, double *values) = 0;

            // Model metadata accessors - these will access the member variables
            int GetNumStates() const { return NUM_STATES; }
            int GetNumParameters() const { return NUM_PARAMS; }
            int GetNumMonitored() const { return NUM_MONITORED; }

            virtual void explicit_euler(const double *__restrict states, const double t, const double dt,
                                        const double *__restrict parameters, double *values)
            {
                MFEM_ABORT("GotranxODEModel::explicit_euler()\n"
                           "   is not implemented for this class.");
            }

            virtual void generalized_rush_larsen(const double *__restrict states, const double t, const double dt,
                                                 const double *__restrict parameters, double *values)
            {
                MFEM_ABORT("GotranxODEModel::generalized_rush_larsen()\n"
                           "   is not implemented for this class.");
            }

            virtual void forward_explicit_euler(const double *__restrict states, const double t, const double dt,
                                                const double *__restrict parameters, double *values)
            {
                MFEM_ABORT("GotranxODEModel::forward_explicit_euler()\n"
                           "   is not implemented for this class.");
            }

            virtual void forward_generalized_rush_larsen(const double *__restrict states, const double t, const double dt,
                                                         const double *__restrict parameters, double *values)
            {
                MFEM_ABORT("GotranxODEModel::forward_generalized_rush_larsen()\n"
                           "   is not implemented for this class.");
            }

            virtual void hybrid_rush_larsen(const double *__restrict states, const double t, const double dt,
                                            const double *__restrict parameters, double *values)
            {
                MFEM_ABORT("GotranxODEModel::hybrid_rush_larsen()\n"
                           "   is not implemented for this class.");
            }

        protected:
            int NUM_STATES = -1;
            int NUM_PARAMS = -1;
            int NUM_MONITORED = -1;

            int potential_idx = -1; // Index of the transmembrane potential in the states array
            int stim_idx = -1;      // Index of the stimulation current in the parameters array
        };

    } // namespace electrophysiology
} // namespace mfem