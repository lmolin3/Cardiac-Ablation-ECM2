#pragma once

#include "mfem.hpp"
#include <cstring>
#include <cmath>

namespace mfem
{
    namespace electrophysiology
    {


        /**  
         * @brief  Base class for Gotranx-generated ODE models.
         * */
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

            // Set the internal stimulation parameters for continuous stimulation
            // This way timing can be handled via the coefficients passed to the ReactionSolver::SetStimulation method
            void DisableInternalTimeManagement(double *parameters)
            {
                // Set stimulation parameters to continuously enable internal stimulation
                parameters[stim_ampl_idx] = 0.0;
                parameters[stim_start_idx] = 0.0;
                parameters[stim_end_idx] = 1e6;
                parameters[stim_period_idx] = 1e6;
                parameters[stim_duration_idx] = 1e6;
            }

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

            int potential_idx = -1;     // Index of the transmembrane potential in the states array
            int stim_ampl_idx = -1;          // Index of the stimulation current in the parameters array
            int stim_duration_idx = -1; // Index of the stimulation duration in the parameters array
            int stim_start_idx = -1;    // Index of the stimulation start time in the parameters array
            int stim_end_idx = -1;      // Index of the stimulation end time in the parameters array
            int stim_period_idx = -1;   // Index of the stimulation period in the parameters array

            real_t stim_sign = 1.0; // Sign of the stimulation current
            
            bool dimensionless = false; // Flag indicating if the model uses dimensionless potential
        };



        /**  
         * @brief  Gotranx-generated ODE models with temperature and damage dependency.
         * */
        class GotranxODEModelWithThermalDamage : public GotranxODEModel
        {
        public:
            // Allow ReactionSolver to access the indices
            friend class ReactionSolver;
        
            GotranxODEModelWithThermalDamage() : GotranxODEModel()
            {}

            // Call this after construction
            void InitializeParameterIndices()
            {
                eta_idx = parameter_index("eta");
                gamma_idx = parameter_index("gamma");
                Q_idx = parameter_index("Q");
            }

            std::vector<int> GetTimeConstantsIdxs()
            {
                return time_constants_idxs;
            }
        
        protected:
            // Additional indices for temperature and damage dependency
            int eta_idx = -1;   // Index of the eta parameter in the parameters array
            int gamma_idx = -1; // Index of the gamma parameter in the parameters array
            int Q_idx = -1;     // Index of the Q parameter in the parameters array

            std::vector<int> time_constants_idxs; // Indices of time constants in the parameters array
        }; 
        


        } // namespace electrophysiology
} // namespace mfem