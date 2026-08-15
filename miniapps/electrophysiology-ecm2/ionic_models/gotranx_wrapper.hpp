#pragma once

#include "mfem.hpp"
#include <cstring>
#include <cmath>

namespace mfem
{
    namespace electrophysiology
    {

        /**
         * @brief Fallback time integration schemes for a model's device Kernel.
         *
         * gotranx only emits the schemes requested when the model was generated, so
         * not every model provides all of them (e.g. Mitchell-Schaeffer has no
         * forward_* variants). Each model's `Kernel` derives from this struct and
         * defines only what was generated; C++ name hiding then selects the real
         * implementation, and any scheme the model lacks resolves to the stub here
         * and aborts with a clear message.
         *
         * This is what lets ReactionSolver's scheme dispatch name all schemes
         * unconditionally, with no per-model capability flag to keep in sync and no
         * `if constexpr` guards.
         */
        struct IonicKernelDefaults
        {
#define MFEM_EP_UNAVAILABLE_SCHEME(NAME)                                          \
    MFEM_HOST_DEVICE static void NAME(const real_t *__restrict, const real_t,     \
                                      const real_t, const real_t *__restrict,     \
                                      real_t *)                                   \
    {                                                                             \
        MFEM_ABORT_KERNEL("ionic model was not generated with the '" #NAME "' "    \
                          "integration scheme\n");                                \
    }

            MFEM_EP_UNAVAILABLE_SCHEME(explicit_euler)
            MFEM_EP_UNAVAILABLE_SCHEME(generalized_rush_larsen)
            MFEM_EP_UNAVAILABLE_SCHEME(forward_explicit_euler)
            MFEM_EP_UNAVAILABLE_SCHEME(forward_generalized_rush_larsen)
            MFEM_EP_UNAVAILABLE_SCHEME(hybrid_rush_larsen)

#undef MFEM_EP_UNAVAILABLE_SCHEME
        };


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

            // Model metadata accessors - these will access the member variables
            int GetNumStates() const { return NUM_STATES; }
            int GetNumParameters() const { return NUM_PARAMS; }
            int GetNumMonitored() const { return NUM_MONITORED; }

            // NOTE: the model math (rhs, monitor_values and the time integration
            // schemes) is deliberately NOT part of this interface. It lives in each
            // model's nested `Kernel` struct as MFEM_HOST_DEVICE static functions,
            // which are callable from both host and device code:
            //
            //     MitchellSchaeffer::Kernel::generalized_rush_larsen(s, t, dt, p, v);
            //
            // A virtual method cannot be called from a device kernel (the vtable is
            // host-side), and it would also block inlining, so ReactionSolver
            // dispatches on the model type once and instantiates a templated
            // mfem::forall over the Kernel instead. This base class is therefore a
            // host-side metadata/introspection interface only: name lookup, default
            // states and parameters, and the index/flag metadata below.

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