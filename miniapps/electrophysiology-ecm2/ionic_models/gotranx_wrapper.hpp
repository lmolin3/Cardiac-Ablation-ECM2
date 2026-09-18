#pragma once

#include "mfem.hpp"
#include "general/forall.hpp"
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mfem
{
    namespace electrophysiology
    {

        /**
         * @brief Time integration schemes emitted by gotranx.
         *
         * Declared here rather than in reaction_solver.hpp because it selects
         * which generated Kernel function to call, and both the EP reaction
         * kernels and the contraction kernels below dispatch on it.
         */
        enum class TimeIntegrationScheme : int
        {
            EXPLICIT_EULER = 0,
            FORWARD_EXPLICIT_EULER = 1,
            GENERALIZED_RUSH_LARSEN = 2,
            FORWARD_GENERALIZED_RUSH_LARSEN = 3,
            HYBRID_RUSH_LARSEN = 4
        };

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
            virtual int GetNumStates() const { return NUM_STATES; }
            virtual int GetNumParameters() const { return NUM_PARAMS; }
            virtual int GetNumMonitored() const { return NUM_MONITORED; }

            /// Human-readable model name, used in error messages and logs.
            virtual std::string GetName() const = 0;

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

            // Potential range [mV] this model lives in. ReactionSolver adopts
            // these in Setup() unless the caller overrode them with SetVRange().
            //
            // The two meanings are different and both matter:
            //   * dimensionless models   -- this is the affine map applied to the
            //     [0,1] state, so it defines the physical potential outright;
            //   * physiological models   -- the ODE already produces millivolts,
            //     so this is only the sanity clamp guarding against blow-up, and
            //     it must be wide enough to leave the action potential untouched.
            //
            // The defaults below are the dimensionless ones (Mitchell-Schaeffer,
            // Fenton-Karma). A physiological model MUST widen them in its
            // constructor -- leaving them would truncate its upstroke and its
            // resting potential against the rails.
            real_t Vmin_default = -80.0;
            real_t Vmax_default = -20.0;
        };



        /**
         * @brief Base class for electrophysiology (cell membrane) models.
         *
         * Adds the two things a coupled electromechanics solver needs to know
         * about an EP model: where the transmembrane potential lives in the
         * state vector, and whether the model resolves intracellular calcium at
         * all. Phenomenological models (Mitchell-Schaeffer, Fenton-Karma) do
         * not, which is what makes the fail-fast check in
         * ReactionSolver::RegisterModels() meaningful.
         */
        class EPModelBase : public GotranxODEModel
        {
        public:
            EPModelBase() = default;

            /// Index of the transmembrane potential within the state vector.
            virtual int GetPotentialIndex() const = 0;

            /// True if the model resolves intracellular calcium [Ca2+]i.
            virtual bool HasCalcium() const { return false; }

            /**
             * @brief Index of [Ca2+]i within the state vector.
             *
             * Only meaningful when HasCalcium() is true; the default exists so
             * that a mis-paired model fails loudly instead of silently reading
             * an unrelated state.
             */
            virtual int GetCalciumIndex() const
            {
                throw std::runtime_error(
                    "EPModelBase::GetCalciumIndex(): model '" + GetName() +
                    "' does not resolve intracellular calcium.");
            }
        };


        /**
         * @brief Arguments for one contraction-model ODE step over all dofs.
         *
         * Grouped into a struct so that AdvanceODE() stays a single virtual call
         * made once per time step from the host, while the work itself is a
         * templated mfem::forall over the model's device Kernel. All pointers
         * are device pointers into structure-of-arrays storage: entry k of dof i
         * lives at [k * ndofs + i], matching ReactionSolver's EP state layout.
         */
        struct ContractionStepArgs
        {
            int ndofs = 0;               ///< Number of true dofs
            real_t dt = 0.0;             ///< Time step [ms]
            int scheme = 0;              ///< TimeIntegrationScheme value

            const real_t *calcium = nullptr; ///< [ndofs] [Ca2+]i read from the EP states
            real_t calcium_scale = 1.0;      ///< Converts EP calcium units into the model's

            real_t *states = nullptr;        ///< [nstates * ndofs] contraction states, updated in place
            const real_t *params = nullptr;  ///< [nparams] uniform, or [nparams * ndofs]
            bool per_dof_params = false;     ///< Selects the params layout above

            real_t *active_tension = nullptr; ///< [ndofs] output, active tension Ta [kPa]
        };


        /**
         * @brief Base class for active-contraction (excitation-contraction) models.
         *
         * A contraction model is driven by the calcium transient produced by an
         * EP model and returns the active tension developed along the fiber.
         */
        class ContractionModelBase : public GotranxODEModel
        {
        public:
            ContractionModelBase() = default;

            /// True if the model needs [Ca2+]i from the paired EP model.
            virtual bool RequiresCalcium() const { return true; }

            /**
             * @brief Index of the active tension within the *monitored* values.
             *
             * In the Land 2017 encoding the tension Ta is an intermediate
             * expression rather than a state, so it is addressed in the monitor
             * array produced by Kernel::monitor_values().
             */
            virtual int GetActiveTensionIndex() const = 0;

            /**
             * @brief Advance the contraction ODEs at every dof and extract Ta.
             *
             * Host-side entry point: called once per step, it launches the
             * device kernel over all dofs. It is virtual (and therefore host
             * only) precisely because the per-dof math must not be -- the
             * implementation dispatches to the model's static Kernel functions.
             */
            virtual void AdvanceODE(const ContractionStepArgs &args) const = 0;
        };


        /**
         * @brief Select the integration scheme for a generated Kernel.
         *
         * The branch is uniform across all threads, so it costs nothing in terms
         * of divergence; keeping it at runtime avoids instantiating the kernel
         * once per (model, scheme) pair.
         */
        template <typename K>
        MFEM_HOST_DEVICE inline void ApplyOdeScheme(int scheme,
                                                    const real_t *__restrict s, real_t t, real_t dt,
                                                    const real_t *__restrict p, real_t *v)
        {
            if (scheme == (int)TimeIntegrationScheme::EXPLICIT_EULER)
            {
                K::explicit_euler(s, t, dt, p, v);
            }
            else if (scheme == (int)TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN)
            {
                K::generalized_rush_larsen(s, t, dt, p, v);
            }
            else if (scheme == (int)TimeIntegrationScheme::FORWARD_EXPLICIT_EULER)
            {
                K::forward_explicit_euler(s, t, dt, p, v);
            }
            else if (scheme == (int)TimeIntegrationScheme::FORWARD_GENERALIZED_RUSH_LARSEN)
            {
                K::forward_generalized_rush_larsen(s, t, dt, p, v);
            }
            else if (scheme == (int)TimeIntegrationScheme::HYBRID_RUSH_LARSEN)
            {
                K::hybrid_rush_larsen(s, t, dt, p, v);
            }
        }


        /**
         * @brief One contraction ODE step at every dof, followed by tension extraction.
         *
         * PerDofParams=false reads NP uniform values (a broadcast load from
         * cache); true reads the full NP x n array. Templated rather than a
         * runtime flag so neither carries a branch, mirroring the EP reaction
         * kernel.
         */
        template <typename K, bool PerDofParams>
        void ContractionAdvanceKernel(const ContractionStepArgs &a)
        {
            constexpr int NS = K::nstates;
            constexpr int NP = K::nparams;
            constexpr int NM = K::nmonitored;
            constexpr int CA = K::calcium_param_idx;
            constexpr int TA = K::active_tension_monitor_idx;

            const int n = a.ndofs;
            const int scheme = a.scheme;
            const real_t dt = a.dt;
            const real_t ca_scale = a.calcium_scale;
            const real_t *d_ca = a.calcium;
            real_t *d_states = a.states;
            const real_t *d_params = a.params;
            real_t *d_Ta = a.active_tension;

            mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
            {
                real_t s[NS], v[NS], p[NP], mon[NM];

                for (int k = 0; k < NS; k++) { s[k] = d_states[k * n + i]; v[k] = s[k]; }
                if (PerDofParams)
                {
                    for (int k = 0; k < NP; k++) { p[k] = d_params[k * n + i]; }
                }
                else
                {
                    for (int k = 0; k < NP; k++) { p[k] = d_params[k]; }
                }

                // The one-way coupling: [Ca2+]i from the EP model is an input
                // parameter of the contraction model, overwritten every step.
                p[CA] = d_ca[i] * ca_scale;

                // The contraction models carry no explicit time dependence (the
                // calcium input is what drives them), so t is irrelevant here.
                ApplyOdeScheme<K>(scheme, s, 0.0, dt, p, v);

                for (int k = 0; k < NS; k++) { d_states[k * n + i] = v[k]; }

                // Ta is an algebraic function of the state, so it is evaluated
                // from the *updated* state v, not the incoming s.
                K::monitor_values(0.0, v, p, mon);
                d_Ta[i] = mon[TA];
            });
        }

        /// Dispatch on the parameter layout, keeping the branch in one place.
        template <typename K>
        inline void ContractionAdvance(const ContractionStepArgs &a)
        {
            if (a.per_dof_params) { ContractionAdvanceKernel<K, true>(a); }
            else { ContractionAdvanceKernel<K, false>(a); }
        }


        /**
         * @brief  Gotranx-generated ODE models with temperature and damage dependency.
         * */
        class GotranxODEModelWithThermalDamage : public EPModelBase
        {
        public:
            // Allow ReactionSolver to access the indices
            friend class ReactionSolver;
        
            GotranxODEModelWithThermalDamage() : EPModelBase()
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
