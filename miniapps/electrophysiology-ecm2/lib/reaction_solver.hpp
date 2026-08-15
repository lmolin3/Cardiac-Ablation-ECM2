#pragma once

#include "mfem.hpp"

// Include all available ionic models here
#include "../ionic_models/mitchell_schaeffer_2003.h"
#include "../ionic_models/fenton_karma_1998.h"
#include "../ionic_models/mitchell_schaeffer_2003_td_dependent.h"

namespace mfem
{
    namespace electrophysiology
    {

        enum class TimeIntegrationScheme : int
        {
            EXPLICIT_EULER = 0,
            FORWARD_EXPLICIT_EULER = 1,
            GENERALIZED_RUSH_LARSEN = 2,
            FORWARD_GENERALIZED_RUSH_LARSEN = 3,
            HYBRID_RUSH_LARSEN = 4
        };

        enum class IonicModelType : int
        {
            MITCHELL_SCHAEFFER = 0,
            FENTON_KARMA = 1,
            TEN_TUSCHER_PANFILOV = 2,
            MITCHELL_SCHAEFFER_TD_DEPENDENT = 10
        };


        /**
         * @brief The ReactionSolver class represents a model for handling the Reaction step of
         * the Monodomain/Bidomain models after operator splitting.
         */
        class ReactionSolver
        {
        private:
            ParFiniteElementSpace *fes = nullptr; // Finite element space (for dof info and registering fields) //< NOT OWNED
            int fes_truevsize; // Number of true dofs in the finite element space

            IonicModelType model_type;
            TimeIntegrationScheme scheme;

            int ode_substeps = 1; // Number of inner ODE time steps (1: dt = dt_ode)
            std::unique_ptr<GotranxODEModel>
                model; // Pointer to the model for ODE pointwise solution

            // Pointwise ODE data, stored flat in structure-of-arrays layout so that the
            // reaction kernels can run on device with coalesced access: entry k of dof i
            // lives at [k * fes_truevsize + i]. Using mfem::Vector (rather than
            // std::vector<std::vector<>>) puts these under the MFEM memory manager, so
            // host/device placement and transfers are handled automatically.
            Vector states;     // states (input) for each dof,  size [num_states * fes_truevsize]
            Vector values;     // values (output) for each dof, size [num_states * fes_truevsize]
            Vector parameters; // params for each dof,          size [num_params * fes_truevsize]

            std::vector<real_t> parameters_default; // Default parameters from the model

            Coefficient *stimulation_coeff = nullptr; // Stimulation function   //< NOT OWNED
            // Optional window outside which the stimulation is known to be zero,
            // used to skip the (host-side, per-substep) projection. See
            // SetStimulationWindow().
            bool has_stim_window = false;
            real_t stim_window_begin = 0.0;
            real_t stim_window_end = 0.0;
            mutable bool stim_vec_is_zero = false;
            // Set when the stimulation coefficient has no time dependence, so that it
            // is projected once instead of on every ODE substep.
            bool stim_time_independent = false;
            mutable bool stim_projected = false;
            // When false (default) the stimulation is sampled once per outer time
            // step; when true it is re-sampled at every ODE substep. See
            // EnableSubstepStimulusProjection().
            bool substep_stim_projection = false;
            ParGridFunction stimulation_gf;           // GridFunction to hold the stimulation values at dofs
            Coefficient *chi_coeff = nullptr;         // Chi coefficient function   //< NOT OWNED
            Coefficient *Cm_coeff = nullptr;          // Membrane capacitance function //< NOT OWNED
            ParGridFunction chi_gf;                   // GridFunction to hold the chi values at dofs
            ParGridFunction Cm_gf;                    // GridFunction to hold the Cm values at dofs
            mutable Vector stimulation_vec;           // Vector to hold the stimulation values at dofs
            mutable Vector chi_vec;                   // Vector to hold the chi values at dofs
            mutable Vector Cm_vec;                    // Vector to hold the Cm values at dofs

            // GridFunctions for temperature and damage dependency (optional)
            bool has_td_dependency = false;
            ParGridFunction *temperature_gf = nullptr;   //< NOT OWNED
            ParGridFunction *damage_gf = nullptr;        //< NOT OWNED
            mutable Vector temperature_vec;              // Vector to hold temperature values at dofs
            mutable Vector damage_vec;                   // Vector to hold damage values at dofs

            real_t td_A = 1.0;       // Moore term A (peak conductance scaling)
            real_t td_B = 0.0;       // Moore term B (peak conductance T-sensitivity)
            real_t td_Tref = 310.15;  // Reference temperature [K] (37 °C)
            real_t td_Q10 = 1.0;      // Q10 coefficient for gating kinetics
            std::function<real_t(real_t)> td_damage_func = nullptr; // Damage function
            std::vector<real_t> td_delta_tau = {};                  // Delta tau values for damage dependency (might be different for each time constant)
            std::vector<real_t> td_healthy_tau = {};                   // Original values of time constants for undamaged tissue

            // Device-resident mirrors of the thermal/damage data, populated lazily on
            // the first Step() that needs them. td_damage_func is a std::function and
            // cannot be called from device code, so it is applied on host into
            // damage_transformed and the kernel consumes the result.
            mutable Vector damage_transformed;
            mutable Array<int> td_tau_idx_d;
            mutable Vector td_healthy_tau_d;
            mutable Vector td_delta_tau_d;
            mutable bool td_device_data_ready = false;

            // We need this to: 1) possibly use it for output in DataCollection, 2) Update the state after change in Mesh/FESpace (AMR)
            std::vector<ParGridFunction *> states_gfs; // States grid functions for all states except potential
            std::vector<Vector *> states_vectors;      // Corresponding vectors for states

            real_t Vmin = -80;
            real_t Vmax = -20;
            real_t Vrange;
            real_t invVrange;

            inline real_t ToDimensionless(real_t u) const
            {
                return std::abs((u - Vmin) * invVrange);
            }
            
            inline real_t FromDimensionless(real_t u_dimless) const
            {
                return u_dimless * Vrange + Vmin;
            }

        public:
            /**
             * @brief Constructor for the ReactionSolver class.
             */
            ReactionSolver(ParFiniteElementSpace *fes, Coefficient *chi_coeff_, Coefficient *Cm_coeff_, IonicModelType model_type,
                           TimeIntegrationScheme solver_type = TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN, int dt_ode = 1);

            /**
             * @brief Destructor for the ReactionSolver class.
             */
            ~ReactionSolver();


            /**
             * @brief Set the temperature and damage grid functions for temperature and damage dependent models.
                During the setup phase, if a TD depedent model is used but no grid functions are provided,
                a warning is issued (code is still functional, but this is equivalent to having no dependency).
             */
            void SetThermalParameters(
                ParGridFunction *temperature_gf_ = nullptr,
                real_t A = 1.0,
                real_t B = 0.0,
                real_t T_ref = 310.15,
                real_t Q10 = 1.0);

            void SetDamageParameters(
                ParGridFunction *damage_gf_ = nullptr,
                std::function<real_t(real_t)> damage_func = nullptr,
                std::vector<real_t> delta_tau = {});


            /**
             * @brief Initializes the states and parameters.
             * @param initial_states Vector of initial states to set for all dofs. If empty, defaults are used.
             * @param params Vector of parameters to set for all dofs. If empty, defaults are used.
             * If the size of initial_states or params does not match the model's requirements, an error is raised.
             */
            void Setup(const std::vector<real_t> &initial_states, const std::vector<real_t> &params);
            void Setup() { Setup({}, {}); }

            /**
             * @brief Update the MonodomainDiffusionSolver in case of changes in Mesh or FiniteElementSpace
             */
            void Update();

            /**
             * @brief Sets the voltage range for dimensionless models.
             */
            void SetVRange(real_t V_min, real_t V_max)
            {
                Vmin = V_min;
                Vmax = V_max;
                Vrange = (Vmax - Vmin);
                invVrange = 1.0 / Vrange;
            }

            /**
             * @brief Get the default states and parameters from the ionic model.
             * Useful for initializing and potentially modifying the parameters passed to Setup(). 
             */
            void GetDefaultStates(std::vector<real_t> &default_states);
            void GetDefaultParameters(std::vector<real_t> &default_params);


            /**
             * @brief Get a ParGridFunction representing a specific state variable.
             * @param state_index Index of the state variable to retrieve.
             */
            ParGridFunction *GetStateGridFunction(int state_index);

            /**
             * @brief Register fields for output.
             */
            void RegisterFields(DataCollection &dc);

            /**
             * @brief Get model object.
             */
            GotranxODEModel* GetModel() { return model.get(); }

            /**
             * @brief Update the potential t-dof vector from the internal state.
             */
            void GetPotential(Vector &u);

            /**
             * @brief Get the stimulation grid function.
             * @return Pointer to the stimulation ParGridFunction.
             */
            ParGridFunction* GetStimulationGF() { return &stimulation_gf; }

            /**
             * @brief Updates the internal state from the potential vector u.
             */
            void SetPotential(const Vector &u);

            //=================================================================
            //  Enforcing the stimulation efficiently
            //=================================================================
            //
            // The stimulation current reaches the ionic model as a per-dof value, so
            // it has to be projected from its Coefficient onto the FE space. That
            // projection (ParGridFunction::ProjectCoefficient + GetTrueDofs) is a
            // *host* operation: it loops elements, builds an ElementTransformation,
            // and evaluates the user's callback at every dof. Measured at ~225 ns per
            // dof, i.e. ~48 ms for 216k dofs.
            //
            // Done naively it runs once per ODE substep, which makes it dominate
            // everything else. On a 216k-dof 3D case the split was:
            //
            //     stimulation projection ... 48.5 ms/substep   (99.9%)
            //     ODE integration kernel ...  0.05 ms/substep   (0.1%)
            //
            // Under a device backend it is also a synchronisation point, since the
            // projected values must then be pushed back to the GPU.
            //
            // The ODE integration itself is essentially free. Any time spent in the
            // reaction step beyond a millisecond or so is almost certainly this.
            //
            // WHICH STRATEGY TO USE
            //
            // Different stimulation strategies are available.
            // For each different need a user might have we provide 
            // the optimal strategy to avoid the per-substep projection. 
            //
            //  (a) Coefficient is a function of x only (time independent)  
            //          ->  SetStimulation(c, true)
            //      A fixed spatial mask; the model gates it in time. Projected once for the whole simulation. 
            //      NOTE: if you encode timing in the coefficient itself, this will not be re-projected when the timing changes.  
            //
            //  (b) Coefficient depends on t but vanishes outside a known interval
            //          ->  SetStimulation(c, false) + SetStimulationWindow(t0, t1)
            //      Projected only while the stimulus can be nonzero; elsewhere the
            //      t-dof vector is zeroed once and reused. 
            //
            //  (c) Coefficient depends on t with a window only known at runtime
            //          ->  call SetStimulationWindow() again later
            //      An S2 stimulus whose timing is decided from the solution (e.g.
            //      when a wavefront reaches a probe) still fits (b); just MOVE the
            //      window when the trigger fires:
            //
            //          if (S2_triggered)
            //          {
            //              reaction_solver->SetStimulationWindow(t_S2, t_S2 + d_S2);
            //          }
            //
            //      Do NOT use ClearStimulationWindow() for this. It restores the
            //      always-project default, so you would pay a full projection on
            //      every substep for the rest of the run -- including the long quiet
            //      stretch after S2, which is usually most of the simulation.
            //      Moving the window is both correct and cheap: the window bounds are
            //      re-checked on every substep, so a new window re-arms the
            //      projection by itself, with no need to reset anything first.
            //
            //      HAZARD: if you set a window and then forget to move it when the
            //      trigger fires, the S2 stimulus is silently dropped -- the run
            //      completes with no error and simply never re-stimulates. Keep the
            //      SetStimulationWindow() call right next to the code that sets the
            //      S2 start time so the two cannot drift apart.
            //
            //  (d) Genuinely time dependent everywhere  ->  do nothing
            //      The conservative default projects every substep. Correct, but
            //      expect the reaction step to cost ~1 projection per substep. If
            //      this shows up in a profile, consider reformulating the stimulus
            //      as separable, I(x,t) = A(t) * S(x): project S(x) once via (a) and
            //      fold A(t) into the ionic model's stimulus parameters.
            //
            // WARNING: (b) and (c) are assertions the solver cannot verify. If the
            // coefficient is in fact nonzero outside the declared window, that part
            // of the stimulus is silently dropped -- the run will not fail, it will
            // just produce wrong results. 
            //
            // A note on ode_substeps: the stimulation is sampled once per *outer*
            // time step by default, so running with -dode N does NOT multiply the
            // projection cost -- only the ODE integration, which is the cheap part,
            // scales with N. EnableSubstepStimulusProjection() restores per-substep
            // sampling for the rare case where a stimulus edge has to be resolved
            // more finely than dt; that reinstates the N-fold cost. Strategies (a)
            // and (b) apply on top of either choice.
            //=================================================================

            /**
             * @brief Sets the stimulation function for the ionic model.
             *
             * @param stim  Spatial (and possibly temporal) stimulation current.
             * @param time_independent  Set when @a stim does not depend on time, i.e.
             *   it is a fixed spatial mask and the temporal gating is done by the
             *   ionic model itself through its IstimStart / IstimEnd /
             *   IstimPulseDuration parameters. This is the case for a
             *   FunctionCoefficient built from a function of x only. Strategy (a)
             *   above: the coefficient is projected once and reused, which is exact.
             *
             * @note If the coefficient does depend on time, leave this false (the
             *   default) and use SetStimulationWindow() instead when the stimulus is
             *   known to vanish outside a bounded interval.
             */
            void SetStimulation(Coefficient *stim, bool time_independent = false);

            /**
             * @brief Declare the time interval outside of which a *time dependent*
             * stimulation coefficient is identically zero.
             *
             * Strategy (b) above. This is a contract: the caller asserts the
             * coefficient evaluates to zero for t outside [t_begin, t_end], letting
             * the solver skip the per-substep projection there and reuse a zeroed
             * t-dof vector instead. If no window is set, the coefficient is projected
             * on every substep (the conservative default).
             *
             * Call again to move the window -- strategy (c) -- e.g. when an S2
             * stimulus is triggered at a time determined at runtime. The bounds are
             * re-checked on every substep, so a new window takes effect immediately
             * and re-arms the projection on its own; nothing needs resetting first.
             *
             * The window is compared against the time at which the coefficient would
             * be evaluated, i.e. the end of the current ODE substep, so a window that
             * exactly matches the pulse [t_start, t_start + duration] is correct.
             *
             * @warning Not verifiable by the solver. If the coefficient is nonzero
             * outside the window, that part of the stimulus is silently discarded and
             * the run produces wrong results without any error. Validate a new
             * protocol once against the unflagged default.
             */
            void SetStimulationWindow(real_t t_begin, real_t t_end)
            {
                stim_window_begin = t_begin;
                stim_window_end = t_end;
                has_stim_window = true;
            }

            /**
             * @brief Drop the stimulation window and go back to projecting the
             * coefficient on every ODE substep.
             *
             * This is an escape hatch, not part of the normal S1/S2 flow: use it when
             * the stimulus stops being characterisable, e.g. switching to a protocol
             * whose timing cannot be bounded in advance. It trades all of the saving
             * back for safety.
             *
             * To retarget an existing window at a new pulse -- the usual case -- call
             * SetStimulationWindow() again instead. Clearing it would leave the
             * solver projecting every substep for the remainder of the run, which is
             * typically far longer than the pulse you were trying to catch.
             */
            void ClearStimulationWindow() { has_stim_window = false; }

            /**
             * @brief Re-sample the stimulation coefficient at every ODE substep
             * instead of once per outer time step.
             *
             * DEFAULT (disabled): the coefficient is projected once per call to
             * Step(), evaluated at the end of the outer step (t + dt), and that one
             * t-dof vector is reused by all ode_substeps inner ODE steps.
             *
             * ENABLED: the coefficient is re-projected before each substep, at
             * t + k*dt_ode. This resolves a stimulus that turns on or off *within* a
             * time step to substep resolution, at N times the projection cost.
             *
             * WHEN IT MATTERS
             *
             * With ode_substeps == 1 the two are identical -- dt_ode == dt, so the
             * single sample is taken at exactly the same time either way (verified
             * bit-for-bit). The setting only has any effect when running with more
             * than one ODE substep (-dode N), where it trades accuracy of the
             * stimulus *timing* against N host-side projections per step.
             *
             * Measured, 3D spiral protocol (time dependent coefficient), GPU:
             *
             *     dode   reaction/step, default   reaction/step, enabled
             *        1            5.9 ms                   5.9 ms
             *        4            6.2 ms                  26.4 ms
             *        8            6.1 ms                  45.5 ms
             *
             * i.e. the default is flat in N while the enabled path scales linearly
             * with it, because the ODE integration itself is nearly free compared to
             * one projection.
             *
             * WHAT YOU GIVE UP BY LEAVING IT OFF
             *
             * The stimulus edge is resolved to dt rather than dt_ode, so a pulse is
             * effectively snapped to the outer step grid: its on/off transitions move
             * by up to dt. For the usual case -- a multi-millisecond pulse with
             * dt ~ 0.05 ms -- that is a sub-percent change in delivered charge. It
             * matters if the pulse is comparable to dt, and a pulse *shorter* than dt
             * can be missed entirely (that is equally true of the enabled path unless
             * dt_ode also resolves it).
             *
             * Measured on the same spiral case, as a relative difference in ||u||_2
             * after 400 steps:
             *
             *     dode=1   0          (identical by construction)
             *     dode=4   1.2e-5
             *     dode=8   2.5e-5
             *
             * For scale, refining dode from 1 to 8 changes ||u||_2 by 4.3e-4 -- about
             * 17x more. The error introduced by sampling the stimulus once per step
             * is therefore well below the ODE discretisation error that substepping
             * exists to reduce, which is why per-step sampling is the default: paying
             * Nx for the projection would buy accuracy you are already not resolving.
             *
             * Note this setting is orthogonal to the strategies above: a
             * time-independent coefficient (a) is projected once regardless, and a
             * declared window (b) still suppresses projection outside the window.
             * This only controls how often the coefficient is sampled *within* a step
             * that does need it.
             */
            void EnableSubstepStimulusProjection(bool enable = true)
            {
                substep_stim_projection = enable;
            }

            /**
             * @brief Solves the ionic model.
             */
            void Step(Vector &x, real_t &t, real_t &dt, bool provisional = false);

        private:
            /**
             * @brief Project the stimulation coefficient for time @a t_stim, unless
             * one of the opt-in strategies says it can be skipped.
             */
            void ProjectStimulation(real_t t_stim);

        public:

            /**
             * @brief Prints the conversion index table.
             */
            void PrintIndexTable();
        };

    } // namespace electrophysiology
} // namespace mfem