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
            MITCHELL_SCHAEFFER_TD_DEPENDENT = 2
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

            std::vector<std::vector<real_t>>
                states; // Data structure holding the states (input) for each dof, size [fes_truevsize][num_states]
            std::vector<std::vector<real_t>>
                values; // Data structure holding the values (output) for each dof, size [fes_truevsize][num_values]
            std::vector<std::vector<real_t>>
                parameters; // Data structure holding the params for each dof, size          [fes_truevsize][num_params]

            std::vector<real_t> parameters_default; // Default parameters from the model

            Coefficient *stimulation_coeff = nullptr; // Stimulation function   //< NOT OWNED
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

            real_t td_A = 1.0;      // Moore term A
            real_t td_B = 0.0;      // Moore term B
            real_t td_Tref = 310.15; // Reference temperature
            real_t td_Q10 = 1.0;    // Q10 coefficient
            std::function<real_t(real_t)> td_damage_func = nullptr; // Damage function
            std::vector<real_t> td_delta_tau = {};                  // Delta tau values for damage dependency (might be different for each time constant)
            std::vector<real_t> td_healthy_tau = {};                   // Original values of time constants for undamaged tissue

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

            /**
             * @brief Sets the stimulation function for the ionic model.
             */
            void SetStimulation(Coefficient *stim);

            /**
             * @brief Solves the ionic model.
             */
            void Step(Vector &x, real_t &t, real_t &dt, bool provisional = false);

            /**
             * @brief Prints the conversion index table.
             */
            void PrintIndexTable();
        };

    } // namespace electrophysiology
} // namespace mfem