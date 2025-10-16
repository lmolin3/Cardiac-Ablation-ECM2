#pragma once

#include "mfem.hpp"

// Include all available ionic models here
#include "../ionic_models/mitchell_schaeffer_2003.h"

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
            MITCHELL_SCHAEFFER = 0
        };


        /**
         * @brief The ReactionSolver class represents a model for handling the Reaction step of
         * the Monodomain/Bidomain models after operator splitting.
         */
        class ReactionSolver
        {
        private:
            int fes_truevsize; // Number of true dofs in the finite element space

            IonicModelType model_type;
            TimeIntegrationScheme scheme;

            int ode_substeps = 1; // Number of inner ODE time steps (1: dt = dt_ode)
            std::unique_ptr<GotranxODEModel>
                model; // Pointer to the model for ODE pointwise solution

            std::vector<std::vector<double>>
                states; // Data structure holding the states (input) for each dof, size [fes_truevsize][num_states]
            std::vector<std::vector<double>>
                values; // Data structure holding the values (output) for each dof, size [fes_truevsize][num_values]
            std::vector<std::vector<double>>
                parameters; // Data structure holding the params for each dof, size          [fes_truevsize][num_params]

            Coefficient *stimulation_coeff = nullptr; // Stimulation function   //< NOT OWNED
            ParGridFunction stimulation_gf;           // GridFunction to hold the stimulation values at dofs
            mutable Vector stimulation_vec;           // Vector to hold the stimulation values at dofs

        public:
            /**
             * @brief Constructor for the ReactionSolver class.
             */
            ReactionSolver(ParFiniteElementSpace &fes, IonicModelType model_type, TimeIntegrationScheme solver_type = TimeIntegrationScheme::GENERALIZED_RUSH_LARSEN, int dt_ode = 1);

            /**
             * @brief Destructor for the ReactionSolver class.
             */
            ~ReactionSolver() = default;

            /**
             * @brief Initializes the states and parameters.
             * @param initial_states Vector of initial states to set for all dofs. If empty, defaults are used.
             * @param params Vector of parameters to set for all dofs. If empty, defaults are used.
             * If the size of initial_states or params does not match the model's requirements, an error is raised.
             */
            void Setup(std::vector<double> initial_states, std::vector<double> params);
            void Setup() { Setup({}, {}); }

            /**
             * @brief Get the default states and parameters from the ionic model.
             * Useful for initializing and potentially modifying the parameters passed to Setup(). 
             */
            void GetDefaultStates(std::vector<double> &default_states);
            void GetDefaultParameters(std::vector<double> &default_params);

            /**
             * @brief Update the potential t-dof vector from the internal state.
             */
            void GetPotential(Vector &u);

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