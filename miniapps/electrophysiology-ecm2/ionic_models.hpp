#ifndef IONICMODEL_HPP
#define IONICMODEL_HPP

/**
 * @brief The IonicModel class represents a model for handling ionic states and parameters.
 */
class IonicModel {
private:
    static const int num_param = -1;    // Initialized depending on model
    static const int num_states = -1;   // Initialized depending on model

    static const int dofs = -1;         // Initialized depending on FE discretization

    int model_type;                      // Ionic Model:    0 = MitchellSchaeffer, 1 = FentonKarma
    int solver_type                      // ODE solver:     0 = Explicit Euler, 1 = Rush Larsen (RL), 2 = Generalized Rush Larsen (GRL), 3 = Hybrid GRL, 4 = Simplified Implicit Euler
    int ode_steps = 1;                   // Number of inner ODE time steps (1: dt = dt_ode)
    ODEModel* model = nullptr;

    std::vector<std::vector<double>> states;
    std::vector<std::vector<double>> parameters;

    mutable std::vector<double> values;


public:
    /**
     * @brief Constructor for the IonicModel class.
     */
    IonicModel(int dofs, int model_type=0, int solver_type=0, int dt_ode=1);

    /**
     * @brief Destructor for the IonicModel class.
     */
    ~IonicModel();

    /**
     * @brief Initializes the states and parameters.
     */
    void Init();

    /**
     * @brief Prints the conversion index table.
     */
    void PrintIndexTable();


    /**
     * @brief Computes the ionic current.
     */
    void ComputeIonicCurrent(double time);

    /**
     * @brief Solves the ionic model.
     */
    void Solve(double time, double dt, Vector* potential);

    /**
     * @brief Update membrane potential in the ODE model.
     */
    void UpdatePotential(Vector* new_potential);

};

#endif // IONICMODEL_HPP
