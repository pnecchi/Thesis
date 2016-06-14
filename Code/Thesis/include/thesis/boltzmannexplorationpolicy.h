#ifndef BOLTZMANNEXPLORATIONPOLICY_H
#define BOLTZMANNEXPLORATIONPOLICY_H

#include <thesis/stochasticpolicy.h>
#include <armadillo>
#include <vector>
#include <random>

class BoltzmannExplorationPolicy : public StochasticPolicy
{
    public:
        // Default constructor
        BoltzmannExplorationPolicy(size_t dimObservation_,
                                   std::vector<double> possibleActions_);

        // Default destructor
        virtual ~BoltzmannExplorationPolicy() = default;

        // Getter methods for sizes
        virtual size_t getDimParameters() const { return dimParameters; }

        // Getter and setter methods for parameters
        virtual arma::vec getParameters() const;
        virtual void setParameters(arma::vec const &parameters);

        // Select action given
        virtual arma::vec getAction(arma::vec const &observation) const;

        // Likelihood score function
        virtual arma::vec likelihoodScore(arma::vec const &observation,
                                          arma::vec const &action) const;

    private:
        // Virtual inner clone method
        virtual std::unique_ptr<Policy> cloneImpl() const;

        // Initialize policy parameters
        void initializeParameters();

        // Possible actions
        std::vector<double> possibleActions;

        // Policy sizes
        size_t numPossibleActions;
        size_t dimParametersPerAction;
        size_t dimParameters;

        // Policy parameters: Theta = [ theta_1 | thetha_2 | ... | theta_D ]
        arma::mat parametersMat;
        arma::vec paramatersVec;  // Linearized matrix (shares memory)

        // Boltzmann probability distribution and random number generator
        // Need to be mutable because the generator state changes when
        // simulating an action.
        // TODO: check boost for simulating random variables
        mutable std::mt19937 generator;
        mutable std::vector<double> boltzmannProbabilities;

        // TODO: Consider generic features of the input
};

#endif // BOLTZMANNEXPLORATIONPOLICY_H
