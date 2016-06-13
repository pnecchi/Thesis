#include "thesis/boltzmannexplorationpolicy.h"
#include <cmath>   /* abs */
#include <limits>  /* eps */

BoltzmannExplorationPolicy::BoltzmannExplorationPolicy(size_t dimObservation_,
                                                       std::vector<double> possibleActions_)
    : StochasticPolicy(dimObservation_, 1ul),
      possibleActions(possibleActions_),
      numPossibleActions(possibleActions.size()),
      dimParametersPerAction(dimObservation_ + 1),
      dimParameters(dimParametersPerAction * numPossibleActions),
      parametersMat(dimObservation_, numPossibleActions),
      boltzmannDistribution()
{
    initializeParameters();
}

void BoltzmannExplorationPolicy::initializeParameters()
{
    parametersMat.randu();
    parametersMat -= 0.5;
    parametersMat *= 0.001;
}

arma::vec BoltzmannExplorationPolicy::getParameters() const
{
    return arma::vectorise(parametersMat);
}

void BoltzmannExplorationPolicy::setParameters(arma::vec const &parameters)
{
    parametersMat = arma::mat(parameters,
                              dimParametersPerAction,
                              numPossibleActions,
                              false,   /* copy_aux_mem */
                              false);  /* strict */
}

arma::vec BoltzmannExplorationPolicy::getAction(arma::vec const &observation) const
{
    // Compute actions probabilities according to Boltzmann distribution
    arma::vec boltzmannWeights = arma::exp(parametersMat.t() * observation)
    boltzmannDistribution = std::discrete_distribution<>(boltzmannWeights.begin(),
                                                         boltzmannWeights.end());

    // Generate action
    size_t actionIdx = boltzmannDistribution();
    arma::vec action(1);
    action(0) = possibleActions[actionIdx];
    return action;
}

arma::vec BoltzmannExplorationPolicy::likelihoodScore(arma::vec const &observation,
                                                      arma::vec const &action) const
{
    // Get Boltzmann probabilities
    std::vector<double> boltzmannProbabilities = boltzmannDistribution.probabilities();

    // Compute likelihood score
    arma::vec likScore(dimParameters);
    double coeff = 0.0;
    for (size_t i = 0; i < numPossibleActions; ++i)
    {
        // Compute coefficient for the different actions
        if (std::abs(action[0] - possibleActions[i]) < std::numeric_limits<double>::epsilon())
            coeff = 1.0 - boltzmannProbabilities[i];
        else
            coeff = - boltzmannProbabilities[i];

        // Compute likScore associated to the given action
        likScore.rows(i * dimParametersPerAction, (i + 1) * dimParametersPerAction - 1)
            = coeff * observation;
    }

    return likScore;
}

std::unique_ptr<Policy> BoltzmannExplorationPolicy::cloneImpl() const
{
    return std::unique_ptr<Policy>(new BoltzmannExplorationPolicy(*this));
}


