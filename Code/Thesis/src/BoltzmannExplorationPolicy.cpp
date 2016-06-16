#include "thesis/BoltzmannExplorationPolicy.h"
#include <cmath>   /* abs */
#include <limits>  /* eps */
#include <random>

BoltzmannExplorationPolicy::BoltzmannExplorationPolicy(size_t dimObservation_,
                                                       std::vector<double> possibleActions_)
    : StochasticPolicy(dimObservation_, 1ul),
      possibleActions(possibleActions_),
      numPossibleActions(possibleActions.size()),
      dimParametersPerAction(dimObservation_ + 1),
      dimParameters(dimParametersPerAction * numPossibleActions),
      parametersMat(dimParametersPerAction, numPossibleActions),
      paramatersVec(parametersMat.memptr(), dimParameters, false, false),
      generator(),
      boltzmannProbabilities(numPossibleActions)
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
    return paramatersVec;
}

void BoltzmannExplorationPolicy::setParameters(arma::vec const &parameters)
{
    paramatersVec = parameters;
}

arma::vec BoltzmannExplorationPolicy::getAction(arma::vec const &observation_) const
{
    // Compute features
    arma::vec features(dimParametersPerAction);
    features(0) = 1.0;
    features.rows(1, features.n_elem - 1) = observation_;

    // Compute actions probabilities according to Boltzmann distribution
    arma::vec boltzmannWeights = arma::exp(parametersMat.t() * features);
    std::discrete_distribution<int> boltzmannDistribution(boltzmannWeights.begin(),
                                                          boltzmannWeights.end());

    // Cache action probabilities
    boltzmannProbabilities = boltzmannDistribution.probabilities();

    // Generate action
    int actionIdx = boltzmannDistribution(generator);
    arma::vec action(1);
    action(0) = possibleActions[actionIdx];
    return action;
}

arma::vec BoltzmannExplorationPolicy::likelihoodScore(arma::vec const &observation,
                                                      arma::vec const &action) const
{
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


