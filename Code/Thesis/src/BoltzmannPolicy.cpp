#include "thesis/BoltzmannPolicy.h"
#include <cmath>   /* abs */
#include <limits>  /* eps */
#include <random>
#include <iostream>
#include <algorithm>  /* find */
#include <fstream>

BoltzmannPolicy::BoltzmannPolicy(size_t dimObservation_,
                                 std::vector<double> possibleActions_)
    : StochasticPolicy(dimObservation_, 1ul),
      possibleActions(possibleActions_),
      numPossibleActions(possibleActions.size()),
      dimParametersPerAction(dimObservation_ + 1),
      dimParameters(dimParametersPerAction * (numPossibleActions - 1)),
      parametersMat(dimParametersPerAction, numPossibleActions - 1),
      parametersVec(parametersMat.memptr(), dimParameters, false, false),
      generator(),
      boltzmannProbabilities(numPossibleActions)
{
    initializeParameters();
}

void BoltzmannPolicy::initializeParameters()
{
    parametersMat.randu();
    parametersMat -= 0.5;
    parametersMat *= 0.1;
}

arma::vec BoltzmannPolicy::getParameters() const
{
    return parametersVec;
}

void BoltzmannPolicy::setParameters(arma::vec const &parameters)
{
    parametersVec = parameters;
}

arma::vec BoltzmannPolicy::getAction(arma::vec const &observation_) const
{
    // Compute features
    arma::vec features(dimParametersPerAction);
    features(0) = 1.0;
    features.rows(1, features.n_elem - 1) = observation_;

    // Compute actions probabilities according to Boltzmann distribution
    arma::vec boltzmannWeights(numPossibleActions, arma::fill::ones);
    boltzmannWeights.rows(0, boltzmannWeights.n_elem - 2) = arma::exp(parametersMat.t() * features);
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

arma::vec BoltzmannPolicy::likelihoodScore(arma::vec const &observation_,
                                           arma::vec const &action_) const
{
    // Compute features
    arma::vec features(dimParametersPerAction);
    features(0) = 1.0;
    features.rows(1, features.n_elem - 1) = observation_;

    // Find index of selected action
    size_t actionIdx = std::distance(possibleActions.begin(),
                                     std::find(possibleActions.begin(), possibleActions.end(), action_[0]));

    // Compute likelihood score
    arma::vec likScore(dimParameters);
    for (size_t i = 0; i < numPossibleActions - 1; ++i)
    {
        likScore.rows(i * dimParametersPerAction, (i + 1) * dimParametersPerAction - 1)
            = - boltzmannProbabilities[i] * features;
    }
    if (actionIdx != numPossibleActions - 1)
        likScore.rows(actionIdx * dimParametersPerAction,
                      (actionIdx + 1) * dimParametersPerAction - 1) += features;
    return likScore;
}

std::unique_ptr<Policy> BoltzmannPolicy::cloneImpl() const
{
    return std::unique_ptr<Policy>(new BoltzmannPolicy(*this));
}

void BoltzmannPolicy::reset()
{
    initializeParameters();
}

