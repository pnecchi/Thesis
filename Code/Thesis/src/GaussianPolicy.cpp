#include "thesis/GaussianPolicy.h"
#include <random>

GaussianPolicy::GaussianPolicy(size_t dimObservation_,
                               size_t dimAction_)
    : StochasticPolicy(dimObservation_, dimAction_),
      dimParameters((dimObservation_ + 1) * dimAction_ + 1),
      parameters(dimParameters),
      psiMat(parameters.memptr(), dimObservation_ + 1, dimAction_, false, false)
{
    initializeParameters();
}

void GaussianPolicy::initializeParameters()
{
    psiMat.randu();
    psiMat -= 0.5;
    psiMat *= 0.01;
    parameters(dimParameters - 1) = 1;  // sigma
}

arma::vec GaussianPolicy::getParameters() const
{
    return parameters;
}

void GaussianPolicy::setParameters(arma::vec const &parameters_)
{
    parameters = parameters_;
    if (parameters(dimParameters - 1) < 0)
        parameters(dimParameters - 1) = 0.01;
}

arma::vec GaussianPolicy::getAction(arma::vec const &observation_) const
{
    // Compute features
    arma::vec features(getDimObservation() + 1, arma::fill::zeros);
    features(0) = 1.0;
    features.rows(1, features.size() - 1) = observation_;

    // Compute mean
    arma::vec mean = psiMat.t() * features;
    double stddev = parameters(dimParameters - 1);

    // Simulate action
    arma::vec action(getDimAction());
    for (size_t i = 0; i < getDimAction(); ++i)
    {
        std::normal_distribution<double> d(mean[i], stddev);
        action[i] = d(generator);
    }
    return action;
}

arma::vec GaussianPolicy::likelihoodScore(arma::vec const &observation_,
                                          arma::vec const &action_) const
{
    // Compute features
    arma::vec features(getDimObservation() + 1);
    features(0) = 1.0;
    features.rows(1, features.size() - 1) = observation_;

    // Compute mean and stddev
    arma::vec mean = psiMat.t() * features;
    double stddev = parameters(dimParameters - 1);
    double stddev2 = stddev * stddev;
    double stddev3 = stddev2 * stddev;

    // Compute gradient with respect to parameters
    arma::vec deltaAction = action_ - mean;
    arma::vec gradientMean = deltaAction / stddev2;
    double gradientSigma = arma::norm(deltaAction, 2) / stddev3 - getDimAction() / stddev;

    // Compute gradient with respect to mean hyperparameters
    arma::vec likScore(dimParameters, arma::fill::zeros);
    likScore(dimParameters - 1) = gradientSigma;
    for (size_t i = 0; i < getDimAction(); ++i)
    {
        likScore.rows((getDimObservation() + 1) * i,
                      (getDimObservation() + 1) * (i + 1) - 1) =
            features * gradientMean(i);
    }
    return likScore;
}

std::unique_ptr<Policy> GaussianPolicy::cloneImpl() const
{
    return std::unique_ptr<Policy>(new GaussianPolicy(*this));
}

void GaussianPolicy::reset()
{
    initializeParameters();
}
