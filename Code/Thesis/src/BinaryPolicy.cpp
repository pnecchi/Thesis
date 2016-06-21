#include "thesis/BinaryPolicy.h"

BinaryPolicy::BinaryPolicy(size_t dimObservation_)
    : Policy(dimObservation_, 1ul),
      dimParameters(dimObservation_ + 1),
      parameters(dimObservation_ + 1)
{
    initializeParameters();
}

void BinaryPolicy::initializeParameters()
{
    parameters.randu();
    parameters -= 0.5;
    parameters *= 0.01;
}

arma::vec BinaryPolicy::getAction(arma::vec const & observation_) const
{
    // Compute features
    arma::vec features(dimParameters);
    features(0) = 1.0;
    features.rows(1, dimParameters - 1) = observation_;

    // Compute action
    double activation = arma::dot(parameters, features);
    arma::vec action(1);
    action(0) = (activation > 0.0) ? 1.0 : -1.0;
    return action;
}

void BinaryPolicy::reset()
{
    initializeParameters();
}

std::unique_ptr<Policy> BinaryPolicy::cloneImpl() const
{
    return std::unique_ptr<Policy>(new BinaryPolicy(*this));
}
