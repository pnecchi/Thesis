#include "thesis/LongShortPolicy.h"

LongShortPolicy::LongShortPolicy(size_t dimObservation_,
                           double paramMinValue_,
                           double paramMaxValue_)
    : Policy(dimObservation_, 2ul),
      dimParameters(dimObservation_ + 1),
      parameters(dimObservation_ + 1),
      paramMinValue(paramMinValue_),
      paramMaxValue(paramMaxValue_)
{
    initializeParameters();
}

void LongShortPolicy::initializeParameters()
{
    parameters.randu();
    parameters -= 0.5;
    parameters *= 0.001;
}

void LongShortPolicy::setParameters(arma::vec const & parameters_)
{
    parameters = arma::clamp(parameters_, paramMinValue, paramMaxValue);
}

arma::vec LongShortPolicy::getAction(arma::vec const & observation_) const
{
    // Compute features
    arma::vec features(dimParameters);
    features(0) = 1.0;
    features.rows(1, dimParameters - 1) = observation_;

    // Compute action
    double activation = arma::dot(parameters, features);
    arma::vec action(2);
    action(0) = (activation > 0.0) ? 1.0 : -1.0;
    action(1) = - action(0);
    return action;
}

void LongShortPolicy::reset()
{
    initializeParameters();
}

std::unique_ptr<Policy> LongShortPolicy::cloneImpl() const
{
    return std::unique_ptr<Policy>(new LongShortPolicy(*this));
}
