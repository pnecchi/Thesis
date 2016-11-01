#include "thesis/NpgpePolicy.h"

NPGPEPolicy::NPGPEPolicy(Policy const &policy_,
                         double resamplingProbability_)
    : StochasticPolicy(policy_.getDimObservation(), policy_.getDimAction()),
      policyPtr(policy_.clone()),
      resamplingProbability(resamplingProbability_),
      dimParameters(policy_.getDimParameters()),
      dimHyperParameters(dimParameters * (dimParameters + 1)),
      parameters(dimHyperParameters, arma::fill::zeros),
      mean(parameters.memptr(), dimParameters, false, false),
      cholFactor(parameters.memptr() + dimParameters, dimParameters, dimParameters, false, false),
      xi(dimParameters),
      generator(214),
      gaussianDistr(0.0, 1.0)
{
    initializeParameters();
}

void NPGPEPolicy::initializeParameters()
{
    parameters.zeros();
    cholFactor.diag().ones();
    cholFactor *= 0.1;
}

NPGPEPolicy::NPGPEPolicy(NPGPEPolicy const &other_)
    : StochasticPolicy(other_.getDimObservation(), other_.getDimAction()),
      policyPtr(other_.policyPtr->clone()),
      resamplingProbability(other_.resamplingProbability),
      dimParameters(other_.dimParameters),
      dimHyperParameters(other_.dimHyperParameters),
      parameters(other_.parameters),
      mean(parameters.memptr(), dimParameters, false, false),
      cholFactor(parameters.memptr() + dimParameters, dimParameters, dimParameters, false, false),
      xi(other_.xi),
      generator(other_.generator()),
      gaussianDistr(other_.gaussianDistr)
{
    // Nothing to do
}

arma::vec NPGPEPolicy::getAction(arma::vec const &observation_) const
{
    // Simulate policy parameters: w = mean + cholFactor' * xi
    xi.imbue( [&]() { return gaussianDistr(generator); } );
    arma::vec newPolicyParameters = mean + cholFactor.t() * xi;
    policyPtr->setParameters(newPolicyParameters);

    // Select action
    return policyPtr->getAction(observation_);
}

arma::vec NPGPEPolicy::likelihoodScore(arma::vec const &observation_,
                                       arma::vec const &action_) const
{
    arma::vec likScore(dimHyperParameters);
    arma::vec policyParameters = policyPtr->getParameters();
    likScore.rows(0, dimParameters-1) = policyParameters - mean;
    likScore.rows(dimParameters, dimHyperParameters-1) = arma::vectorise(
        (arma::trimatu(xi * xi.t()) -
         0.5 * arma::diagmat(xi * xi.t()) -
         0.5 * arma::diagmat(arma::eye(dimParameters, dimParameters))) *
         cholFactor);
    return likScore;
}

void NPGPEPolicy::reset()
{
    policyPtr->reset();
    initializeParameters();
}

std::unique_ptr<Policy> NPGPEPolicy::cloneImpl() const
{
    return std::unique_ptr<Policy>(new NPGPEPolicy(*this));
}
