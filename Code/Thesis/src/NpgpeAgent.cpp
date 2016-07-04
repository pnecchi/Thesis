#include "thesis/NpgpeAgent.h"

NPGPEAgent::NPGPEAgent(Policy const &policy_,
                       double alpha_,
                       double discountFactor_)
    : policyPtr(policy_.clone()),
      mean(policy_.getDimParameters(), arma::fill::zeros),
      choleskyFactor(policy_.getDimParameters(), policy_.getDimParameters(), arma::fill::zeros),
      generator(215),
      gaussianDistr(0.0, 1.0),
      xi(policy_.getDimParameters()),
      baseline(alpha_),
      gradientMean(policy_.getDimParameters(), arma::fill::zeros),
      gradientChol(policy_.getDimParameters(), policy_.getDimParameters(), arma::fill::zeros),
      alpha(alpha_),
      discountFactor(discountFactor_),
      observation(policy_.getDimObservation()),
      action(policy_.getDimAction())
{
    initializeParameters();
}

NPGPEAgent::NPGPEAgent(NPGPEAgent const &other_)
    : policyPtr(other_.policyPtr->clone()),
      mean(other_.mean),
      choleskyFactor(other_.choleskyFactor),
      generator(other_.generator),
      gaussianDistr(other_.gaussianDistr),
      xi(other_.xi),
      baseline(other_.baseline),
      gradientMean(other_.gradientMean),
      gradientChol(other_.gradientChol),
      alpha(other_.alpha),
      discountFactor(other_.discountFactor),
      observation(other_.observation),
      action(other_.action)
{
    /* Nothing to do */
}

void NPGPEAgent::initializeParameters()
{
    mean.zeros();
    choleskyFactor.zeros();
    choleskyFactor.diag().ones();
}

std::unique_ptr<Agent> NPGPEAgent::clone() const
{
    return std::unique_ptr<Agent>(new NPGPEAgent(*this));
}

arma::vec NPGPEAgent::getAction()
{
    // Simulate policy parameters: w = mean + cholFactor * xi
    xi.imbue( [&]() { return gaussianDistr(generator); } );
    policyPtr->setParameters(mean + choleskyFactor * xi);

    // Select action
    return policyPtr->getAction(observation);
}

void NPGPEAgent::learn()
{
    // 1) Update baseline
    baseline.dumpOneResult(reward);
    double b = baseline.getStatistics()[0][0];

    // 2) Compute likelihood score
    arma::vec likelihoodMean = policyPtr->getParameters() - mean;
    arma::mat likelihoodChol =
        (arma::trimatu(xi * xi.t()) -
        0.5 * arma::diagmat(xi * xi.t()) -
        0.5 * arma::eye(policyPtr->getDimParameters(), policyPtr->getDimParameters())) *
        choleskyFactor.t();

    // 3) Update gradients
    gradientMean = discountFactor * gradientMean + likelihoodMean;
    gradientChol = discountFactor * gradientChol + likelihoodChol;

    // 4) Update hyperparameters
    mean += alpha * (reward - b) * gradientMean;
    choleskyFactor += alpha * (reward - b) * gradientChol;
}

void NPGPEAgent::reset()
{
    initializeParameters();
}
