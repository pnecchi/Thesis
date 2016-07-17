#include "thesis/NpgpeAgent.h"

NPGPEAgent::NPGPEAgent(Policy const &policy_,
                       LearningRate const &baselineLearningRate_,
                       LearningRate const &hyperparamsLearningRate_,
                       double lambda_)
    : policyPtr(policy_.clone()),
      baselineLearningRatePtr(baselineLearningRate_.clone()),
      hyperparamsLearningRatePtr(hyperparamsLearningRate_.clone()),
      mean(policy_.getDimParameters(), arma::fill::zeros),
      choleskyFactor(policy_.getDimParameters(), policy_.getDimParameters(), arma::fill::zeros),
      generator(215),
      gaussianDistr(0.0, 1.0),
      xi(policy_.getDimParameters()),
      baseline(0.0),
      gradientMean(policy_.getDimParameters(), arma::fill::zeros),
      gradientChol(policy_.getDimParameters(), policy_.getDimParameters(), arma::fill::zeros),
      lambda(lambda_),
      observation(policy_.getDimObservation()),
      action(policy_.getDimAction())
{
    initializeParameters();
}

NPGPEAgent::NPGPEAgent(NPGPEAgent const &other_)
    : policyPtr(other_.policyPtr->clone()),
      baselineLearningRatePtr(other_.baselineLearningRatePtr->clone()),
      hyperparamsLearningRatePtr(other_.hyperparamsLearningRatePtr->clone()),
      mean(other_.mean),
      choleskyFactor(other_.choleskyFactor),
      generator(other_.generator),
      gaussianDistr(other_.gaussianDistr),
      xi(other_.xi),
      baseline(other_.baseline),
      gradientMean(other_.gradientMean),
      gradientChol(other_.gradientChol),
      lambda(other_.lambda),
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
    double alphaBaseline = baselineLearningRatePtr->get();
    baseline += alphaBaseline * (reward - baseline);

    // 2) Compute likelihood score
    arma::vec likelihoodMean = policyPtr->getParameters() - mean;
    arma::mat likelihoodChol =
        (arma::trimatu(xi * xi.t()) -
        0.5 * arma::diagmat(xi * xi.t()) -
        0.5 * arma::eye(policyPtr->getDimParameters(), policyPtr->getDimParameters())) *
        choleskyFactor.t();

    // 3) Update gradients
    gradientMean = lambda * gradientMean + likelihoodMean;
    gradientChol = lambda * gradientChol + likelihoodChol;

    // 4) Update hyperparameters
    double alphaHyperparams = hyperparamsLearningRatePtr->get();
    mean += alphaHyperparams * (reward - baseline) * gradientMean;
    choleskyFactor += alphaHyperparams * (reward - baseline) * gradientChol;
}

void NPGPEAgent::newEpoch()
{
    // Update learning rate
    baselineLearningRatePtr->update();
    hyperparamsLearningRatePtr->update();

    // TODO: print current parameters and gradient norm to file for debug purposess
}

void NPGPEAgent::reset()
{
    // Reset deterministic policy
    policyPtr->reset();

    // Reset parameters
    initializeParameters();

    // Reset cache variables
    gradientMean.zeros();
    gradientChol.zeros();

    // Reset reward baseline
    baseline = 0.0;

    // Reset learning rate
    baselineLearningRatePtr->reset();
    hyperparamsLearningRatePtr->reset();
}
