#include "thesis/RiskSensitiveNpgpeAgent.h"
#include <math.h>  /* sqrt */

RiskSensitiveNPGPEAgent::RiskSensitiveNPGPEAgent
    (Policy const &policy_,
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
      rewardBaseline(0.02),
      squareRewardBaseline(0.02),
      gradientMean(policy_.getDimParameters(), arma::fill::zeros),
      gradientChol(policy_.getDimParameters(), policy_.getDimParameters(), arma::fill::zeros),
      lambda(lambda_),
      observation(policy_.getDimObservation()),
      action(policy_.getDimAction())
{
    initializeParameters();
}

RiskSensitiveNPGPEAgent::RiskSensitiveNPGPEAgent(RiskSensitiveNPGPEAgent const &other_)
    : policyPtr(other_.policyPtr->clone()),
      baselineLearningRatePtr(other_.baselineLearningRatePtr->clone()),
      hyperparamsLearningRatePtr(other_.hyperparamsLearningRatePtr->clone()),
      mean(other_.mean),
      choleskyFactor(other_.choleskyFactor),
      generator(other_.generator),
      gaussianDistr(other_.gaussianDistr),
      xi(other_.xi),
      rewardBaseline(other_.rewardBaseline),
      squareRewardBaseline(other_.squareRewardBaseline),
      gradientMean(other_.gradientMean),
      gradientChol(other_.gradientChol),
      lambda(other_.lambda),
      observation(other_.observation),
      action(other_.action)
{
    /* Nothing to do */
}

void RiskSensitiveNPGPEAgent::initializeParameters()
{
    mean.zeros();
    choleskyFactor.zeros();
    choleskyFactor.diag().ones();
}

std::unique_ptr<Agent> RiskSensitiveNPGPEAgent::clone() const
{
    return std::unique_ptr<Agent>(new RiskSensitiveNPGPEAgent(*this));
}

arma::vec RiskSensitiveNPGPEAgent::getAction()
{
    // Simulate policy parameters: w = mean + cholFactor * xi
    xi.imbue( [&]() { return gaussianDistr(generator); } );
    policyPtr->setParameters(mean + choleskyFactor * xi);

    // Select action
    return policyPtr->getAction(observation);
}

void RiskSensitiveNPGPEAgent::learn()
{
    // 1) Update baseline
    double alphaBaseline = baselineLearningRatePtr->get();
    rewardBaseline += alphaBaseline * (reward - rewardBaseline);
    squareRewardBaseline += alphaBaseline * (reward * reward - squareRewardBaseline);
    double var = squareRewardBaseline - rewardBaseline * rewardBaseline;
    double stddev = sqrt(var);

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

    arma::vec gradientRewardMean = (reward - rewardBaseline) * gradientMean;
    arma::vec gradientSquareRewardMean = (reward * reward - squareRewardBaseline) * gradientMean;
    arma::mat gradientRewardChol = (reward - rewardBaseline) * gradientChol;
    arma::mat gradientSquareRewardChol = (reward * reward - squareRewardBaseline) * gradientChol;

    arma::vec gradientSharpeMean =
        (squareRewardBaseline * gradientRewardMean -
        0.5 * rewardBaseline * gradientSquareRewardMean) / (var * stddev);
    arma::mat gradientSharpeChol =
        (squareRewardBaseline * gradientRewardChol -
        0.5 * rewardBaseline * gradientSquareRewardChol)  / (var * stddev);

    // 4) Update hyperparameters
    double alphaHyperparams = hyperparamsLearningRatePtr->get();
    mean += alphaHyperparams * gradientSharpeMean;
    choleskyFactor += alphaHyperparams * gradientSharpeChol;
}

void RiskSensitiveNPGPEAgent::newEpoch()
{
    // Update learning rate
    baselineLearningRatePtr->update();
    hyperparamsLearningRatePtr->update();
}

void RiskSensitiveNPGPEAgent::reset()
{
    // Reset deterministic policy
    policyPtr->reset();

    // Reset parameters
    initializeParameters();

    // Reset cache variables
    gradientMean.zeros();
    gradientChol.zeros();

    // Reset reward baseline
    rewardBaseline = 0.0;
    squareRewardBaseline = 0.0;

    // Reset learning rates
    baselineLearningRatePtr->reset();
    hyperparamsLearningRatePtr->reset();
}

