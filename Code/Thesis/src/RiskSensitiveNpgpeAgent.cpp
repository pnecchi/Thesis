#include "thesis/RiskSensitiveNpgpeAgent.h"
#include <math.h>  /* sqrt */

RiskSensitiveNPGPEAgent::RiskSensitiveNPGPEAgent(Policy const &policy_,
                                                 LearningRate const &learningRate_,
                                                 double discountFactor_)
    : policyPtr(policy_.clone()),
      learningRatePtr(learningRate_.clone()),
      mean(policy_.getDimParameters(), arma::fill::zeros),
      choleskyFactor(policy_.getDimParameters(), policy_.getDimParameters(), arma::fill::zeros),
      generator(215),
      gaussianDistr(0.0, 1.0),
      xi(policy_.getDimParameters()),
      rewardBaseline(0.02),
      squareRewardBaseline(0.02),
      gradientMean(policy_.getDimParameters(), arma::fill::zeros),
      gradientChol(policy_.getDimParameters(), policy_.getDimParameters(), arma::fill::zeros),
      discountFactor(discountFactor_),
      observation(policy_.getDimObservation()),
      action(policy_.getDimAction())
{
    initializeParameters();
}

RiskSensitiveNPGPEAgent::RiskSensitiveNPGPEAgent(RiskSensitiveNPGPEAgent const &other_)
    : policyPtr(other_.policyPtr->clone()),
      learningRatePtr(other_.learningRatePtr->clone()),
      mean(other_.mean),
      choleskyFactor(other_.choleskyFactor),
      generator(other_.generator),
      gaussianDistr(other_.gaussianDistr),
      xi(other_.xi),
      rewardBaseline(other_.rewardBaseline),
      squareRewardBaseline(other_.squareRewardBaseline),
      gradientMean(other_.gradientMean),
      gradientChol(other_.gradientChol),
      discountFactor(other_.discountFactor),
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
    rewardBaseline.dumpOneResult(reward);
    squareRewardBaseline.dumpOneResult(reward * reward);
    double rb = rewardBaseline.getStatistics()[0][0];
    double r2b = squareRewardBaseline.getStatistics()[0][0];
    double var = r2b - rb * rb;
    double stddev = sqrt(var);

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

    arma::vec gradientRewardMean = (reward - rb) * gradientMean;
    arma::vec gradientSquareRewardMean = (reward * reward - r2b) * gradientMean;
    arma::mat gradientRewardChol = (reward - rb) * gradientChol;
    arma::mat gradientSquareRewardChol = (reward * reward - r2b) * gradientChol;

    arma::vec gradientSharpeMean =
        (r2b * gradientRewardMean - 0.5 * rb * gradientSquareRewardMean) /
        (var * stddev);
    arma::mat gradientSharpeChol =
        (r2b * gradientRewardChol - 0.5 * rb * gradientSquareRewardChol)  /
        (var * stddev);

    // 4) Update hyperparameters
    double alpha = learningRatePtr->get();
    mean += alpha * gradientSharpeMean;
    choleskyFactor += alpha * gradientSharpeChol;
}

void RiskSensitiveNPGPEAgent::newEpoch()
{
    // Update learning rate
    learningRatePtr->update();
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
    rewardBaseline.reset();
    squareRewardBaseline.reset();

    // Reset learning rate
    learningRatePtr->reset();
}

