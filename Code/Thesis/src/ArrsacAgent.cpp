#include "thesis/ArrsacAgent.h"
#include <math.h>  /* sqrt */
#include <iostream>

ARRSACAgent::ARRSACAgent(StochasticActor const & actor_,
                         Critic const & criticV_,
                         Critic const & criticU_,
                         LearningRate const & baselineLearningRate_,
                         LearningRate const & criticLearningRate_,
                         LearningRate const & actorLearningRate_,
                         double lambda_)
    : averageReward(0.0),
      averageSquareReward(0.0),
      criticV(criticV_),
      criticU(criticU_),
      actor(actor_),
      baselineLearningRatePtr(baselineLearningRate_.clone()),
      criticLearningRatePtr(criticLearningRate_.clone()),
      actorLearningRatePtr(actorLearningRate_.clone()),
      lambda(lambda_),
      gradientCriticV(criticV.getDimParameters(), arma::fill::zeros),
      gradientCriticU(criticU.getDimParameters(), arma::fill::zeros),
      gradientActor(actor.getDimParameters(), arma::fill::zeros),
      gradientSharpe(actor.getDimParameters(), arma::fill::zeros),
      observation(actor_.getDimObservation()),
      action(actor_.getDimAction()),
      nextObservation(actor_.getDimObservation())
{
    /* Nothing to do */
}

ARRSACAgent::ARRSACAgent(ARRSACAgent const &other_)
    : averageReward(other_.averageReward),
      averageSquareReward(other_.averageSquareReward),
      criticV(other_.criticV),
      criticU(other_.criticU),
      actor(other_.actor),
      baselineLearningRatePtr(other_.baselineLearningRatePtr->clone()),
      criticLearningRatePtr(other_.criticLearningRatePtr->clone()),
      actorLearningRatePtr(other_.actorLearningRatePtr->clone()),
      lambda(other_.lambda),
      gradientCriticV(other_.gradientCriticV),
      gradientCriticU(other_.gradientCriticU),
      gradientActor(other_.gradientActor),
      gradientSharpe(other_.gradientSharpe),
      observation(other_.observation),
      action(other_.action),
      nextObservation(other_.nextObservation),
      reward(other_.reward),
      rewardSquared(other_.rewardSquared)
{
    /* Nothing to do */
}


std::unique_ptr<Agent> ARRSACAgent::clone() const
{
    return std::unique_ptr<Agent>(new ARRSACAgent(*this));
}

void ARRSACAgent::receiveObservation(arma::vec const &observation_)
{
    observation = observation_;
}

arma::vec ARRSACAgent::getAction()
{
    action = actor.getAction(observation);
    return action;
}

void ARRSACAgent::receiveReward(double reward_)
{
    reward = reward_;
    rewardSquared = reward_ * reward_;
}

void ARRSACAgent::receiveNextObservation(arma::vec const &nextObservation_)
{
    nextObservation = nextObservation_;
}

void ARRSACAgent::learn()
{
    // 1) Update baselines
    double alphaBaseline = baselineLearningRatePtr->get();
    averageReward += alphaBaseline * (reward - averageReward);
    averageSquareReward += alphaBaseline * (rewardSquared - averageSquareReward);

    // 2) Compute TD errors
//    double tdV = reward - averageReward +
//                 criticV.evaluate(nextObservation) -
//                 criticV.evaluate(observation);
//    double tdU = rewardSquared - averageSquareReward +
//                 criticU.evaluate(nextObservation) -
//                 criticU.evaluate(observation);

    // 3) Update critics
//    double alphaCritic = criticLearningRatePtr->get();
//    gradientCriticV = criticV.gradient(observation);
//    gradientCriticU = criticU.gradient(observation);
//    criticV.setParameters(criticV.getParameters() + alphaCritic * tdV * gradientCriticV);
//    criticU.setParameters(criticU.getParameters() + alphaCritic * tdU * gradientCriticU);

    // 4) Update actor
    double alphaActor = actorLearningRatePtr->get();
    double var = averageSquareReward - averageReward * averageReward;
    double sqrtVar = sqrt(var);
//    double coeffGradientSR = (averageSquareReward * tdV - 0.5 * averageReward * tdU) /
//                             (var * sqrtVar);


    gradientActor = actor.likelihoodScore(observation, action);
    double coeffGradientSR = (averageSquareReward * (reward - averageReward) - 0.5 * averageReward * (rewardSquared - averageSquareReward)) /
                             (var * sqrtVar);

    gradientSharpe = lambda * gradientSharpe + coeffGradientSR * gradientActor;
    // gradientSharpe /= arma::norm(gradientSharpe, 2);
    actor.setParameters(actor.getParameters() + alphaActor * gradientSharpe);
}

void ARRSACAgent::newEpoch()
{
    baselineLearningRatePtr->update();
    criticLearningRatePtr->update();
    actorLearningRatePtr->update();
}

void ARRSACAgent::reset()
{
    averageReward = 0.0;
    averageSquareReward = 0.0;
    criticV.reset();
    criticU.reset();
    actor.reset();
    baselineLearningRatePtr->reset();
    criticLearningRatePtr->reset();
    actorLearningRatePtr->reset();
    gradientActor.zeros();
    gradientCriticU.zeros();
    gradientCriticV.zeros();
}

