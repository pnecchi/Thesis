#include "thesis/AracAgent.h"
#include <math.h>  /* sqrt */
#include <iostream>

ARACAgent::ARACAgent(StochasticActor const & actor_,
                     Critic const & critic_,
                     LearningRate const & baselineLearningRate_,
                     LearningRate const & criticLearningRate_,
                     LearningRate const & actorLearningRate_,
                     double lambda_)
    : actor(actor_),
      critic(critic_),
      averageReward(0.0),
      baselineLearningRatePtr(baselineLearningRate_.clone()),
      criticLearningRatePtr(criticLearningRate_.clone()),
      actorLearningRatePtr(actorLearningRate_.clone()),
      lambda(lambda_),
      gradientCritic(critic.getDimParameters(), arma::fill::zeros),
      gradientActor(actor.getDimParameters(), arma::fill::zeros),
      observation(actor_.getDimObservation()),
      action(actor_.getDimAction()),
      nextObservation(actor_.getDimObservation())
{
    /* Nothing to do */
}

ARACAgent::ARACAgent(ARACAgent const & other_)
    : actor(other_.actor),
      critic(other_.critic),
      averageReward(other_.averageReward),
      baselineLearningRatePtr(other_.baselineLearningRatePtr->clone()),
      criticLearningRatePtr(other_.criticLearningRatePtr->clone()),
      actorLearningRatePtr(other_.actorLearningRatePtr->clone()),
      lambda(other_.lambda),
      gradientActor(other_.gradientActor),
      gradientCritic(other_.gradientCritic),
      observation(other_.observation),
      action(other_.action),
      reward(other_.reward),
      nextObservation(other_.nextObservation)
{
    /* Nothing to do */
}

std::unique_ptr<Agent> ARACAgent::clone() const
{
    return std::unique_ptr<Agent>(new ARACAgent(*this));
}

void ARACAgent::receiveObservation(arma::vec const &observation_)
{
    observation = observation_;
}

arma::vec ARACAgent::getAction()
{
    action = actor.getAction(observation);
    return action;
}

void ARACAgent::receiveReward(double reward_)
{
    reward = reward_;
}

void ARACAgent::receiveNextObservation(arma::vec const &nextObservation_)
{
    nextObservation = nextObservation_;
}

void ARACAgent::learn()
{
    // 1) Update baseline
    double alphaBaseline = baselineLearningRatePtr->get();
    averageReward += alphaBaseline * (reward - averageReward);

    // 2) Compute TD errors
    double tdErr = reward - averageReward +
                   critic.evaluate(nextObservation) -
                   critic.evaluate(observation);

    // 3) Update critics
    double alphaCritic = criticLearningRatePtr->get();
    gradientCritic = lambda * gradientCritic + critic.gradient(observation);
    gradientCritic /= arma::norm(gradientCritic, 2);
    critic.setParameters(critic.getParameters() + alphaCritic * tdErr * gradientCritic);

    // 4) Update actor
    double alphaActor = actorLearningRatePtr->get();
    gradientActor = lambda * gradientActor + actor.likelihoodScore(observation, action);
    gradientActor /= arma::norm(gradientActor, 2);
    actor.setParameters(actor.getParameters() + alphaActor * tdErr * gradientActor);
}

void ARACAgent::newEpoch()
{
    baselineLearningRatePtr->update();
    criticLearningRatePtr->update();
    actorLearningRatePtr->update();
}

void ARACAgent::reset()
{
    actor.reset();
    critic.reset();
    averageReward = 0.0;
    baselineLearningRatePtr->reset();
    criticLearningRatePtr->reset();
    actorLearningRatePtr->reset();
    gradientCritic.zeros();
    gradientActor.zeros();
}

