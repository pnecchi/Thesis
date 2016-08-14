#include "thesis/ArAgent.h"

ARAgent::ARAgent(StochasticActor const & actor_,
                 LearningRate const & baselineLearningRate_,
                 LearningRate const & actorLearningRate_,
                 double lambda_)
    : actor(actor_),
      averageReward(0.0),
      baselineLearningRatePtr(baselineLearningRate_.clone()),
      actorLearningRatePtr(actorLearningRate_.clone()),
      lambda(lambda_),
      gradientActor(actor.getDimParameters(), arma::fill::zeros),
      observation(actor_.getDimObservation()),
      action(actor_.getDimAction()),
      nextObservation(actor_.getDimObservation())
{
    /* Nothing to do */
}

ARAgent::ARAgent(ARAgent const & other_)
    : actor(other_.actor),
      averageReward(other_.averageReward),
      baselineLearningRatePtr(other_.baselineLearningRatePtr->clone()),
      actorLearningRatePtr(other_.actorLearningRatePtr->clone()),
      lambda(other_.lambda),
      gradientActor(other_.gradientActor),
      observation(other_.observation),
      action(other_.action),
      reward(other_.reward),
      nextObservation(other_.nextObservation)
{
    /* Nothing to do */
}

std::unique_ptr<Agent> ARAgent::clone() const
{
    return std::unique_ptr<Agent>(new ARAgent(*this));
}

void ARAgent::receiveObservation(arma::vec const &observation_)
{
    observation = observation_;
}

arma::vec ARAgent::getAction()
{
    action = actor.getAction(observation);
    return action;
}

void ARAgent::receiveReward(double reward_)
{
    reward = reward_;
}

void ARAgent::receiveNextObservation(arma::vec const &nextObservation_)
{
    nextObservation = nextObservation_;
}

void ARAgent::learn()
{
    // 1) Update baseline
    double alphaBaseline = baselineLearningRatePtr->get();
    averageReward += alphaBaseline * (reward - averageReward);

    // 2) Update actor
    double alphaActor = actorLearningRatePtr->get();
    gradientActor = lambda * gradientActor + actor.likelihoodScore(observation, action);
    gradientActor /= arma::norm(gradientActor, 2);
    actor.setParameters(actor.getParameters() + alphaActor * (reward - averageReward) * gradientActor);
}

void ARAgent::newEpoch()
{
    baselineLearningRatePtr->update();
    actorLearningRatePtr->update();
}

void ARAgent::reset()
{
    actor.reset();
    averageReward = 0.0;
    baselineLearningRatePtr->reset();
    actorLearningRatePtr->reset();
    gradientActor.zeros();
}

