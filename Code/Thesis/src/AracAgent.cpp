#include "thesis/AracAgent.h"
#include <math.h>  /* sqrt */
#include <iostream>

ARACAgent::ARACAgent(StochasticActor const & actor_,
                     Critic const & critic_,
                     double lambda_,
                     double alphaActor_,
                     double alphaCritic_,
                     double alphaBaseline_)
    : actor(actor_),
      critic(critic_),
      alphaActor(alphaActor_),
      alphaCritic(alphaCritic_),
      alphaBaseline(alphaBaseline_),
      averageReward(alphaBaseline),
      lambda(lambda_),
      gradientCritic(critic.getDimParameters(), arma::fill::zeros),
      gradientActor(actor.getDimParameters(), arma::fill::zeros),
      observation(actor_.getDimObservation()),
      action(actor_.getDimAction()),
      nextObservation(actor_.getDimObservation())
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
    averageReward.dumpOneResult(reward);
    double rho = averageReward.getStatistics()[0][0];

    // 1) Compute TD errors
    double tdErr = reward - rho + critic.evaluate(nextObservation) -
                   critic.evaluate(observation);

    // 3) Update critics
    gradientCritic = lambda * gradientCritic + critic.gradient(observation);
    critic.setParameters(critic.getParameters() + alphaCritic * tdErr * gradientCritic);

    // 4) Update actor
    gradientActor = lambda * gradientActor + actor.likelihoodScore(observation, action);
    actor.setParameters(actor.getParameters() + alphaActor * tdErr * gradientActor);
}

void ARACAgent::reset()
{
    actor.reset();
    critic.reset();
    averageReward.reset();
}

