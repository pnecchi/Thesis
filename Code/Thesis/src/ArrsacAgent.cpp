#include "thesis/ArrsacAgent.h"
#include <math.h>  /* sqrt */
#include <iostream>

ARRSACAgent::ARRSACAgent(StochasticActor const & actor_,
                         Critic const & criticV_,
                         Critic const & criticU_,
                         double lambda_,
                         double alphaActor_,
                         double alphaCritic_,
                         double alphaBaseline_)
    : actor(actor_),
      criticV(criticV_),
      criticU(criticU_),
      lambda(lambda_),
      alphaActor(alphaActor_),
      alphaCritic(alphaCritic_),
      alphaBaseline(alphaBaseline_),
      averageReward(alphaBaseline_),
      averageSquareReward(alphaBaseline_),
      gradientCriticV(criticV.getDimParameters(), arma::fill::zeros),
      gradientCriticU(criticU.getDimParameters(), arma::fill::zeros),
      gradientActor(actor.getDimParameters(), arma::fill::zeros),
      observation(actor_.getDimObservation()),
      action(actor_.getDimAction()),
      nextObservation(actor_.getDimObservation())
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
}

void ARRSACAgent::receiveNextObservation(arma::vec const &nextObservation_)
{
    nextObservation = nextObservation_;
}

void ARRSACAgent::learn()
{
    // 1) Update baselines
    rewardSquared = reward * reward;
    averageReward.dumpOneResult(reward);
    averageSquareReward.dumpOneResult(rewardSquared);
    double rho = averageReward.getStatistics()[0][0];
    double eta = averageSquareReward.getStatistics()[0][0];

    // 2) Compute TD errors
    double tdV = reward - rho + criticV.evaluate(nextObservation) -
                 criticV.evaluate(observation);
    double tdU = rewardSquared - eta + criticU.evaluate(nextObservation) -
                 criticU.evaluate(observation);

    // 3) Update critics
    gradientCriticV = lambda * gradientCriticV + criticV.gradient(observation);
    gradientCriticU = lambda * gradientCriticU + criticU.gradient(observation);
    criticV.setParameters(criticV.getParameters() + alphaCritic * tdV * gradientCriticV);
    criticU.setParameters(criticU.getParameters() + alphaCritic * tdU * gradientCriticU);

    // 4) Update actor
    double var = eta - rho * rho;
    double sqrtVar = sqrt(var);
    double coeffGradientSR = (eta * tdV - 0.5 * rho * tdU) / (var * sqrtVar);
    gradientActor = lambda * gradientActor + actor.likelihoodScore(observation, action);
    actor.setParameters(actor.getParameters() + alphaActor * coeffGradientSR * gradientActor);
}

void ARRSACAgent::reset()
{
    actor.reset();
    criticV.reset();
    criticU.reset();
    averageReward.reset();
    averageSquareReward.reset();
}

