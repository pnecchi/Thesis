#include "thesis/arrsacagent.h"
#include <math.h>  /* sqrt */

ARRSACAgent::ARRSACAgent(StochasticActor const & actor_,
                         Critic const & criticV_,
                         Critic const & criticU_,
                         double alphaBaseline_,
                         double alphaCritic_,
                         double alphaActor_)
    : actor(actor_),
      criticV(criticV_),
      criticU(criticU_),
      alphaBaseline(alphaBaseline_),
      alphaCritic(alphaCritic_),
      alphaActor(alphaActor_),
      averageReward(StatisticsEMA(alphaBaseline)),
      averageSquareReward(StatisticsEMA(alphaBaseline))
{
    //ctor
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
    arma::vec newParametersV = criticV.getParameters() +
                               alphaCritic * tdV * criticV.gradient(observation);
    arma::vec newParametersU = criticU.getParameters() +
                               alphaCritic * tdU * criticU.gradient(observation);
    criticV.setParameters(newParametersV);
    criticU.setParameters(newParametersU);

    // 4) Update actor
    double lambda = eta - rho * rho;
    double sqrtLambda = sqrt(lambda);
    double coeffGradientSR = (eta * tdV - 0.5 * rho * tdU) / (lambda * sqrtLambda);
    arma::vec newParametersActor = actor.getParameters() +
        alphaActor * coeffGradientSR * actor.likelihoodScore(observation, action);
    actor.setParameters(newParametersActor);
}
