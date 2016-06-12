#include <thesis/learningagent.h>
#include <armadillo>

LearningAgent::LearningAgent(std::unique_ptr<Actor> const &actor_)
	: actorPtr(actorPtr_),
	  lastObservation(arma::vec(actor.dimInput)),
	  lastAction(arma::vec(actor.dimOutput)),
	  reward(0.0)
{
	/* Nothing to do */
}

LearningAgent::LearningAgent* clone() const
{
    return new LearningAgent(*this);
}

void LearningAgent::receiveObservation (arma::vec const &observation_)
{
	observation = observation_;
}

void LearningAgent::getAction(arma::vec &action_)
{
	// TODO: check with actor implementation
	action_ = actorPtr->selectAction(lastObservation);
	action = action_;
}

void LearningAgent::receiveReward (double reward_)
{
	reward = reward_;
}
