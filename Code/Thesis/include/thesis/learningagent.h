//----------------------------------------------------------------------
// Description: Learning agent abstract class
// Author:      Pierpaolo Necchi
// Email:       pierpaolo.necchi@gmail.com
// Date:        sab 11 giu 2016 13:29:19 CEST
//----------------------------------------------------------------------

#ifndef LEARNINGAGENT_H
#define LEARNINGAGENT_H

#include <armadillo>
#include <memory>
#include <thesis/agent.h>
#include <thesis/actor.h>


class LearningAgent : public Agent
{
public:
    // Standard constructor
	LearningAgent(std::unique_ptr<Actor> const &actorPtr_);

    // Standard destructor
	virtual ~LearningAgent() = default;

	// Clone method for virtual copy constructor
    virtual LearningAgent* clone() const;

    // Receive observation from the system
	virtual void receiveObservation (arma::vec const &observation_);

	// Get action to perform on the environment
	virtual void getAction (arma::vec &action_);

    // Receive reward from the environment
	virtual void receiveReward (double reward_);

    // Learning step given previous experience
	virtual void learn()=0;

private:
    // Actor (wrapped)
    std::unique_ptr<Actor> actorPtr;

    // Cache variables
	arma::vec lastObservation;
	arma::vec lastAction;
	double lastReward;
};

#endif /* end of include guard: LEARNINGAGENT_H */
