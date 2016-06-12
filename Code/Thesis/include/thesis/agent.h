//----------------------------------------------------------------------
// Description: Agent abstract class
// Author:      Pierpaolo Necchi
// Email:       pierpaolo.necchi@gmail.com
// Date:        sab 11 giu 2016 12:51:09 CEST
//----------------------------------------------------------------------

#ifndef AGENT_H
#define AGENT_H

#include <armadillo>

/**
 * An Agent is an entity capable of producing actions based on previous
 * observations. Generally it will interact with a task and will be able to
 * learn from experience.
 */

class Agent
{
public:
    // Default constructor
	Agent(){}

	// Virtual destructor
	virtual ~Agent(){}

	// Clone method for virtual copy constructor
	virtual Agent* clone() const=0;

	// Receive observation of the system state
	virtual void receiveObservation(arma::vec const &observation_)=0;

	// Get action to be performed on the system
	virtual void getAction(arma::vec &action_)=0;

	// Receive reward from the system
	virtual void receiveReward(double reward_)=0;
};

#endif /* end of include guard: AGENT_H */
