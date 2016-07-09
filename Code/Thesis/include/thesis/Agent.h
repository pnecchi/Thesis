#ifndef AGENT_H
#define AGENT_H

#include <armadillo>
#include <memory>

/**
 * An Agent is an entity capable of producing actions based on previous
 * observations. Generally it will interact with a task and will be able to
 * learn from experience.
 */

class Agent
{
    public:
        // Virtual destructor
        virtual ~Agent() = default;

        // Clone method for virtual copy constructor
        virtual std::unique_ptr<Agent> clone() const=0;

        // Receive observation of the system state --> O_t
        virtual void receiveObservation(arma::vec const &observation_)=0;

        // Get action size
        virtual size_t getDimAction() const=0;

        // Get action to be performed on the system --> A_t
        virtual arma::vec getAction()=0;

        // Receive reward from the system --> R_{t+1}
        virtual void receiveReward(double reward_)=0;

        // Receive next observation --> O_{t+1}
        virtual void receiveNextObservation(arma::vec const &nextObservation_)=0;

        // Learning step given previous experience
        virtual void learn()=0;

        // Reset agent
        virtual void reset()=0;
};

#endif /* end of include guard: AGENT_H */
