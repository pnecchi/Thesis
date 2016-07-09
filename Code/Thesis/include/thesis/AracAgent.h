#ifndef ARACAGENT_H
#define ARACAGENT_H

#include <thesis/Agent.h>
#include <thesis/StochasticActor.h>
#include <thesis/Critic.h>
#include <thesis/Statistics.h>
#include <armadillo>
#include <memory>

/**
 * ARACAgent implements the Average Reward TD(0) Actor-Critic agent. This agent
 * is used for the online optimization of the Sharpe-Ratio of rewards.
 */

class ARACAgent : public Agent
{
    public:
        // Default constructor
        ARACAgent(StochasticActor const & actor_,
                  Critic const & critic_,
                  double lambda_=0.5,
                  double alphaActor_=0.001,
                  double alphaCritic_=0.01,
                  double alphaBaseline_=0.1);

        // Default destructor
        virtual ~ARACAgent() = default;

        // Clone method for virtual copy constructor
        virtual std::unique_ptr<Agent> clone() const;

        // Receive observation of the system state --> O_t
        virtual void receiveObservation(arma::vec const &observation_);

        // Get action size
        virtual size_t getDimAction() const { return actor.getDimAction(); }

        // Get action to be performed on the system --> A_t
        virtual arma::vec getAction();

        // Receive reward from the system --> R_{t+1}
        virtual void receiveReward(double reward_);

        // Receive next observation --> O_{t+1}
        virtual void receiveNextObservation(arma::vec const &nextObservation_);

        // Learning step given previous experience
        virtual void learn();

        // New epoch
        virtual void newEpoch();

        // Reset
        virtual void reset();

    private:
        // Actor
        StochasticActor actor;

        // State-value function critic
        Critic critic;

        // Average reward (Exponential Moving Average)
        StatisticsEMA averageReward;

        // Learning rates
        // TODO: implement time-varying learning rates?
        double alphaCritic;
        double alphaActor;
        double alphaBaseline;

        // Gradient cache vectors
        double lambda;
        arma::vec gradientCritic;
        arma::vec gradientActor;

        // Cache variables
        arma::vec observation;
        arma::vec action;
        double reward;
        arma::vec nextObservation;
};


#endif // ARACAGENT_H
