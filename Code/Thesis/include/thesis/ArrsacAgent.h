#ifndef ARRSACAGENT_H
#define ARRSACAGENT_H

#include <thesis/Agent.h>
#include <thesis/StochasticActor.h>
#include <thesis/Critic.h>
#include <thesis/Statistics.h>
#include <armadillo>
#include <memory>

/**
 * ARRSACAgent implements the Average Reward Risk-Sensitive Actor-Critic agent
 * proposed in Prashanth, L A and Ghavamzadeh, M. - "Variance-Constrained Actor-
 * Critic Algorithms for discounted and average reward MDPs" (2015). This agent
 * is used for the online optimization of the Sharpe-Ratio of rewards.
 */

// TODO: Implement also mean-variance optimization criterion. use templatization?

class ARRSACAgent : public Agent
{
    public:
        // Default constructor
        ARRSACAgent(StochasticActor const & actor_,
                    Critic const & criticV_,
                    Critic const & criticU_,
                    double alphaActor_=0.01,
                    double alphaCritic_=0.05,
                    double alphaBaseline_=0.1,
                    double benchmark_=0.0);

        // Default destructor
        virtual ~ARRSACAgent() = default;

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

        // Reset
        virtual void reset();

    private:
        // Actor
        StochasticActor actor;

        // State-value function critic
        Critic criticV;

        // Square state-value function critic
        Critic criticU;

        // Average reward (Exponential Moving Average)
        StatisticsEMA averageReward;

        // Average square reward (Exponential Moving Average)
        StatisticsEMA averageSquareReward;

        // Learning rates
        // TODO: implement time-varying learning rates?
        double alphaBaseline;
        double alphaCritic;
        double alphaActor;

        // Benchmark for Sharpe ratio
        double benchmark;

        // Cache variables
        arma::vec observation;
        arma::vec action;
        double reward;
        double rewardSquared;
        arma::vec nextObservation;
};



#endif // ARRSACAGENT_H
