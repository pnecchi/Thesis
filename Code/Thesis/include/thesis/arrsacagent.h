#ifndef ARRSACAGENT_H
#define ARRSACAGENT_H

#include <thesis/agent.h>
#include <thesis/stochasticactor.h>
#include <thesis/critic.h>
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
                    Critic const & criticU_);

        // Default destructor
        virtual ~ARRSACAgent();

    private:
        // Actor
        StochasticActor actor;

        // State-value function critic
        Critic criticV;

        // Square state-value function critic
        Critic criticU;

        // Average reward
        MovingAverage averageReward;

        // Average square reward
        MovingAverage averageSquareReward;

};



#endif // ARRSACAGENT_H
