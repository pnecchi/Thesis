#ifndef ARRSACAGENT_H
#define ARRSACAGENT_H

#include <thesis/Agent.h>
#include <thesis/StochasticActor.h>
#include <thesis/Critic.h>
#include <thesis/LearningRate.h>
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
                    LearningRate const & baselineLearningRate_,
                    LearningRate const & criticLearningRate_,
                    LearningRate const & actorLearningRate_,
                    double lambda_=0.5);

        // Copy constructor
        ARRSACAgent(ARRSACAgent const &other_);

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

        // New epoch
        virtual void newEpoch();

        // Reset
        virtual void reset();

    private:
        // Average reward (Exponential Moving Average)
        double averageReward;

        // Average square reward (Exponential Moving Average)
        double averageSquareReward;

        // State-value function critic
        Critic criticV;

        // Square state-value function critic
        Critic criticU;

        // Actor
        StochasticActor actor;

        // Learning rates
        std::unique_ptr<LearningRate> baselineLearningRatePtr;
        std::unique_ptr<LearningRate> criticLearningRatePtr;
        std::unique_ptr<LearningRate> actorLearningRatePtr;

        // Gradient cache vectors
        double lambda;
        arma::vec gradientCriticV;
        arma::vec gradientCriticU;
        arma::vec gradientActor;

        // Cache variables
        arma::vec observation;
        arma::vec action;
        double reward;
        double rewardSquared;
        arma::vec nextObservation;
};



#endif // ARRSACAGENT_H
