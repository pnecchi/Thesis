/*
 * Copyright (c) 2016 Pierpaolo Necchi
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ARACAGENT_H
#define ARACAGENT_H

#include <thesis/Agent.h>            /* Agent */
#include <thesis/StochasticActor.h>  /* StochasticActor */
#include <thesis/Critic.h>           /* Critic */
#include <thesis/LearningRate.h>     /* LearningRate */
#include <armadillo>                 /* arma::vec */
#include <memory>                    /* std::unique_ptr */

/**
 * ARACAgent implements the Average Reward TD(lambda) Actor-Critic (ARAC)
 * learning algorithm, which is used for the online optimization of the average
 * reward obtaned by the learning agent. In particular, it is based on the
 * policy gradient theorem where a parametric critic is used to estimate the
 * state-value function and reduce the variance of the gradient estimate. For
 * more information on the algorithm, please refer to the report.
 */

class ARACAgent : public Agent
{
    public:
        /*!
         * Default constructor.
         * Initializes an ARACAgent object given a stochastic actor and a
         * critic, the three learning rates and the lambda parameter used in the
         * learning process.
         * \param actor_ stochatic actor.
         * \param critic_ critic.
         * \param baselineLearningRate_ learning rate used to update the average reward baseline.
         * \param criticLearningRate_ learning rate used to update the critic.
         * \param actorLearningRate_ learning rate used to update the actor.
         * \param lambda_ lambda factor for eligibility traces.
         */
        ARACAgent(StochasticActor const & actor_,
                  Critic const & critic_,
                  LearningRate const & baselineLearningRate_,
                  LearningRate const & criticLearningRate_,
                  LearningRate const & actorLearningRate_,
                  double lambda_=0.5);

        /*!
         * Copy constructor.
         * \param other_ ARACAgent object to copy.
         */
        ARACAgent(ARACAgent const & other_);

        //! Default destructor
        virtual ~ARACAgent() = default;

        /*!
         * Clone method.
         * The class is polymorphically clonable, making it possible to write
         * copy constructors and assignment operators for classes that aggregate
         * objects of the Agent hierarchy by composition.
         * \return a unique pointer to a copy of the policy.
         */
        virtual std::unique_ptr<Agent> clone() const;

        /*!
         * Receive observation O_t of the system state from the task. This
         * observation will be typically cached and reused for the action
         * selection and learning steps.
         * \param observation system state observation.
         */
        virtual void receiveObservation(arma::vec const &observation_);

        //! Get action size
        virtual size_t getDimAction() const { return actor.getDimAction(); }

        /*!
         * Get action A_t to be performed on the system. This action will be
         * passed to a task object, which manages the interaction between an
         * agent and an environment.
         * \return action action selected by the agent.
         */
        virtual arma::vec getAction();

        /*!
         * Receive reward R_{t+1} from the system. This reward will be typically
         * cached and reused during the learning step.
         * \param reward reward received by the agent.
         */
        virtual void receiveReward(double reward_);

        /*!
         * Receive observation O_{t+1} of the system state after the transition
         * induced by the action selected by the agent. This observation is
         * typically used in TD(lambda) learning methods for bootstrapping
         * purposes.
         * \param nextObservation_ observation of the new state of the system.
         */
        virtual void receiveNextObservation(arma::vec const &nextObservation_);

        /*!
         * Learning step given previous experience. The agent modifies its
         * behavior to improve his performance on the task. This is the core of
         * an agent and of a reinforcement learning algorithm.
         */
        virtual void learn();

        /*!
         * Tell the agent that a new learning epoch has started. This is
         * typically used to update the learning rates according to a predefined
         * schedule.
         */
        virtual void newEpoch();

        /*!
         * Reset agent to its initial conditions. This is typically used to
         * reset the agent before a new independent learning experiment starts.
         */
        virtual void reset();

    private:
        /*!
         * Average reward baseline. It simply consists of a moving average of
         * the past reward observed by the agent that is used to compute the TD
         * errors appearing in the critic update rule.
         */
        double averageReward;

        /*!
         * State-value function critic used to reduce the policy gradient
         * estimate variance.
         */
        Critic critic;

        /*!
         * Stochastic actor used for selecting actions.
         */
        StochasticActor actor;

        //! Learning rate used in the baseline update rule.
        std::unique_ptr<LearningRate> baselineLearningRatePtr;

        //! Learning rate used in the critic update rule.
        std::unique_ptr<LearningRate> criticLearningRatePtr;

        //! Learning rate used in the actor update rule.
        std::unique_ptr<LearningRate> actorLearningRatePtr;

        //! TD(lambda) parameter.
        double lambda;

        //! Cache vectors for the actor and the critic gradients.
        arma::vec gradientCritic;
        arma::vec gradientActor;

        //! Cache variables for observations, action and reward.
        arma::vec observation;
        arma::vec action;
        double reward;
        arma::vec nextObservation;
};

#endif // ARACAGENT_H
