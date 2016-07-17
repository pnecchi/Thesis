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

#ifndef RISKSENSITIVENPGPEAGENT_H
#define RISKSENSITIVENPGPEAGENT_H

#include <thesis/Agent.h>
#include <thesis/Policy.h>
#include <thesis/Statistics.h>
#include <thesis/LearningRate.h>
#include <memory>

/*!
 * RiskSensitiveNPGPEAgent implements a risk-sensitive natural PGPE agent based
 * on a deterministic controller and a gaussian probability distribution for the
 * controller parameters. For further information on Risk-Sensitive NPGPE, refer
 * to the thesis report.
 */

class RiskSensitiveNPGPEAgent : public Agent
{
    public:

        /*!
         * Constructor.
         * Initialize aRiskSensitiveNPGPEAgent given a deterministic policy.
         * \param policy_ deterministic controller.
         * \param learningRate learning rate object.
         * \param discount_ discount factor
         */
        RiskSensitiveNPGPEAgent(Policy const &policy_,
                                LearningRate const &baselineLearningRate_,
                                LearningRate const &hyperparamsLearningRate_,
                                double lambda_);

        /*!
         * Copy constructor.
         * \param other_ RiskSensitiveNPGPEAgent to copy.
         */
        RiskSensitiveNPGPEAgent(RiskSensitiveNPGPEAgent const &other_);

        //! Default destructor.
        virtual ~RiskSensitiveNPGPEAgent() = default;

        //! Clone method for virtual copy constructor
        virtual std::unique_ptr<Agent> clone() const;

        //! Get action size
        virtual size_t getDimAction() const { return policyPtr->getDimAction(); }

        /*!
         * Receive observation O_t of the system state from the task. This
         * observation will be typically cached and reused for the action
         * selection and learning steps.
         * \param observation system state observation.
         */
        virtual void receiveObservation(arma::vec const &observation_)
            { observation = observation_; }

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
        virtual void receiveReward(double reward_) { reward = reward_; }

        /*!
         * Receive observation O_{t+1} of the system state after the transition
         * induced by the action selected by the agent. NPGPE does not use this
         * observation in the learning step.
         * \param nextObservation_ observation of the new state of the system.
         */
        virtual void receiveNextObservation(arma::vec const &nextObservation_) {}

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
         * Initialize the NPGPE agent parameters, i.e. the mean and the cholesky
         * factor of the covariance matrix of the Gaussian parameter
         * distribution.
         */

        void initializeParameters();

        /*!
         * Deterministic controller.
         * A deterministic mapping from a state observation to an action.
         */
        std::unique_ptr<Policy> policyPtr;

        //! Random number generator.
        mutable std::mt19937 generator;

        /*!
         * Parameter distribution. The controller parameters are sampled from a
         * multi-variate Gaussian distribution, parametrized by the mean and the
         * Cholesky factor of the covariance matrix.
         */
        mutable std::normal_distribution<double> gaussianDistr;

        // Cache variable for the parameter simulation used in the learning.
        arma::vec xi;

        //! Parameter distribution hyperparameters
        arma::vec mean;
        arma::mat choleskyFactor;  // Sigma = choleskyFactor * choleskyFactor'

        /*!
         * Average reward baseline, i.e. a moving average of the past reward
         * observed by the agent.
         */
        double rewardBaseline;

        /*!
         * Average reward baseline, i.e. a moving average of the past reward
         * observed by the agent.
         */
        double squareRewardBaseline;

        //! Gradient cache
        arma::vec gradientMean;
        arma::mat gradientChol;

        //! Learning rate for the baseline
        std::unique_ptr<LearningRate> baselineLearningRatePtr;

        //! Learning rate for the hyperparameters
        std::unique_ptr<LearningRate> hyperparamsLearningRatePtr;

        //! Lambda parameter for gradient compuation.
        double lambda;

        // Cache variables
        arma::vec observation;
        arma::vec action;
        double reward;
};

#endif // RISKSENSITIVENPGPEAGENT_H
