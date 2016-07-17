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

#ifndef AGENT_H
#define AGENT_H

#include <armadillo>
#include <memory>

/*!
 * An Agent is an entity capable of producing actions based on previous
 * observations of the system. Generally it will interact with a task and will
 * be able to learn from experience, i.e. improve his performance while
 * interacting with the environment. The Agent abstract base class provides a
 * generic interface for an agent.
 */

class Agent
{
    public:
        //! Virtual destructor
        virtual ~Agent() = default;

        /*!
         * Clone method.
         * The class is polymorphically clonable, making it possible to write
         * copy constructors and assignment operators for classes that aggregate
         * objects of the Agent hierarchy by composition.
         * \return a unique pointer to a copy of the policy.
         */
        virtual std::unique_ptr<Agent> clone() const=0;

        /*!
         * Receive observation O_t of the system state from the task. This
         * observation will be typically cached and reused for the action
         * selection and learning steps.
         * \param observation system state observation.
         */
        virtual void receiveObservation(arma::vec const &observation_)=0;

        //! Get action size
        virtual size_t getDimAction() const=0;

        /*!
         * Get action A_t to be performed on the system. This action will be
         * passed to a task object, which manages the interaction between an
         * agent and an environment.
         * \return action action selected by the agent.
         */
        virtual arma::vec getAction()=0;

        /*!
         * Receive reward R_{t+1} from the system. This reward will be typically
         * cached and reused during the learning step.
         * \param reward reward received by the agent.
         */
        virtual void receiveReward(double reward_)=0;

        /*!
         * Receive observation O_{t+1} of the system state after the transition
         * induced by the action selected by the agent. This observation is
         * typically used in TD(lambda) learning methods for bootstrapping
         * purposes.
         * \param nextObservation_ observation of the new state of the system.
         */
        virtual void receiveNextObservation(arma::vec const &nextObservation_)=0;

        /*!
         * Learning step given previous experience. The agent modifies its
         * behavior to improve his performance on the task. This is the core of
         * an agent and of a reinforcement learning algorithm.
         */
        virtual void learn()=0;

        /*!
         * Tell the agent that a new learning epoch has started. This is
         * typically used to update the learning rates according to a predefined
         * schedule.
         */
        virtual void newEpoch()=0;

        /*!
         * Reset agent to its initial conditions. This is typically used to
         * reset the agent before a new independent learning experiment starts.
         */
        virtual void reset()=0;
};

#endif /* end of include guard: AGENT_H */
