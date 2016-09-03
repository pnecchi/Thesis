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

#ifndef TASK_H
#define TASK_H

#include <thesis/Environment.h>     /* Environment */
#include <armadillo>                /* arma::vec */
#include <memory>                   /* std::unique_ptr */

/**
 * Generic interface for a reinforcement learning task. The Task specifies what
 * the goal is in an Environment and how the agent is rewarded for its actions.
 * Hence, the composition of an Environment and a Task fully defines the MDP.
 */

class Task
{
    public:
        /**
         * Constructor.
         * Initialize a task given the underlying environment.
         * @param environment on which the task is defined.
         */
        Task(Environment const &environment_);

        //! Copy constructor.
        Task(Task const &task_);

        //! Virtual destructor.
        virtual ~Task() = default;

        //! Clone method for polymorphic composition.
        virtual std::unique_ptr<Task> clone() const = 0;

        /**
         * Get dimension of the observation space.
         * \return dimension of the observation space.
         */
        virtual size_t getDimObservation () const = 0;

        /**
         * Get dimension of the action space.
         * \return dimension of the action space.
         */
        size_t getDimAction () const { return environmentPtr->getDimAction(); }

        /**
         * Provide an observation of the state. This mechanism can be used to
         * implement a partially observable MDP (POMDP) or augment the state of
         * the system.
         * \return observation of the state.
         */
        virtual arma::vec getObservation () const = 0;

        /**
         * Perform action.
         * Select new portfolio allocation on the risky assets. It is assumed
         * that the portfolio weight on the risk-free asset is 1 - sum(u_i).
         * \param action portfolio allocation.
         */
        virtual void performAction (arma::vec const &action) = 0;

        /**
         * Provide reward.
         * The agent receives the log-return of his portfolio as a feedback.
         * \return portfolio log-return
         */
        virtual double getReward () const = 0;

        /**
         * Reset task to the initial conditions. This method is used in
         * episodic tasks to reset the environment when a terminal state is
         * reached. It is also used when a new learning epochs begins.
         */
        virtual void reset() = 0;

    protected:

        /**
         * Environment on which the task is defined.
         * Need a unique_ptr to implement polymorphic composition.
         */
        std::unique_ptr<Environment> environmentPtr;

};

#endif // TASK_H
