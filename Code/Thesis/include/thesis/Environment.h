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

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <armadillo>
#include <memory>

/** The general interface for an environment in which an agents acts and learns.
 * The environment is characterized by a state that can be (partially) accessed
 * by the agent and used to select actions. The Environment corresponds to the
 * system in an MDP.
 */

class Environment
{
    public:
        //! Default constructor.
        Environment() = default;

        //! Default copy constructor.
        Environment(Environment const &environment_) = default;

        //! Virtual destructor.
        virtual ~Environment() = default;

        //! Clone method for polymorphic composition.
        virtual std::unique_ptr<Environment> clone() const = 0;

        /**
         * Get system state.
         * \return system state.
         */
        virtual arma::vec getState() const = 0;

        /**
         * Perform action on the system.
         * \param action portfolio allocation
         */
        virtual void performAction(arma::vec const &action) = 0;

        /**
         * Get dimension of the state space.
         * \return dimension of the state space.
         */
        virtual size_t getDimState() const = 0;

        /**
         * Get dimension of the action space.
         * \return dimension of the action space.
         */
        virtual size_t getDimAction() const = 0;

        /**
         * Reset environment to initial condition. This method is used in
         * episodic tasks to reset the environment when a terminal state is
         * reached.
         */
        virtual void reset() = 0;
};

#endif // ENVIRONMENT_H
