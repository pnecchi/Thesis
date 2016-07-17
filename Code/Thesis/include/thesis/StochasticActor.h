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

#ifndef STOCHASTICACTOR_H
#define STOCHASTICACTOR_H

#include <armadillo>
#include <memory>
#include <thesis/Actor.h>
#include <thesis/StochasticPolicy.h>

/**
 * A StochasticActor is the stochastic policy employed by an Agent for selecting
 * an action given an observation of the system. The class is based on the
 * StochasticPolicy hierarchy by virtual composition.
 */

class StochasticActor : public Actor
{
    public:
        /*!
         * Constructor.
         * Initialize a stochastic actor given a stochastic policy.
         * \param policy_ stochastic policy polymorphic object.
         */
        StochasticActor(StochasticPolicy const &policy_)
            : policyPtr(policy_.clone()) {}

        /*!
         * Copy constructor
         * \param actor_ another stochastic actor
         */
        StochasticActor(StochasticActor const &actor_)
            : policyPtr(actor_.policyPtr->clone()) {}

        //! Default destructor
        virtual ~StochasticActor() = default;

        /*!
         * Get observation size, i.e. size of the observation vector.
         * \return observation size
         */
        size_t getDimObservation() const { return policyPtr->getDimObservation(); }

        /*!
         * Get action size, i.e. size of the action vector.
         * \return action size
         */
        virtual size_t getDimAction() const { return policyPtr->getDimAction(); }

        /*!
         * Get actor's parameters size, i.e. size of the parameter vector
         * \return parameters size
         */
        size_t getDimParameters() const { return policyPtr->getDimParameters(); }

        /*!
         * Get method for the actor's parameters.
         * \return parameters stored in an arma::vector
         */
        arma::vec getParameters() const
            { return policyPtr->getParameters(); }

        /*!
         * Set method for the actor's parameters.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        void setParameters(arma::vec const &parameters)
            { policyPtr->setParameters(parameters); }

        /*!
         * Given an observation, select an action accordind to the policy.
         * \param observation_ observation
         * \return action
         */
        arma::vec getAction(arma::vec const &observation) const
            { return policyPtr->getAction(observation); }

        /*!
         * Evaluate the Likelihood score function at a given observation and
         * action.
         * \param observation_ observation
         * \param action_ action
         * \return likelihood score evaluated at observation_ and action_
         */
        arma::vec likelihoodScore(arma::vec const &observation,
                                  arma::vec const &action) const
            { return policyPtr->likelihoodScore(observation, action); }

        /*!
         * Reset stochatic actor to initial conditions.
         */
        void reset() { policyPtr->reset(); }

    private:
        //! Stochastic policy employed by the agent
        std::unique_ptr<StochasticPolicy> policyPtr;
};

#endif // STOCHASTICACTOR_H
