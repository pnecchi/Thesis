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

#ifndef BOLTZMANNEXPLORATIONPOLICY_H
#define BOLTZMANNEXPLORATIONPOLICY_H

#include <thesis/StochasticPolicy.h>  /* StochasticPolicy */
#include <armadillo>                  /* arma::vec */
#include <vector>                     /* std::vector */
#include <random>

/*!
 * BoltzmannPolicy is a class that implements the Boltzmann stochastic policy
 * for a finite action space. For further information, cf. Wiering, M. and Van
 * Otterlo, M. - "Reinforcement learning." Adaptation, Learning, and
 * Optimization 12 (2012).
 */

class BoltzmannPolicy : public StochasticPolicy
{
    public:
        /*!
         * Constructor.
         * Initialize a Boltzmann policy given the sizes of the observation
         * space and a list of the possible actions that can be selected.
         * \param dimObservation_ dimension of the observation space
         * \param possibleActions_ vector of possible actions
         */
        BoltzmannPolicy(size_t dimObservation_,
                        std::vector<double> possibleActions_);

        //! Default destructor
        virtual ~BoltzmannPolicy() = default;

        //! Clone method for polymorphic clone
        std::unique_ptr<StochasticPolicy> clone() const
        {
            return checkedClone<StochasticPolicy>();
        }

        /*!
         * Get policy parameters size, i.e. size of the parameter vector
         * \return parameters size
         */
        virtual size_t getDimParameters() const { return dimParameters; }

        /*!
         * Get method for the policy parameters.
         * \return parameters stored in an arma::vector
         */
        virtual arma::vec getParameters() const;

        /*!
         * Set method for the policy parameters.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        virtual void setParameters(arma::vec const &parameters_);

        /*!
         * Given an observation, select an action accordind to the policy.
         * \param observation_ observation
         * \return action
         */
        virtual arma::vec getAction(arma::vec const &observation_) const;

        /*!
         * Evaluate the Likelihood score function at a given observation and
         * action.
         * \param observation_ observation
         * \param action_ action
         * \return likelihood score evaluated at observation_ and action_
         */
        virtual arma::vec likelihoodScore(arma::vec const &observation_,
                                          arma::vec const &action_) const;

        /*!
         * Reset policy to initial conditions.
         */
        virtual void reset();

    private:
        //! Virtual inner clone method
        virtual std::unique_ptr<Policy> cloneImpl() const;

        //! Initialize the Boltzmann policy parameters
        void initializeParameters();

        //! Possible actions
        std::vector<double> possibleActions;

        //! Number of possible actions
        size_t numPossibleActions;

        //! Number of parameters per action
        size_t dimParametersPerAction;

        //! Parameters size
        size_t dimParameters;

        //! Policy parameters: Theta = [ theta_1 | thetha_2 | ... | theta_D ]
        arma::mat parametersMat;
        arma::vec paramatersVec;  // Linearized matrix (shares memory)

        /*!
         * Boltzmann probability distribution and random number generator
         * Need to be mutable because the generator state changes when
         * simulating an action.
         */
        mutable std::mt19937 generator;
        mutable std::vector<double> boltzmannProbabilities;

        // TODO: Consider generic features of the input
};

#endif // BOLTZMANNEXPLORATIONPOLICY_H
