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

#ifndef NPGPEPOLICY_H
#define NPGPEPOLICY_H

#include <thesis/StochasticPolicy.h>
#include <armadillo>  /* arma::vec */
#include <memory>     /* std::unique_ptr */

/*!
 * NPGPEPolicy implements a Natural PGPE stochastic policy for online learning,
 * based on a deterministic controller and a gaussian probability distribution
 * for the controller parameters. For further information on NPGPE, refer to
 * "Miyamae et Al. - Natural Policy Gradient Methods with Parameter-based
 * Exploration for Control Tasks (2010)".
 */

class NPGPEPolicy : public StochasticPolicy
{
    public:
        /*!
         * Constructor.
         * Initialize a NPGPEPolicy object given a deterministic controller.
         * \param policy_ deterministic controller
         * \param resamplingProbability_ probability of sampling new controller parameters
         */
        NPGPEPolicy(Policy const &policy_,
                    double resamplingProbability_=0.01);

        /*!
         * Copy constructor for correct instantiation of polymorphic objects.
         */
        NPGPEPolicy(NPGPEPolicy const &other_);

        //! Default destructor.
        virtual ~NPGPEPolicy() = default;

        /*!
         * Get policy parameters size, i.e. size of the parameter vector
         * \return parameters size
         */
        virtual size_t getDimParameters() const { return dimHyperParameters; }

        /*!
         * Get method for the policy parameters.
         * \return parameters stored in an arma::vector
         */
        virtual arma::vec getParameters() const
            { return parameters; }

        /*!
         * Set method for the policy parameters.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        virtual void setParameters(arma::vec const &parameters_)
            { parameters = parameters_; }

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
        //! Initialize parameters
        void initializeParameters();

        //! Virtual inner clone method
        virtual std::unique_ptr<Policy> cloneImpl() const;

        //! Deterministic controller
        std::unique_ptr<Policy> policyPtr;

        //! Controller parameters distributions
        mutable std::normal_distribution<double> gaussianDistr;

        //! Policy parameters size
        size_t dimParameters;

        //! Policy hyperparameters size
        size_t dimHyperParameters;

        //! Parameters distribution parameters
        arma::vec parameters;  // Shares memory with mean and cholFactor
        arma::vec mean;
        arma::mat cholFactor;

        //! Mutable cache variable for random policy parameters
        mutable arma::vec xi;

        //! Resampling probability
        double resamplingProbability;

        //! Random number generator
        mutable std::mt19937 generator;
        mutable std::uniform_real_distribution<double> randDistr;
};

#endif // NPGPEPOLICY_H
