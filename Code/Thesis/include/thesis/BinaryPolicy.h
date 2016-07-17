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

#ifndef BINARYPOLICY_H
#define BINARYPOLICY_H

#include <thesis/Policy.h>
#include <armadillo>        /* arma::vec */
#include <memory>           /* std::unique_ptr */
#include <limits>           /* std::numeric_limits<double> */

/*!
 * BinaryPolicy implements a deterministic parametric policy that produces
 * actions in {-1, 1} following
 *     a = sign( theta' * observation )
 * This policy is used to implemnet a long-short trading system that invests
 * in a single risky asset and can be used as a controller for a PGPE policy.
 */

class BinaryPolicy : public Policy
{
    public:
        /*!
         * Constructor.
         * Initialize a BinaryPolicy object given the size of the observation.
         * \param dimObservation_ dimension of the observation space
         * \param paramMinValue_ parameters lower bound
         * \param paramMaxValue_ parameters upper bound
         */
        BinaryPolicy(size_t dimObservation_,
                     double paramMinValue_=std::numeric_limits<double>::min(),
                     double paramMaxValue_=std::numeric_limits<double>::max());

        //! Default copy constructor
        BinaryPolicy(BinaryPolicy const &other_) = default;

        //! Default destructor.
        virtual ~BinaryPolicy() = default;

        /*!
         * Get policy parameters size, i.e. size of the parameter vector
         * \return parameters size
         */
        virtual size_t getDimParameters() const { return dimParameters; }

        /*!
         * Get method for the policy parameters.
         * \return parameters stored in an arma::vector
         */
        virtual arma::vec getParameters() const { return parameters; }

        /*!
         * Set method for the policy parameters. The parameters bounds are enforced.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        virtual void setParameters(arma::vec const & parameters_);

        /*!
         * Given an observation, select an action accordind to the policy.
         * \param observation_ observation
         * \return action
         */
        virtual arma::vec getAction(arma::vec const & observation_) const;

        /*!
         * Reset policy to initial conditions.
         */
        virtual void reset();

    private:
        //! Virtual inner clone method
        virtual std::unique_ptr<Policy> cloneImpl() const;

        //! Initialize BinaryPolicy parameters
        void initializeParameters();

        //! Policy parameters
        arma::vec parameters;
        size_t dimParameters;

        /*!
         * Parameters bounds.
         * The parameters must lie in the interval [paramMinValue, paramMaxValue]
         * This constraint can be useful to avoid divergence when modifying the
         * parameters by gradient ascent in an optimization procedure.
         */
        double paramMinValue;
        double paramMaxValue;
};

#endif // BINARYPOLICY_H
