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

#ifndef CRITIC_H
#define CRITIC_H

#include <armadillo>
#include <memory>
#include <thesis/FunctionApproximator.h>

/*!
 * Critic implements the generic interface of a critic for a state-value
 * function. It is based on the FunctionApproximator hierarchy by composition,
 * so that different function approximators can be easily used as the core of
 * critic.
 */

class Critic
{
    public:
        /*!
         * Constructor.
         * Initialize a critic using a function approximator.
         * \param approximator_ function approximator
         */
        Critic(FunctionApproximator const &approximator_)
            : approximatorPtr(approximator_.clone()) {}

        /*!
         * Copy constructor.
         * \param rhs critic to copy
         */
        Critic(Critic const &rhs)
            : approximatorPtr(rhs.approximatorPtr->clone()) {}

        //! Default destructor
        virtual ~Critic() = default;

        /*!
         * Get method for the input dimention.
         * \return input size
         */
        size_t getDimInput() const { return approximatorPtr->getDimInput(); }

        /*!
         * Get method for the parameters dimension.
         * \return parameters size
         */
        size_t getDimParameters() const { return approximatorPtr->getDimParameters(); }

        /*!
         * Get method for the critic parameters.
         * \return parameters stored in an arma::vector
         */
        arma::vec getParameters() const
            { return approximatorPtr->getParameters(); }

        /*!
         * Set method for the critic parameters.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        void setParameters(arma::vec const &parameters_)
            { approximatorPtr->setParameters(parameters_); }

        /*!
         * Evaluate the critic for a given observation.
         * \param observation_ observation
         * \return evaluation of the critic for this observation
         */
        double evaluate(arma::vec &observation_) const
            { return approximatorPtr->evaluate(observation_); }

        /*!
         * Evaluate the critic's gradient wrt the parameters.
         * \param observation_ observation
         * \return evaluation of the critic's gradient for this observation
         */
        arma::vec gradient(arma::vec const &observation) const
            { return approximatorPtr->gradient(observation); }

        //! Reset critic to initial conditions
        void reset() { approximatorPtr->reset(); }

    private:
        //! Function approximator
        std::unique_ptr<FunctionApproximator> approximatorPtr;
};

#endif // CRITIC_H
