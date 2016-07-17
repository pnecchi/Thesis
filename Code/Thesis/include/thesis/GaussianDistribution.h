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

#ifndef GAUSSIANDISTRIBUTION_H
#define GAUSSIANDISTRIBUTION_H

#include <thesis/ProbabilityDistribution.h>
#include <armadillo>  /* arma::vec */
#include <memory>     /* std::unique_ptr */

/*!
 * GaussianDistribution defines an axis-aligned multivariate Gaussian
 * probability distribution.
 */

class GaussianDistribution : public ProbabilityDistribution
{
    public:
        /*!
         * Default constructor.
         * \param dimOutput_ output size
         */
        GaussianDistribution(size_t dimOutput_);

        //! Default copy constructor.
        GaussianDistribution(GaussianDistribution const &other) = default;

        //! Default destructor.
        virtual ~GaussianDistribution() = default;

        //! Virtual clone method for polymorphic copy.
        virtual std::unique_ptr<ProbabilityDistribution> clone() const;

        /*!
         * Get distribution output size.
         * \return output size
         */
        virtual size_t getDimOutput() const { return dimOutput; }

        /*!
         * Get probability distribution size, i.e. size of the parameter vector
         * \return parameters size
         */
        virtual size_t getDimParameters() const { return dimParameters; }

        /*!
         * Get method for the distribution parameters.
         * \return parameters stored in an arma::vector
         */
        virtual arma::vec getParameters() const { return parameters; }

        /*!
         * Set method for the distribution parameters.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        virtual void setParameters(arma::vec const &parameters_)
            { parameters = parameters_; }

        /*!
         * Simulate a realization of the probability distribution.
         * \return realization of the probability distribution
         */
        virtual arma::vec simulate() const;

        /*!
         * Evaluate the Likelihood score of a given realization
         * \param output_ distribution realization
         * \return likelihood score evaluated at output_
         */
        virtual arma::vec likelihoodScore(arma::vec const &output_) const;

        /*!
         * Reset distribution to initial conditions.
         */
        virtual void reset();

    private:

        //! Initialize distribution parameters
        void initializeParameters();

        //! Parameters: [mu_1; ... ; mu_D; sigma_1; ... ; sigma_D]
        arma::vec parameters;

        //! Sizes
        size_t dimOutput;
        size_t dimParameters;

        //! Random number generator
        mutable std::mt19937 generator;
};

#endif // GAUSSIANDISTRIBUTION_H
