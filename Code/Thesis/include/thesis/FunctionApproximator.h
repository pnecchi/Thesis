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

#ifndef FUNCTIONAPPROXIMATOR_H
#define FUNCTIONAPPROXIMATOR_H

#include <armadillo>  /* arma::vec */
#include <memory>     /* std::unique_ptr */

/*!
 * FunctionApproximator is a pure abstract class that provides a generic
 * interface for a 1-dimensional function approximator F: (x, theta) --> y. It
 * can be used as the core of a critic.
 */

class FunctionApproximator
{
    public:
        /*!
         * Constructor.
         * Initialize a function approximator that takes as input armadillo
         * vectors of a given size.
         * \param dimInput_ the input vector size
         */
        FunctionApproximator(size_t dimInput_) : dimInput(dimInput_) {}

        //! Default destructor.
        virtual ~FunctionApproximator() = default;

        /*!
         * Virtual clone method (abstract)
         * The class is polymorphically clonable, making it possible to write
         * copy constructors and assignment operators for classes that aggregate
         * objects of the FunctionApproximator hierarchy by composition.
         * \return a unique pointer to a copy of the object.
         */
        virtual std::unique_ptr<FunctionApproximator> clone() const = 0;

        /*!
         * Get method for the input dimention.
         * \return input size
         */
        size_t getDimInput() const { return dimInput; }

        /*!
         * Abstract get method for the parameters dimension.
         * \return parameters size
         */
        virtual size_t getDimParameters() const = 0;

        /*!
         * Get method for the function approximator parameters.
         * \return parameters stored in an arma::vector
         */
        virtual arma::vec getParameters() const = 0;

        /*!
         * Set method for the function approximator parameters.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        virtual void setParameters(arma::vec const &parameters_) = 0;

        /*!
         * Evaluate the function approximator for a given input.
         * \param x input vector
         * \return evaluation of the function approximator in x
         */
        virtual double evaluate(arma::vec const &x) const = 0;

        /*!
         * Evaluate the function approximator gradient wrt the parameters.
         * \param x input vector
         * \return function approximator gradient evaluated in x
         */
        virtual arma::vec gradient(arma::vec const &x) const = 0;

        //! Reset function approximator parameters to initial conditions
        virtual void reset() = 0;

    private:
        // Input size
        size_t dimInput;
};

#endif // FUNCTIONAPPROXIMATOR_H
