#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H

#include <thesis/FunctionApproximator.h>
#include <armadillo>

// TODO: Consider generic features received as input by the constructor (lambda)

/*!
 * LinearRegressor is a concrete class that implements a 1-dim linear function
 * approximator F: (x, theta) --> y = <theta, x>. It inherits from the Function
 * Approximator abstract class.
 */

class LinearRegressor : public FunctionApproximator
{
    public:
        /*!
         * Constructor.
         * Initialize a linear regressor that takes as input armadillo vectors
         * of a given size.
         * \param dimInput_ size of the input vector
         */
        LinearRegressor(size_t dimInput_);

        //! Default destructor
        virtual ~LinearRegressor() = default;

        /*!
         * Virtual clone method.
         * \return a unique pointer to a copy of the object.
         */
        virtual std::unique_ptr<FunctionApproximator> clone() const;

        /*!
         * Get method for the parameters dimension.
         * \return parameters size
         */
        virtual size_t getDimParameters() const { return parameters.n_elem; }

        /*!
         * Get method for the linear regressor parameters.
         * \return parameters stored in an arma::vector
         */
        virtual arma::vec getParameters() const;

        /*!
         * Set method for the linear regressor parameters.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        virtual void setParameters(arma::vec const &parameters_);

        /*!
         * Evaluate the linear regressor at a given input.
         * \param x input vector
         * \return evaluation of the linear regressor in x
         */
        virtual double evaluate(arma::vec const &x) const;

        /*!
         * Evaluate the function approximator gradient wrt the parameters.
         * \param x input vector
         * \return function approximator gradient evaluated in x
         */
        virtual arma::vec gradient(arma::vec const &x) const;

    private:
        //! Initialize the linear regressor parameters.
        void initializeParameters();

        //! Parameters
        arma::vec parameters;
};

#endif // LINEARREGRESSOR_H
