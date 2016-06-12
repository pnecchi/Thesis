#ifndef STOCHASTICPOLICY_H
#define STOCHASTICPOLICY_H

#include <armadillo>
#include <memory>
#include <thesis/policy.h>

/**
 * StochasticPolicy is a pure abstract class which inherits from the
 * Policy abstract class and provides a generic interface for a stochastic
 * policy pi(x, a; theta). It provides the likelihoodScore method so that it can
 * be used as the core of an actor in policy gradient method.
 */

class StochasticPolicy : public Policy
{
    public:
        // Default constructor
        StochasticPolicy(size_t dimInput_, size_t dimOutput_)
            : ParametricFunction(dimInput_, dimOutput_) {}

        // Default destructor
        virtual ~StochasticPolicy() = default;

        // Virtual clone method for polymorphic clone
        virtual std::unique_ptr<ParameterFunction> clone() const = 0;

        // Getter methods for sizes
        virtual size_t getDimParams() const = 0;

        // Getter and setter methods for parameters
        virtual arma::vec getParameters() const = 0;
        virtual void setParameters(arma::vec const &parameters) = 0;

        // Select action given
        virtual arma::vec getAction(arma::vec const &observation) = 0;

        // Likelihood score function
        virtual arma::vec likelihoodScore(arma::vec const &observation,
                                          arma::vec const &action) const = 0;
};

#endif // STOCHASTICPOLICY_H
