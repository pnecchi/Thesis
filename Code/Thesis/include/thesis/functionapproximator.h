#ifndef FUNCTIONAPPROXIMATOR_H
#define FUNCTIONAPPROXIMATOR_H

#include <armadillo>  /* arma::vec */
#include <memory>     /* std::unique_ptr */

/**
 * FunctionApproximator is a pure abstract class that provides a generic
 * interface for a 1-dimensional function approximator F: (x, theta) --> y. It
 * can be used as the core of a critic.
 */

class FunctionApproximator
{
    public:
        // Default constructor
        ParametricFunction(size_t dimInput_)
            : dimInput(dimInput_), dimOutput(dimOutput_) {}

        // Virtual destructor
        virtual ~FunctionApproximator() = default;

        /**
         * The class is polymorphically clonable, making it possible to write
         * copy constructors and assignment operators for classes that aggregate
         * objects of the FunctionApproximator hierarchy by composition.
         */
        virtual std::unique_ptr<FunctionApproximator> clone() const = 0;

        // Getter methods for sizes
        size_t getDimInput() const { return dimInput; }
        virtual size_t getDimParams() const = 0;

        // Getter and setter methods for parameters
        virtual arma::vec getParameters() const = 0;
        virtual void setParameters(arma::vec const &parameters) = 0;

        // Evaluate function approximator
        virtual double evaluate(arma::vec const &x) const = 0;

        // Evaluate function approximator gradient wrt the parameters
        virtual arma::vec gradient(arma::vec const &x) const = 0;

    private:
        // Input and output sizes
        size_t dimInput;
};

#endif // FUNCTIONAPPROXIMATOR_H
