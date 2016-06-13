#ifndef CRITIC_H
#define CRITIC_H

#include <armadillo>
#include <memory>
#include <thesis/functionapproximator.h>

class Critic
{
    public:
        // Default constructor
        Critic(FunctionApproximator const &approximator_)
            : approximatorPtr(approximator_.clone()) {}

        // Copy constructor
        Critic(Critic const &critic_)
            : approximatorPtr(critic_.approximatorPtr->clone()) {}

        // Default destructor
        virtual ~Critic() = default;

        // Get sizes
        size_t getDimInput() const { return approximatorPtr->getDimInput(); }
        size_t getDimParams() const { return approximatorPtr->getDimParams(); }

        // Getter and setter methods for parameters
        arma::vec getParameters() const
            { return approximatorPtr->getParameters(); }
        void setParameters(arma::vec const &parameters)
            { approximatorPtr->setParameters(parameters); }

        // Evaluate
        double evaluate(arma::vec &observation) const
            { return approximatorPtr->evaluate(observation); }

        // Gradient
        arma::vec gradient(arma::vec const &observation) const
            { return approximatorPtr->gradient(observation); }

    private:
        // Function approximator
        std::unique_ptr<FunctionApproximator> approximatorPtr;
};

#endif // CRITIC_H
