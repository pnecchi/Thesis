#ifndef STOCHASTICACTOR_H
#define STOCHASTICACTOR_H

#include <armadillo>
#include <memory>
#include <thesis/actor.h>
#include <thesis/stochasticpolicy.h>

class StochasticActor : public Actor
{
    public:
        // Constructor
        StochasticActor(StochasticPolicy const &policy_)
            : policyPtr(policy_.clone()) {}

        // Copy constructor
        StochasticActor(StochasticActor const &actor_)
            : policyPtr(actor_.policyPtr->clone()) {}

        // Default destructor
        virtual ~StochasticActor() = default;

        // Get sizes
        size_t getDimObservation() const { return policyPtr->getDimObservation(); }
        size_t getDimAction() const { return policyPtr->getDimAction(); }
        size_t getDimParameters() const { return policyPtr->getDimParameters(); }

        // Getter and setter methods for parameters
        arma::vec getParameters() const
            { return policyPtr->getParameters(); }
        void setParameters(arma::vec const &parameters)
            { policyPtr->setParameters(parameters); }

        // Get Action
        arma::vec getAction(arma::vec const &observation) const
            { return policyPtr->getAction(observation); }

        // Likelihood score function
        arma::vec likelihoodScore(arma::vec const &observation,
                                  arma::vec const &action) const
            { return policyPtr->likelihoodScore(observation, action); }

    private:
        // Stochastic policy employed by the agent
        std::unique_ptr<StochasticPolicy> policyPtr;
};

#endif // STOCHASTICACTOR_H
