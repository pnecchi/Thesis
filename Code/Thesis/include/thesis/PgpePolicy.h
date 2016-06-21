#ifndef PGPEPOLICY_H
#define PGPEPOLICY_H

#include <thesis/StochasticPolicy.h>
#include <thesis/Policy.h>
#include <thesis/ProbabilityDistribution.h>
#include <memory>

class PGPEPolicy : public StochasticPolicy
{
    public:
        PGPEPolicy(Policy const &policy_,
                   ProbabilityDistribution const &distribution_);

        PGPEPolicy(PGPEPolicy const &other_);

        virtual ~PGPEPolicy() = default;

        virtual size_t getDimParameters() const
            { return distributionPtr->getDimParameters(); }

        virtual arma::vec getParameters() const
            { return distributionPtr->getParameters(); }

        virtual void setParameters(arma::vec const &parameters_)
            { distributionPtr->setParameters(parameters_); }

        virtual arma::vec getAction(arma::vec const &observation_) const;

        virtual arma::vec likelihoodScore(arma::vec const &observation_,
                                          arma::vec const &action_) const;

        virtual void reset();

    protected:
    private:
        //! Virtual inner clone method
        virtual std::unique_ptr<Policy> cloneImpl() const;

        // Deterministic controller
        std::unique_ptr<Policy> policyPtr;

        // Controller parameters distributions
        std::unique_ptr<ProbabilityDistribution> distributionPtr;
};

#endif // PGPEPOLICY_H
