#include <thesis/PgpePolicy.h>

PGPEPolicy::PGPEPolicy(Policy const &policy_,
                       ProbabilityDistribution const &distribution_)
    : StochasticPolicy(policy_.getDimObservation(), policy_.getDimAction()),
      policyPtr(policy_.clone()),
      distributionPtr(distribution_.clone())
{
    /* Nothing to do */
}

PGPEPolicy::PGPEPolicy(PGPEPolicy const &other_)
    : StochasticPolicy(other_.getDimObservation(), other_.getDimAction()),
      policyPtr(other_.policyPtr->clone()),
      distributionPtr(other_.distributionPtr->clone())
{
    /* Nothing to do */
}

std::unique_ptr<Policy> PGPEPolicy::cloneImpl() const
{
    return std::unique_ptr<Policy>(new PGPEPolicy(*this));
}

arma::vec PGPEPolicy::getAction(arma::vec const &observation_) const
{
    // Simulate policy parameters
    policyPtr->setParameters(distributionPtr->simulate());

    // Select action
    return policyPtr->getAction(observation_);
}

arma::vec PGPEPolicy::likelihoodScore(arma::vec const &observation_,
                          arma::vec const &action_) const
{
    return distributionPtr->likelihoodScore(policyPtr->getParameters());
}

void PGPEPolicy::reset()
{
    policyPtr->reset();
    distributionPtr->reset();
}
