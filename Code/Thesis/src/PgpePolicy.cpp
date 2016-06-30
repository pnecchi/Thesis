#include <thesis/PgpePolicy.h>

PGPEPolicy::PGPEPolicy(Policy const &policy_,
                       ProbabilityDistribution const &distribution_,
                       double resamplingProbability_)
    : StochasticPolicy(policy_.getDimObservation(), policy_.getDimAction()),
      policyPtr(policy_.clone()),
      distributionPtr(distribution_.clone()),
      resamplingProbability(resamplingProbability_),
      generator(456),
      randDistr(0.0, 1.0)
{
    /* Nothing to do */
}

PGPEPolicy::PGPEPolicy(PGPEPolicy const &other_)
    : StochasticPolicy(other_.getDimObservation(), other_.getDimAction()),
      policyPtr(other_.policyPtr->clone()),
      distributionPtr(other_.distributionPtr->clone()),
      resamplingProbability(other_.resamplingProbability),
      generator(other_.generator),
      randDistr(other_.randDistr)
{
    /* Nothing to do */
}

arma::vec PGPEPolicy::getAction(arma::vec const &observation_) const
{
    // Simulate policy parameters
    if (randDistr(generator) < resamplingProbability)
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

std::unique_ptr<Policy> PGPEPolicy::cloneImpl() const
{
    return std::unique_ptr<Policy>(new PGPEPolicy(*this));
}
