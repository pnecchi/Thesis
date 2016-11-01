#include <thesis/FactoryOfAgents.h>
#include <thesis/LinearRegressor.h>
#include <thesis/Critic.h>
#include <thesis/BoltzmannPolicy.h>
#include <thesis/StochasticActor.h>
#include <thesis/BinaryPolicy.h>
#include <thesis/LongShortPolicy.h>
#include <thesis/GaussianDistribution.h>
#include <thesis/PgpePolicy.h>

FactoryOfAgents& FactoryOfAgents::instance(size_t const &dimObservation_,
                                           LearningRate const &baselineLearningRate_,
                                           LearningRate const &criticLearningRate_,
                                           LearningRate const &actorLearningRate_,
                                           double const &lambda_)
{
    static FactoryOfAgents factory(dimObservation_,
                                   baselineLearningRate_,
                                   criticLearningRate_,
                                   actorLearningRate_,
                                   lambda_);
    return factory;
}

FactoryOfAgents::FactoryOfAgents(size_t const &dimObservation_,
                                 LearningRate const &baselineLearningRate_,
                                 LearningRate const &criticLearningRate_,
                                 LearningRate const &actorLearningRate_,
                                 double const &lambda_)
    : dimObservation(dimObservation_),
      baselineLearningRatePtr(baselineLearningRate_.clone()),
      criticLearningRatePtr(criticLearningRate_.clone()),
      actorLearningRatePtr(actorLearningRate_.clone()),
      lambda(lambda_)
{
    /* Nothing to do */
}

std::unique_ptr<Agent> FactoryOfAgents::make(std::string const &agentId) const
{
    if (agentId == "ARAC")
        return makeARACAgent();
    else if (agentId == "PGPE")
        return makePGPEAgent();
    else if (agentId == "NPGPE")
        return makeNPGPEAgent();
    else if (agentId == "RSARAC")
        return makeRSARACAgent();
    else if (agentId == "RSPGPE")
        return makeRSPGPEAgent();
    else if (agentId == "RSNPGPE")
        return makeRSNPGPEAgent();
    else
    {
        throw std::invalid_argument("Unknown learning algorithm " + agentId);
        return makePGPEAgent();
    }
}

//----------//
// Builders //
//----------//

std::unique_ptr<ARACAgent> FactoryOfAgents::makeARACAgent() const
{
    // State-value function critic
    LinearRegressor linearRegV(dimObservation);

    // Initialize critics
    Critic critic(linearRegV);

    // Boltzmann Policy
    std::vector<double> possibleAction {-1.0, 1.0};
    BoltzmannPolicy policy(dimObservation, possibleAction);

    // Stochastic Actor
    StochasticActor actor(policy);

    // ARAC Agent
    return std::unique_ptr<ARACAgent>(new ARACAgent(actor,
                                                    critic,
                                                    *baselineLearningRatePtr,
                                                    *criticLearningRatePtr,
                                                    *actorLearningRatePtr,
                                                    lambda));
}

std::unique_ptr<ARACAgent> FactoryOfAgents::makePGPEAgent() const
{
    // State-value function critic
    LinearRegressor linearRegV(dimObservation);

    // Initialize critics
    Critic critic(linearRegV);

    // Binary policy
    BinaryPolicy controller(dimObservation);
    GaussianDistribution distribution(controller.getDimParameters());
    PGPEPolicy policy(controller, distribution, 1.0);

    // Stochastic Actor
    StochasticActor actor(policy);

    // ARAC Agent
    return std::unique_ptr<ARACAgent> (new ARACAgent(actor,
                                                     critic,
                                                     *baselineLearningRatePtr,
                                                     *criticLearningRatePtr,
                                                     *actorLearningRatePtr,
                                                     lambda));
}

std::unique_ptr<NPGPEAgent> FactoryOfAgents::makeNPGPEAgent() const
{
    // PGPE Binary policy
    BinaryPolicy controller(dimObservation);

    // NPGPE Agent

    return  std::unique_ptr<NPGPEAgent> (new NPGPEAgent(controller,
                                                        *baselineLearningRatePtr,
                                                        *actorLearningRatePtr,
                                                        lambda));
}

std::unique_ptr<ARRSACAgent> FactoryOfAgents::makeRSARACAgent() const
{
    // State-value function critic
    LinearRegressor linearRegV(dimObservation);
    LinearRegressor linearRegU(dimObservation);

    // Initialize critics
    Critic criticV(linearRegV);
    Critic criticU(linearRegU);

    // Boltzmann Policy
    std::vector<double> possibleAction {-1.0, 1.0};
    BoltzmannPolicy policy(dimObservation, possibleAction);

    // Stochastic Actor
    StochasticActor actor(policy);

    // ARSSAC Agent
    return std::unique_ptr<ARRSACAgent> (new ARRSACAgent(actor,
                                                         criticV,
                                                         criticU,
                                                         *baselineLearningRatePtr,
                                                         *criticLearningRatePtr,
                                                         *actorLearningRatePtr,
                                                         lambda));
}

std::unique_ptr<ARRSACAgent> FactoryOfAgents::makeRSPGPEAgent() const
{
    // State-value function critic
    LinearRegressor linearRegV(dimObservation);
    LinearRegressor linearRegU(dimObservation);

    // Initialize critics
    Critic criticV(linearRegV);
    Critic criticU(linearRegU);

    // Binary policy
    BinaryPolicy controller(dimObservation);
    GaussianDistribution distribution(controller.getDimParameters());
    PGPEPolicy policy(controller, distribution, 1.0);

    // Stochastic Actor
    StochasticActor actor(policy);

    // ARSSAC Agent
    return std::unique_ptr<ARRSACAgent> (new ARRSACAgent(actor,
                                                         criticV,
                                                         criticU,
                                                         *baselineLearningRatePtr,
                                                         *criticLearningRatePtr,
                                                         *actorLearningRatePtr,
                                                         lambda));
}

std::unique_ptr<RiskSensitiveNPGPEAgent> FactoryOfAgents::makeRSNPGPEAgent() const
{
    // Binary policy
    BinaryPolicy controller(dimObservation);

    // NPGPE Agent
    return std::unique_ptr<RiskSensitiveNPGPEAgent>
        (new RiskSensitiveNPGPEAgent (controller,
                                      *baselineLearningRatePtr,
                                      *actorLearningRatePtr,
                                      lambda));
}


/*------------------------------------/
 * FactoryOfAgentsForTwoAssetsProblem /
 *-----------------------------------*/

FactoryOfAgentsForTwoAssetsProblem& FactoryOfAgentsForTwoAssetsProblem::instance(size_t const &dimObservation_,
                                           LearningRate const &baselineLearningRate_,
                                           LearningRate const &criticLearningRate_,
                                           LearningRate const &actorLearningRate_,
                                           double const &lambda_)
{
    static FactoryOfAgentsForTwoAssetsProblem factory(dimObservation_,
                                                      baselineLearningRate_,
                                                      criticLearningRate_,
                                                      actorLearningRate_,
                                                      lambda_);
    return factory;
}

FactoryOfAgentsForTwoAssetsProblem::FactoryOfAgentsForTwoAssetsProblem(size_t const &dimObservation_,
                                 LearningRate const &baselineLearningRate_,
                                 LearningRate const &criticLearningRate_,
                                 LearningRate const &actorLearningRate_,
                                 double const &lambda_)
    : dimObservation(dimObservation_),
      baselineLearningRatePtr(baselineLearningRate_.clone()),
      criticLearningRatePtr(criticLearningRate_.clone()),
      actorLearningRatePtr(actorLearningRate_.clone()),
      lambda(lambda_)
{
    /* Nothing to do */
}

std::unique_ptr<Agent> FactoryOfAgentsForTwoAssetsProblem::make(std::string const &agentId) const
{
    if (agentId == "PGPE")
        return makePGPEAgent();
    else if (agentId == "RSNPGPE")
        return makeRSNPGPEAgent();
    else
    {
        throw std::invalid_argument("Unknown learning algorithm " + agentId);
        return makePGPEAgent();
    }
}

//----------//
// Builders //
//----------//

std::unique_ptr<ARACAgent> FactoryOfAgentsForTwoAssetsProblem::makePGPEAgent() const
{
    // State-value function critic
    LinearRegressor linearRegV(dimObservation);

    // Initialize critics
    Critic critic(linearRegV);

    // Binary policy
    LongShortPolicy controller(dimObservation);
    GaussianDistribution distribution(controller.getDimParameters());
    PGPEPolicy policy(controller, distribution, 1.0);

    // Stochastic Actor
    StochasticActor actor(policy);

    // ARAC Agent
    return std::unique_ptr<ARACAgent> (new ARACAgent(actor,
                                                     critic,
                                                     *baselineLearningRatePtr,
                                                     *criticLearningRatePtr,
                                                     *actorLearningRatePtr,
                                                     lambda));
}

std::unique_ptr<RiskSensitiveNPGPEAgent> FactoryOfAgentsForTwoAssetsProblem::makeRSNPGPEAgent() const
{
    // Binary policy
    LongShortPolicy controller(dimObservation);

    // NPGPE Agent
    return std::unique_ptr<RiskSensitiveNPGPEAgent>
        (new RiskSensitiveNPGPEAgent (controller,
                                      *baselineLearningRatePtr,
                                      *actorLearningRatePtr,
                                      lambda));
}

