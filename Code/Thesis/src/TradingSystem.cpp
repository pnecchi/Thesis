#include <thesis/TradingSystem.h>

TradingSystem::TradingSystem(Agent const &agent_,
                             bool backtestMode_,
                             size_t numRecords_)
    : agentPtr(agent_.clone()),
      backtestMode(backtestMode_),
      blog(agentPtr->getDimAction(), numRecords_)
{
    /* Nothing to do */
}

TradingSystem::TradingSystem(TradingSystem const &other_)
    : agentPtr(other_.agentPtr->clone()),
      backtestMode(other_.backtestMode),
      actionCache(other_.actionCache),
      rewardCache(other_.rewardCache),
      blog(other_.blog)
{
    /* Nothing to do */
}

std::unique_ptr<Agent> TradingSystem::clone() const
{
    return std::unique_ptr<Agent>(new TradingSystem(*this));
}

void TradingSystem::receiveObservation(arma::vec const &observation)
{
    agentPtr->receiveObservation(observation);
}

arma::vec TradingSystem::getAction()
{
    actionCache = agentPtr->getAction();
    return actionCache;
}

void TradingSystem::receiveReward(double const reward)
{
    rewardCache = reward;
    agentPtr->receiveReward(reward);

    // Log action and reward for backtesting
    if (backtestMode)
    {
        blog.insertRecord(actionCache, rewardCache);
    }
}

void TradingSystem::receiveNextObservation(arma::vec const &nextObservation_)
{
    agentPtr->receiveNextObservation(nextObservation_);
}

void TradingSystem::learn()
{
    agentPtr->learn();
}
