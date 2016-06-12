#include <thesis/tradingsystem.h>

TradingSystem::TradingSystem(TradingSystem const &other_)
    : agentPtr(other_.agentPtr->clone()),
      backtestMode(other_.backtestMode),
      actionCache(other_.actionCache),
      rewardCache(other_.rewardCache)
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

arma::vec TradingSystem::getAction() const
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
        // TODO: Dump action and result into log data-structure
    }
}

void TradingSystem::learn()
{
    agentPtr->learn();
}
