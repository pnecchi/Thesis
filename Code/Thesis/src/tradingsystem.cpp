#include <thesis/tradingsystem.h>

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
