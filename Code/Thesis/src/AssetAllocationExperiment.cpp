#include "thesis/AssetAllocationExperiment.h"

AssetAllocationExperiment::AssetAllocationExperiment(AssetAllocationTask const &task_,
                                                     Agent const &agent_,
                                                     bool backtestMode_,
                                                     size_t numRecords)
    : task(task_),
      agentPtr(agent_.clone()),
      backtestMode(backtestMode_),
      blog(task.getDimAction(), numRecords),
      actionCache(task.getDimAction()),
      rewardCache(0.0)
{
    /* Nothing to do */
}

void AssetAllocationExperiment::setEvaluationInterval (size_t startDate_,
                                                       size_t endDate_)
{
    task.setEvaluationInterval(startDate_, endDate_);
}

void AssetAllocationExperiment::resetTask()
{
    task.reset();
}

void AssetAllocationExperiment::setBacktestMode(bool backtestMode_)
{
    backtestMode = backtestMode_;
}

void AssetAllocationExperiment::interact()
{
    // 1) Get observation
    agentPtr->receiveObservation(task.getObservation());

    // 2) Perform action
    actionCache = agentPtr->getAction();
    task.performAction(actionCache);

    // 3) Receive reward
    rewardCache = task.getReward();
    agentPtr->receiveReward(rewardCache);

    // 4) Receive next observation
    agentPtr->receiveNextObservation(task.getObservation());

    // 5) Backtest
    if (backtestMode)
        blog.insertRecord(actionCache, rewardCache);

    // 6) Dump results in statistics gatherer
    experimentStats.dumpOneResult(rewardCache);
}

void AssetAllocationExperiment::run(size_t numSteps)
{
    for(size_t n = 0; n < numSteps; n++)
    {
        // Interaction between the task and the agent
        interact();
        // Learning step
        agentPtr->learn();

        if ((n + 1) % 50 == 0)
        {
            std::vector<std::vector<double>> stats = experimentStats.getStatistics();
            std::cout << "Average: " << stats[0][0]
                      << " - Standard Deviation: " << stats[0][1]
                      << " - Sharpe ratio: " << stats[0][2] << std::endl;
        }

    }
}
