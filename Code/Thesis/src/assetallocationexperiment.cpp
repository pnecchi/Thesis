#include "thesis/assetallocationexperiment.h"

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
    agent.setBacktestMode(backtestMode_);
}

void AssetAllocationExperiment::interact()
{
    // Get observation
    agent.receiveObservation(task.getObservation());
    // Perform action
    task.performAction(agent.getAction());
    // Receive reward
    agent.receiveReward(task.getReward());
}

void AssetAllocationExperiment::run(size_t numSteps)
{
    for(size_t n = 0; n < numSteps; n++)
    {
        // Interaction between the task and the agent
        interact();
        // Learning step
        agent.learn();
    }
}
