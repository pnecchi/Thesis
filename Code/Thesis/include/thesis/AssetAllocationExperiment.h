#ifndef ASSETALLOCATIONEXPERIMENT_H
#define ASSETALLOCATIONEXPERIMENT_H

#include <memory>
#include <thesis/AssetAllocationTask.h>
#include <thesis/TradingSystem.h>

/**
 * An AssetAllocationExperiment handles the interactions between the
 * AssetAllocationTask and a trading agent.
 */

class AssetAllocationExperiment
{
    public:
        // Default constructor
        AssetAllocationExperiment(AssetAllocationTask const &task_,
                                  TradingSystem const &agent_)
            : task(task_), agent(agent_) {}

        // Default destructor
        virtual ~AssetAllocationExperiment() = default;

        // Set evaluation interval for the allocation task
        void setEvaluationInterval (size_t startDate_, size_t endDate_);

        // Reset task
        void resetTask();

        // Set backtesting mode
        void setBacktestMode(bool backtestMode_);

        // Run experiment
        void run(size_t numSteps);

    private:
        // Interaction agent-task
        void interact();

        // Asset allocation task
        AssetAllocationTask task;

        // Trading system
        TradingSystem agent;
};

#endif // ASSETALLOCATIONEXPERIMENT_H
