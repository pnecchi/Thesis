#ifndef ASSETALLOCATIONEXPERIMENT_H
#define ASSETALLOCATIONEXPERIMENT_H

#include <armadillo>
#include <memory>
#include <thesis/AssetAllocationTask.h>
#include <thesis/Agent.h>
#include <thesis/BacktestLog.h>

/**
 * An AssetAllocationExperiment handles the interactions between the
 * AssetAllocationTask and a trading agent.
 */

class AssetAllocationExperiment
{
    public:
        // Default constructor
        AssetAllocationExperiment(AssetAllocationTask const &task_,
                                  Agent const &agent_,
                                  bool backtestMode_,
                                  size_t numRecords);

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
        std::unique_ptr<Agent> agentPtr;

        // Backtest Log
        bool backtestMode = false;
        BacktestLog blog;

        // Cache variables
        arma::vec actionCache;
        double rewardCache;

        // TODO: Add experiment statistics class
};

#endif // ASSETALLOCATIONEXPERIMENT_H
