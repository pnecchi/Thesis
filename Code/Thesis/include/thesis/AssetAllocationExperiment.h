#ifndef ASSETALLOCATIONEXPERIMENT_H
#define ASSETALLOCATIONEXPERIMENT_H

#include <armadillo>
#include <memory>
#include <thesis/AssetAllocationTask.h>
#include <thesis/Agent.h>
#include <thesis/BacktestLog.h>
#include <thesis/Statistics.h>

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
                                  size_t numExperiments_,
                                  size_t numEpochs_,
                                  size_t numTrainingSteps_,
                                  size_t numTestSteps_);

        // Default destructor
        virtual ~AssetAllocationExperiment() = default;

        // Run experiment
        void run();

    private:
        // Interaction agent-task
        void oneInteraction();

        // Experiment sizes
        size_t numExperiments;
        size_t numEpochs;
        size_t numTrainingSteps;
        size_t numTestSteps;

        // Asset allocation task
        AssetAllocationTask task;

        // Trading system
        std::unique_ptr<Agent> agentPtr;

        // Backtest Log
        BacktestLog blog;

        // Cache variables
        arma::vec observationCache;
        arma::vec actionCache;
        double rewardCache;

        // Experiment statistics
        StatisticsExperiment experimentStats;
};

#endif // ASSETALLOCATIONEXPERIMENT_H
