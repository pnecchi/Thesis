/*
 * Copyright (c) 2016 Pierpaolo Necchi
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ASSETALLOCATIONEXPERIMENT_H
#define ASSETALLOCATIONEXPERIMENT_H

#include <armadillo>
#include <memory>
#include <thesis/AssetAllocationTask.h>
#include <thesis/Agent.h>
#include <thesis/BacktestLog.h>
#include <thesis/Statistics.h>

/**
 * An AssetAllocationExperiment handles the learning process. In particular, it
 * manages the interactions between the AssetAllocationTask and a trading agent
 * and is responsible for logging the strategy backtest performance for multiple
 * independent experiments.
 */

// TODO: Match experiment cache variables with agent inner cache variables in
// order to avoid useless copies

class AssetAllocationExperiment
{
    public:
        /*!
         * Constructor.
         * Initialize an asset-allocation experiment given an asset-allocation
         * task and a learning agent.
         * \param task_ asset allocation task.
         * \param agent_ learning agent.
         * \param numExperiments_ number of independent experiments.
         * \param numEpochs_ number of learning epochs per experiment.
         * \param numTrainingSteps_ number of training steps per epoch.
         * \param numTestSteps_ number of test steps per experiment.
         */
        AssetAllocationExperiment(AssetAllocationTask const &task_,
                                  Agent const &agent_,
                                  size_t numExperiments_,
                                  size_t numEpochs_,
                                  size_t numTrainingSteps_,
                                  size_t numTestSteps_);

        //! Default destructor
        virtual ~AssetAllocationExperiment() = default;

        //! Run experiment
        void run();

    private:
        /*!
         * One interaction agent-task, consisting of the following steps:
         * 1) the agent observes the current state of the system.
         * 2) the agent selects and perform an action.
         * 3) the environment evolves to the next state.
         * 4) the agent receives a numerical reward from the environment.
         */
        void oneInteraction();

        //! Experiment sizes
        size_t numExperiments;
        size_t numEpochs;
        size_t numTrainingSteps;
        size_t numTestSteps;

        //! Asset allocation task
        AssetAllocationTask task;

        //! Trading system
        std::unique_ptr<Agent> agentPtr;

        /*!
         * Data structure storing the information relevant for the analysis of
         * the backtest performances of the trading strategy.
         */
        BacktestLog blog;

        /*!
         * Experiment statistics, i.e. average reward, standard deviation and
         * Sharpe ratio, used to monitor the learning process convergence.
         */
        StatisticsExperiment experimentStats;

        //! Cache variables
        arma::vec observationCache;
        arma::vec actionCache;
        double rewardCache;
};

#endif // ASSETALLOCATIONEXPERIMENT_H
