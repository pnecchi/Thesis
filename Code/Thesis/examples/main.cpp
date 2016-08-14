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

//-----------------|
// Common includes |
//-----------------|

#include <iostream>
#include <stdexcept>
#include <string>
#include <armadillo>
#include <getpot.h>
#include <memory>
#include <thesis/ExperimentParameters.h>
#include <thesis/MarketEnvironment.h>
#include <thesis/AssetAllocationTask.h>
#include <thesis/Agent.h>
#include <thesis/AssetAllocationExperiment.h>
#include <thesis/LearningRate.h>

//-----------------|
// ARAC and RSARAC |
//-----------------|

#include <thesis/LinearRegressor.h>
#include <thesis/Critic.h>
#include <thesis/BoltzmannPolicy.h>
#include <thesis/StochasticActor.h>
#include <thesis/AracAgent.h>
#include <thesis/ArrsacAgent.h>

//-----------------|
// PGPE and RSPGPE |
//-----------------|

#include <thesis/BinaryPolicy.h>
#include <thesis/GaussianDistribution.h>
#include <thesis/PgpePolicy.h>

//-------------------|
// NPGPE and RSNPGPE |
//-------------------|

#include <thesis/NpgpeAgent.h>
#include <thesis/RiskSensitiveNpgpeAgent.h>


/*! Main function used for debugging. It doesn't not take options from the command
    line and the paths to the parameters file and output directories is hard-coded.
 */

int main()
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "-        Algorithmic Asset Allocation        -" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    // Get algorithm
	std::string algorithm = "ARAC";

	// Get file with parameter values
	std::string parametersFilepath = "/home/pierpaolo/Documents/University/6_Anno_Poli/7_Thesis/Data/Parameters/Single_Synth_RN_P0_F0_S0_N5.pot";

    // Read input file path
    const std::string inputFile = "/home/pierpaolo/Documents/University/6_Anno_Poli/7_Thesis/Data/Input/historical_single.csv";

    // Read output directory path
    const std::string outputDir = "/home/pierpaolo/Documents/University/6_Anno_Poli/7_Thesis/Data/Output/Default/";

    // Read debug directory path
    const std::string debugDir = "/home/pierpaolo/Documents/University/6_Anno_Poli/7_Thesis/Data/Debug/Default/";

    //---------------|
    // 1) Parameters |
    //---------------|

    // 1) Read parameters
    std::cout << "1) Read parameters" << std::endl;
    const ExperimentParameters params(parametersFilepath, true);

    // Copy parameters
    double riskFreeRate = params.riskFreeRate;
    double deltaP = params.deltaP;
    double deltaF = params.deltaF;
    double deltaS = params.deltaS;
    size_t numDaysObserved = params.numDaysObserved;
    double lambda = params.lambda;
    double alphaConstActor = params.alphaConstActor;
    double alphaExpActor = params.alphaExpActor;
    double alphaConstCritic = params.alphaConstCritic;
    double alphaExpCritic = params.alphaExpCritic;
    double alphaConstBaseline = params.alphaConstBaseline;
    double alphaExpBaseline = params.alphaExpBaseline;
    size_t numExperiments = params.numExperiments;
    size_t numEpochs = params.numEpochs;
    size_t numTrainingSteps = params.numTrainingSteps;
    size_t numTestSteps = params.numTestSteps;

    //-------------------|
    // 2) Initialization |
    //-------------------|
    std::cout << std::endl << "2) Initialization" << std::endl;

    //----------------------|
    // 2.1) Market and Task |
    //----------------------|

	// Market
	std::cout << ".. Market environment - ";
	MarketEnvironment market(inputFile);
    size_t startDate = 0;
	size_t endDate = numDaysObserved + numTrainingSteps + numTestSteps - 1;
    market.setEvaluationInterval(startDate, endDate);
    std::cout << "done" << std::endl;

    // Asset allocation task
    std::cout << ".. Asset allocation task - ";
	AssetAllocationTask task(market,
                             riskFreeRate,
                             deltaP,
                             deltaF,
                             deltaS,
                             numDaysObserved);
    std::cout << "done" << std::endl;

    //------------|
    // 2.2) Agent |
    //------------|

    // Learning Rates
    std::cout << ".. Learning Rates - ";
    DecayingLearningRate baselineLearningRate(alphaConstBaseline, alphaExpBaseline);
    DecayingLearningRate criticLearningRate(alphaConstCritic, alphaExpCritic);
    DecayingLearningRate actorLearningRate(alphaConstActor, alphaExpActor);
    std::cout << "done" << std::endl;

    // Pointer to Agent for poymorphic object handling
    std::shared_ptr<Agent> agentPtr;

    if (algorithm == "ARAC")
    {
        // State-value function critic
        std::cout << ".. Linear regressors - ";
        LinearRegressor linearRegV(task.getDimObservation());
        std::cout << "done" << std::endl;

        // Initialize critics
        std::cout << ".. Critics - ";
        Critic critic(linearRegV);
        std::cout << "done" << std::endl;

        // Boltzmann Policy
        std::cout << ".. Boltzmann stochastic policy - ";
        std::vector<double> possibleAction {-1.0, 1.0};
        BoltzmannPolicy policy(task.getDimObservation(), possibleAction);
        std::cout << "done" << std::endl;

        // Stochastic Actor
        std::cout << ".. Actor - ";
        StochasticActor actor(policy);
        std::cout << "done" << std::endl;

        // ARAC Agent
        std::cout << ".. ARAC Agent - ";
        agentPtr = std::make_shared<ARACAgent>(actor,
                                               critic,
                                               baselineLearningRate,
                                               criticLearningRate,
                                               actorLearningRate,
                                               lambda);
        std::cout << "done" << std::endl;
    }
    else if (algorithm == "PGPE")
    {
        // State-value function critic
        std::cout << ".. Linear regressors - ";
        LinearRegressor linearRegV(task.getDimObservation());
        std::cout << "done" << std::endl;

        // Initialize critics
        std::cout << ".. Critics - ";
        Critic critic(linearRegV);
        std::cout << "done" << std::endl;

        // Binary policy
        std::cout << ".. PGPE binary policy - ";
        BinaryPolicy controller(task.getDimObservation());
        GaussianDistribution distribution(controller.getDimParameters());
        PGPEPolicy policy(controller, distribution, 1.0);
        std::cout << "done" << std::endl;

        // Stochastic Actor
        std::cout << ".. Actor - ";
        StochasticActor actor(policy);
        std::cout << "done" << std::endl;

        // ARAC Agent
        std::cout << ".. ARAC Agent - ";
        agentPtr = std::make_shared<ARACAgent> (actor,
                                                critic,
                                                baselineLearningRate,
                                                criticLearningRate,
                                                actorLearningRate,
                                                lambda);
        std::cout << "done" << std::endl;

    }
    else if (algorithm == "NPGPE")
    {
        // PGPE Binary policy
        std::cout << ".. Policy - ";
        BinaryPolicy controller(task.getDimObservation());
        std::cout << "done" << std::endl;

        // NPGPE Agent
        std::cout << ".. NPGPE Agent - ";
        agentPtr = std::make_shared<NPGPEAgent> (controller,
                                                 baselineLearningRate,
                                                 actorLearningRate,
                                                 lambda);
        std::cout << "done" << std::endl;
    }
    else if (algorithm == "RSARAC")
    {
        // State-value function critic
        std::cout << ".. Linear regressors - ";
        LinearRegressor linearRegV(task.getDimObservation());
        LinearRegressor linearRegU(task.getDimObservation());
        std::cout << "done" << std::endl;

        // Initialize critics
        std::cout << ".. Critics - ";
        Critic criticV(linearRegV);
        Critic criticU(linearRegU);
        std::cout << "done" << std::endl;

        // Boltzmann Policy
        std::cout << ".. Boltzmann stochastic policy - ";
        std::vector<double> possibleAction {-1.0, 1.0};
        BoltzmannPolicy policy(task.getDimObservation(), possibleAction);
        std::cout << "done" << std::endl;

        // Stochastic Actor
        std::cout << ".. Actor - ";
        StochasticActor actor(policy);
        std::cout << "done" << std::endl;

        // ARSSAC Agent
        std::cout << ".. ARRSAC Agent - ";
        agentPtr = std::make_shared<ARRSACAgent> (actor,
                                                  criticV,
                                                  criticU,
                                                  baselineLearningRate,
                                                  criticLearningRate,
                                                  actorLearningRate,
                                                  lambda);
        std::cout << "done" << std::endl;

    }
    else if (algorithm == "RSPGPE")
    {
        // State-value function critic
        std::cout << ".. Linear regressors - ";
        LinearRegressor linearRegV(task.getDimObservation());
        LinearRegressor linearRegU(task.getDimObservation());
        std::cout << "done" << std::endl;

        // Initialize critics
        std::cout << ".. Critics - ";
        Critic criticV(linearRegV);
        Critic criticU(linearRegU);
        std::cout << "done" << std::endl;

        // Binary policy
        std::cout << ".. PGPE binary policy - ";
        BinaryPolicy controller(task.getDimObservation());
        GaussianDistribution distribution(controller.getDimParameters());
        PGPEPolicy policy(controller, distribution, 1.0);
        std::cout << "done" << std::endl;

        // Stochastic Actor
        std::cout << ".. Actor - ";
        StochasticActor actor(policy);
        std::cout << "done" << std::endl;

        // ARSSAC Agent
        std::cout << ".. ARRSAC Agent - ";
        agentPtr = std::make_shared<ARRSACAgent> (actor,
                                                  criticV,
                                                  criticU,
                                                  baselineLearningRate,
                                                  criticLearningRate,
                                                  actorLearningRate,
                                                  lambda);
        std::cout << "done" << std::endl;

    }
    else if (algorithm == "RSNPGPE")
    {
        // Binary policy
        std::cout << ".. Policy - ";
        BinaryPolicy controller(task.getDimObservation());
        std::cout << "done" << std::endl;

        // NPGPE Agent
        std::cout << ".. NPGPE Agent - ";
        agentPtr = std::make_shared<RiskSensitiveNPGPEAgent> (controller,
                                                              baselineLearningRate,
                                                              actorLearningRate,
                                                              lambda);
        std::cout << "done" << std::endl;

    }
    else
    {
        throw std::invalid_argument("Unknown learning algorithm " + algorithm);
        return 1;
    }

    //----------------------------------|
    // 2.3) Asset Allocation Experiment |
    //----------------------------------|

    std::cout << ".. Asset allocation experiment - ";
    AssetAllocationExperiment experiment(task,
                                         *agentPtr,
                                         numExperiments,
                                         numEpochs,
                                         numTrainingSteps,
                                         numTestSteps,
                                         outputDir,
                                         debugDir);
    std::cout << "done" << std::endl;

    //-------------------|
    // 3) Run experiment |
    //-------------------|

    std::cout << std::endl << "2) Experiment" << std::endl;
    experiment.run();

	return 0;
}











