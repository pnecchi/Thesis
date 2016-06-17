#include <iostream>
#include <string>
#include <armadillo>
#include <thesis/ExperimentParameters.h>
#include <thesis/MarketEnvironment.h>
#include <thesis/AssetAllocationTask.h>
#include <thesis/LinearRegressor.h>
#include <thesis/Critic.h>
#include <thesis/BoltzmannExplorationPolicy.h>
#include <thesis/StochasticActor.h>
#include <thesis/ArrsacAgent.h>
#include <thesis/AssetAllocationExperiment.h>

int main()
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "-        Algorithmic Asset Allocation        -" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    // 0) Parameters
    std::cout << "0) Read parameters" << std::endl;
    std::string parametersFilepath  = "../../../Data/Parameters/ParametersARRSAC.pot";
    const ExperimentParameters params(parametersFilepath, true);

    // Copy parameters
    std::string inputDataPath = params.inputDataPath;
    std::string outputDataPath = params.outputDataPath;
    std::string debugDataPath = params.debugDataPath;
    double riskFreeRate = params.riskFreeRate;
    double deltaP = params.deltaP;
    double deltaF = params.deltaF;
    double deltaS = params.deltaS;
    size_t numDaysObserved = params.numDaysObserved;
    double alphaActor = params.alphaActor;
    double alphaCritic = params.alphaCritic;
    double alphaBaseline = params.alphaBaseline;
    size_t numExperiments = params.numExperiments;
    size_t numEpochs = params.numEpochs;
    size_t numTrainingSteps = params.numTrainingSteps;
    size_t numTestSteps = params.numTestSteps;

    // 1) Initialization
    std::cout << std::endl << "1) Initialization" << std::endl;

	// Market
	std::cout << ".. Market environment - ";
	size_t startDate = 0;
	size_t endDate = numTrainingSteps;
	MarketEnvironment market(inputDataPath, riskFreeRate, startDate, endDate);
    std::cout << "done" << std::endl;

    // Asset allocation task
    std::cout << ".. Asset allocation task - ";
	AssetAllocationTask task(market, deltaP, deltaF, deltaP, numDaysObserved);
    std::cout << "done" << std::endl;

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

    // Boltzmann Exploration Policy
    std::cout << ".. Boltzmann stochastic policy - ";
    std::vector<double> possibleAction {-1.0, 0.0, 1.0};
    BoltzmannExplorationPolicy policy(task.getDimObservation(), possibleAction);
    std::cout << "done" << std::endl;

    // Stochastic Actor
    std::cout << ".. Actor - ";
    StochasticActor actor(policy);
    std::cout << "done" << std::endl;

    // ARSSAC Agent
    std::cout << ".. ARRSAC Agent - ";
    ARRSACAgent agent(actor,
                      criticV,
                      criticU,
                      alphaActor,
                      alphaCritic,
                      alphaBaseline);
    std::cout << "done" << std::endl;

    // Asset allocation experiment
    std::cout << ".. Asset allocation experiment - ";
    AssetAllocationExperiment experiment(task, agent);
    std::cout << "done" << std::endl;

    // 2) Run experiment
    std::cout << std::endl << "2) Experiment" << std::endl;
    experiment.run(numTrainingSteps);

	return 0;
}
