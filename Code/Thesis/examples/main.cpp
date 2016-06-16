#include <iostream>
#include <string>
#include <armadillo>
#include <thesis/MarketEnvironment.h>
#include <thesis/AssetAllocationTask.h>
#include <thesis/LinearRegressor.h>
#include <thesis/Critic.h>
#include <thesis/BoltzmannExplorationPolicy.h>
#include <thesis/StochasticActor.h>
#include <thesis/ArrsacAgent.h>
#include <thesis/AssetAllocationExperiment.h>

int main(int argc, char *argv[])
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "-        Algorithmic Asset Allocation        -" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    // 0) Parameters
    std::cout << "0) Read parameters" << std::endl;
    // TODO: read parameters from file

	// Market parameters
	std::string inputFilePath = "../../../Data/Input/synthetic.csv";
	double riskFreeRate = 0.0;
	size_t startDate = 0;
	size_t endDate = 10;

	// Task parameters
	double deltaP = 0.0005;
	double deltaF = 0.0;
	double deltaS = 0.0;
	size_t numDaysObserved = 5;

	// Agent parameters
	double alphaBaseline = 0.05;
	double alphaCritic = 0.01;
	double alphaActor = 0.005;

	// Experiment parameters
	size_t numSteps = 5000;

    // 1) Initialization
    std::cout << "1) Initialization" << std::endl;

	// Market
	std::cout << ".. Market environment - ";
	MarketEnvironment market(inputFilePath, riskFreeRate, startDate, endDate);
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
    std::cout << "2) Experiment" << std::endl;
    experiment.run(numSteps);

	return 0;
}
