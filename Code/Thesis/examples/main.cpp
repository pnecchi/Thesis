#include <iostream>
#include <string>
#include <armadillo>
#include <thesis/MarketEnvironment.h>
#include <thesis/AssetAllocationTask.h>
#include <thesis/LinearRegressor.h>

int main(int argc, char *argv[])
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "-        Algorithmic Asset Allocation        -" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

	// Market parameters
	std::string inputFilePath = "../../../Data/Input/synthetic.csv";
	double riskFreeRate = 0.0;
	size_t startDate = 0;
	size_t endDate = 10;

	// Task parameters
	double deltaP = 0.0005;
	double deltaF = 0.0;
	double deltaS = 0.0;
	size_t numDaysObserved = 2;

	// Market
	std::cout << ">> Initialize market environment" << std::endl;
	MarketEnvironment market(inputFilePath, riskFreeRate, startDate, endDate);

    // Asset allocation task
    std::cout << ">> Initialize asset allocation task" << std::endl;
	AssetAllocationTask task(market, deltaP, deltaF, deltaP, numDaysObserved);

	// State-value function critic
	std::cout << ">> Initialize state-value function critic" << std::endl;
	LinearRegressor criticV(task.getDimObservation());



    arma::vec observation(task.getDimObservation());
    arma::vec action(task.getDimAction());
    action.zeros();
    action(1) = 1.0;
    double reward;

    std::cout << ">> 1st interaction" << std::endl;
    std::cout << "Observation: " << std::endl;
    observation = task.getObservation();
    observation.print(std::cout);
    std::cout << "Action:" << std::endl;
    action.print(std::cout);
    task.performAction(action);
    reward = task.getReward();
    std::cout << "Reward: " << reward << std::endl;
    std::cout << "Critic: " << criticV.evaluate(observation) << std::endl;

    std::cout << ">> 2nd interaction" << std::endl;
    std::cout << "Observation: " << std::endl;
    observation = task.getObservation();
    observation.print(std::cout);
    std::cout << "Action:" << std::endl;
    action.print(std::cout);
    task.performAction(action);
    reward = task.getReward();
    std::cout << "Reward: " << reward << std::endl;
    std::cout << "Critic: " << criticV.evaluate(observation) << std::endl;

    std::cout << "Critic parameters:" << std::endl;
    arma::vec parameters = criticV.getParameters();
    parameters.print(std::cout);
    parameters(0) *= 2.0;
    std::cout << "New parameters:" << std::endl;
    parameters.print(std::cout);
    criticV.setParameters(parameters);
    parameters = criticV.getParameters();
    std::cout << "New Critic parameters:" << std::endl;
    parameters.print(std::cout);

    arma::vec gradient = criticV.gradient(observation);
    std::cout << "Critic gradient" << std::endl;
    gradient.print(std::cout);

    std::cout << "Critic parameters size: " << criticV.getDimParameters();

	return 0;
}
