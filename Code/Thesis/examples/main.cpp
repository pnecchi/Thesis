#include <iostream>
#include <string>
#include <armadillo>
#include <thesis/marketenvironment.h>
#include <thesis/assetallocationtask.h>

int main(int argc, char *argv[])
{
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
	MarketEnvironment market(inputFilePath,
							 riskFreeRate,
							 startDate,
							 endDate);


	std::cout << "Number of days: " << market.getNumDays() << std::endl;
	std::cout << "Number of risky assets: " << market.getNumRiskyAssets() << std::endl;

	std::cout << "Assets symbols: ";
	for (auto s : market.getAssetsSymbols())
	{
		std::cout << s << ", ";
	}
	std::cout << std::endl;

	std::cout << "Evaluation dates: "
			  << market.getStartDate() << ", "
			  << market.getCurrentDate() << ", "
			  << market.getEndDate() <<  std::endl;

	std::cout << "Current state: " << std::endl;
	arma::vec state(market.getDimState());
    market.getState(state);
	state.print(std::cout);

	arma::vec action(market.getDimAction());
	action(1) = 1.0;
	market.performAction(action);

	std::cout << "Evaluation dates: "
			  << market.getStartDate() << ", "
			  << market.getCurrentDate() << ", "
			  << market.getEndDate() <<  std::endl;

	std::cout << "Current state: " << std::endl;
	market.getState(state);
	state.print(std::cout);

	// Asset allocation task
	AssetAllocationTask task(market, deltaP, deltaF, deltaP, numDaysObserved);

	std::cout << "Number of days observed: " << task.getNumDaysObserved() << std::endl;
    std::cout << "Observation size: " << task.getDimObservation() << std::endl;
    std::cout << "Action size: " << task.getDimAction() << std::endl;

    std::cout << "Observation: " << std::endl;
    arma::vec observation(task.getDimObservation());
    task.getObservation(observation);
    observation.print(std::cout);

    task.performAction(action);

    std::cout << "Reward: ";
    std::cout << task.getReward() << std::endl;

    std::cout << "Observation: " << std::endl;
    task.getObservation(observation);
    observation.print(std::cout);


	return 0;
}
