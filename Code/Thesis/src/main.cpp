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
	arma::vec state = market.getState();
	state.print(std::cout);

	arma::vec action(market.getDimAction());
	market.performAction(action);

	std::cout << "Evaluation dates: " 
			  << market.getStartDate() << ", " 
			  << market.getCurrentDate() << ", "
			  << market.getEndDate() <<  std::endl;

	std::cout << "Current state: " << std::endl;  
	state = market.getState();
	state.print(std::cout);

	// Asset allocation task
	AssetAllocationTask task(market, deltaP, deltaF, deltaP, numDaysObserved);

	return 0;
}
