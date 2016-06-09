#include <iostream>
#include <string>
#include <thesis/marketenvironment.h>
#include <armadillo>

int main(int argc, char *argv[])
{
	// Parameters 
	std::string inputFilePath = "../../../Data/Input/synthetic.csv";
	double riskFreeRate = 0.0;
	size_t numDaysObserved = 2;
	size_t startDate = 0;
	size_t endDate = 10;
	
	// Market 
	MarketEnvironment market(inputFilePath, 
							 riskFreeRate, 
							 numDaysObserved,
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

	return 0;
}
