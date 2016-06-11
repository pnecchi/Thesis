#include <thesis/marketenvironment.h>
#include <armadillo>  /* arma::mat */
#include <string>     /* std::string */
#include <fstream>    /* std::ifstream */
#include <sstream>    /* std::istringstream */

MarketEnvironment::MarketEnvironment (std::string inputFilePath,
									  double riskFreeRate_,
									  size_t startDate_,
									  size_t endDate_)
	: riskFreeRate(riskFreeRate_),
	  startDate(startDate_),
	  currentDate(startDate),
	  endDate(endDate_)
{
	// Initialize filestream from inputFilePath
	std::ifstream ifs(inputFilePath);
	std::string line;

	// pierpaolo - gio 09 giu 2016 09:57:32 CEST
	// TODO: Test file opening

	// Read number of days and number of risky assets from the first line
	if (getline(ifs, line))
	{
		std::istringstream linestream(line);
		linestream >> numDays >> numRiskyAssets;
	}

	// Read risky assets symbols from the second line
	if (getline(ifs, line))
	{
		std::istringstream linestream(line);
		std::string symbol;
		for (size_t i = 0; i < numRiskyAssets; ++i)
		{
			linestream >> symbol;
			assetSymbols.push_back(symbol);
		}
	}

	// Read risky assets log-returns in an armadillo matrix.
	// For faster slicing, the matrix is of size numRiskyAssets X numDays
	assetsReturns.set_size(numRiskyAssets, numDays);
	double oneReturn = 0.0;
	for(size_t i = 0; i < numDays && getline(ifs, line); ++i)
	{
		std::istringstream linestream(line);
		for(size_t j = 0; j < numRiskyAssets; ++j)
		{
			linestream >> oneReturn;
			assetsReturns(j, i) = oneReturn;
		}
	}

	// Set dimensions of state and action spaces
	dimState = numRiskyAssets + 1;
	dimAction = numRiskyAssets + 1;
}

arma::vec MarketEnvironment::getState() const
{
	// pierpaolo - gio 09 giu 2016 10:28:19 CEST
	// TODO: To avoid copy, pass state vector as input by reference?
	arma::vec pastReturns(dimState);
	pastReturns(0) = riskFreeRate;
	pastReturns.rows(1, dimState-1) = arma::vectorise(assetsReturns.col(currentDate));
	return pastReturns;
}

void MarketEnvironment::performAction(arma::vec const &action)
{
	// The system evolution is independent from the agent's allocation
	currentDate++;
}

void MarketEnvironment::setEvaluationInterval(size_t startDate_,
											  size_t endDate_)
{
	setStartDate(startDate_);
	setEndDate(endDate_);
	reset();
}

void MarketEnvironment::reset()
{
	currentDate = startDate;
}
