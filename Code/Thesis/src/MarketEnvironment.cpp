#include <thesis/MarketEnvironment.h>
#include <armadillo>  /* arma::mat */
#include <string>     /* std::string */
#include <fstream>    /* std::ifstream */
#include <sstream>    /* std::istringstream */

MarketEnvironment::MarketEnvironment (std::string inputFilePath,
									  size_t startDate_,
									  size_t endDate_)
	: startDate(startDate_),
	  currentDate(startDate),
	  endDate(endDate_)
{
	// Initialize filestream from inputFilePath
	std::ifstream ifs(inputFilePath);
	std::string line;
    char ch;
	// pierpaolo - gio 09 giu 2016 09:57:32 CEST
	// TODO: Test file opening

	// Read number of days and number of risky assets from the first line
	if (getline(ifs, line))
	{
		std::istringstream linestream(line);
		linestream >> numDays >> ch >> numRiskyAssets;
	}

	// Read risky assets symbols from the second line
	if (getline(ifs, line))
	{
		std::istringstream linestream(line);
		std::string symbol;

		for(size_t i = 0; i < numRiskyAssets && linestream >> symbol; i++)
		{
            assetSymbols.push_back(symbol);
            if (linestream.peek() == ',')
                linestream.ignore();
		}
	}

	// Read risky assets log-returns in an armadillo matrix.
	// For faster slicing, the matrix is of size numRiskyAssets X numDays
	assetsReturns.set_size(numRiskyAssets, numDays);
	double oneReturn = 0.0;
	for(size_t i = 0; i < numDays && getline(ifs, line); ++i)
	{
		std::istringstream linestream(line);
		for(size_t j = 0; j < numRiskyAssets && linestream >> oneReturn; ++j)
		{
			assetsReturns(j, i) = oneReturn;

			if (linestream.peek() == ',')
                linestream.ignore();
		}
	}

	// Set dimensions of state and action spaces
	dimState = numRiskyAssets;
	dimAction = numRiskyAssets;
}

arma::vec MarketEnvironment::getState() const
{
	return assetsReturns.col(currentDate);
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
