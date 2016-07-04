#include <thesis/MarketEnvironment.h>
#include <fstream>    /* std::ifstream */
#include <sstream>    /* std::istringstream */

MarketEnvironment::MarketEnvironment (std::string inputFilePath)
{
	// Initialize filestream from inputFilePath
	std::ifstream ifs(inputFilePath);
	std::string line;
    char ch;

    // Check file opening
    if (!ifs)
    {
        std::cerr << "Error: input file doesn't exist";
    }

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

    // Set time steps
    startDate = 0;
    currentDate = startDate;
    endDate = numDays - 1;
}

arma::vec MarketEnvironment::getState() const
{
	return assetsReturns.col(currentDate);
}

void MarketEnvironment::performAction(arma::vec const &action)
{
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
