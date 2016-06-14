//----------------------------------------------------------------------
// Description: Description
// Author:      Pierpaolo Necchi
// Email:       pierpaolo.necchi@gmail.com
// Date:        mer 08 giu 2016 15:17:35 CEST
//----------------------------------------------------------------------

#ifndef MARKETENVIRONMENT_H
#define MARKETENVIRONMENT_H

#include <armadillo>
#include <vector>
#include <string>

class MarketEnvironment
{
public:
	// Constructor
	MarketEnvironment (std::string inputFilePath,
					   double riskFreeRate_,
					   size_t startDate_,
					   size_t endDate_);
	MarketEnvironment (MarketEnvironment const &market_) = default;

	// Destructor
	virtual ~MarketEnvironment () = default;

	// Get system state
	arma::vec getState() const;

	// Perform Action on the system
	void performAction(arma::vec const &action);

	// Get methods
	std::vector<std::string> getAssetsSymbols() const { return assetSymbols; }
	double getRiskFreeRate() const { return riskFreeRate; }

	size_t getNumDays() const { return numDays; }
	size_t getNumRiskyAssets() const { return numRiskyAssets; }

	size_t getDimState() const { return dimState; }
	size_t getDimAction() const { return dimAction; }

	size_t getStartDate() const { return startDate; }
	size_t getCurrentDate() const { return currentDate; }
	size_t getEndDate() const { return endDate; }

	// Set methods
	void setStartDate(size_t startDate_) { startDate = startDate_; }
	void setEndDate(size_t endDate_) { endDate = endDate_; }
	void setEvaluationInterval(size_t startDate_, size_t endDate_);

	// Reset
	void reset();

private:
	// Assets
	std::vector<std::string> assetSymbols;
	arma::mat assetsReturns;
	double riskFreeRate;

	// Sizes
	size_t numDays;
	size_t numRiskyAssets;

	// Dimensions of state and action spaces
	size_t dimState;
	size_t dimAction;

	// Evaluation time interval
	size_t startDate;
	size_t currentDate;
	size_t endDate;
};

#endif /* end of include guard: MARKETENVIRONMENT_H */
