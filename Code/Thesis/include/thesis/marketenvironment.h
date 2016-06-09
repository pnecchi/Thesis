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
					   size_t nDaysObserved, 
					   double riskFreeRate, 
					   size_t startDate, 
					   size_t endDate);
	
	// Destructor
	virtual ~MarketEnvironment ();
	
	// Get system state
	arma::vec getState() const;
	
	// Perform Action on the system
	void performAction(arma::vec const &action);
	
	// Get methods
	size_t getDimState() const { return dimState; }
	size_t getDimAction() const { return dimAction; }
	size_t getNumDaysObserved() const { return numDaysObserved; }
	size_t getStartDate() const { return startDate; }
	size_t getCurrentDate() const { return currentDate; }
	size_t getEndDate() const { return endDate; }
	double getRiskFreeRate() const { return riskFreeRate; }
	
	// Set methods
	void setStartDate(size_t startDate_) { startDate = startDate_; }
	void setEndDate(size_t endDate_) { endDate = endDate_; }
	void SetEvaluationInterval(size_t startDate_, size_t EndDate_);

	// Reset
	void reset(); 

private:
	std::vector<std::string>> assetSymbols; 
	arma::mat logReturns;
	double riskFreeRate;
	size_t dimState;
	size_t dimAction;
	size_t numDaysObserved;
	size_t startDate;
	size_t currentDate;
	size_t endDate;	
};

#endif /* end of include guard: MARKETENVIRONMENT_H */
