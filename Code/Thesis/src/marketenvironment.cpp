#include <thesis/marketenvironment.h>

MarketEnvironment::MarketEnvironment (std::string inputFilePath, 
									  size_t nDaysObserved_, 
									  double riskFreeRate_, 
									  size_t startDate_, 
									  size_t endDate_)
	: nDaysObserved(nDaysObserved_), 
	  riskFreeRate(riskFreeRate_), 
	  startDate(startDate_),
	  endDate(endDate_)
{
	// pierpaolo - mer 08 giu 2016 16:02:20 CEST
	// TODO: Read log-returns from file 	
}
		
arma::vec getState() const
{
	// pierpaolo - mer 08 giu 2016 16:03:19 CEST
	// TODO: Extract last P+1 days and returns them
}
	
void performAction(arma::vec const &action)
{
	// The system evolution is independent from the agent's allocation
	currentDate++;
}

void MarketEnvironment::SetEvaluationInterval(size_t startDate_, 
											  size_t EndDate_)
{
	(*this).setStartDate(startDate_);
	(*this).setEndDate(endDate_);
	(*this).reset();
}

void MarketEnvironment::reset() 
{
	currentDate = (startDate > numDaysObserved) ? startDate : numDaysObserved;
}
