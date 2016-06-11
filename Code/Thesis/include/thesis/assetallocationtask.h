//----------------------------------------------------------------------
// Description: Class for the asset allocation task.
// Author:      Pierpaolo Necchi
// Email:       pierpaolo.necchi@gmail.com
// Date:        gio 09 giu 2016 12:33:59 CEST
//----------------------------------------------------------------------

#ifndef ASSETALLOCATIONTASK_H
#define ASSETALLOCATIONTASK_H

#include <thesis/marketenvironment.h>
#include <armadillo>

class AssetAllocationTask
{
public:
	// Constructor
	AssetAllocationTask (MarketEnvironment const & market_,
						 double deltaP_,
						 double deltaF_,
						 double deltaS_,
						 size_t numDaysObserved_);

	// Destructor
	virtual ~AssetAllocationTask () = default;

	// Initialization helper functions
	void initializeStatesCache ();
	void initializeAllocationCache ();

	// Get methods
	double getDeltaP () const { return deltaP; }
	double getDeltaF () const { return deltaF; }
	double getDeltaS () const { return deltaS; }
    size_t getNumDaysObserved () const { return numDaysObserved; }
	size_t getDimObservation () const { return dimObservation; }
	size_t getDimAction () const { return dimAction; }

	// Provide state observation
	arma::vec getObservation () const;

	// Perform action
	void performAction (arma::vec const &action);

	// Provide reward
	double getReward ();

	// Set evaluation interval for the allocation task
	void setEvaluationInterval (size_t startDate_, size_t endDate_);

private:
	// Compute portfolio simple returns
	double computePortfolioSimpleReturn () const;

	// Underlying market environment
	MarketEnvironment market;

	// Transaction costs constants
	double deltaP;
	double deltaF;
	double deltaS;

	// Number of past days observed
	size_t numDaysObserved;

	// Dimensions of the observation and action space
	size_t dimState;
	size_t dimPastStates;
	size_t dimObservation;
	size_t dimAction;

	// Current state cache variable
	arma::vec pastStates;
	arma::vec currentState;

	// Allocations cache variable
	arma::vec currentAllocation;
	arma::vec newAllocation;
};

#endif /* end of include guard: ASSETALLOCATIONTASK_H */
