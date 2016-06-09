#include <thesis/assetallocationtask.h>

AssetAllocationTask::AssetAllocationTask (MarketEnvironment const & market_, 
										  double deltaP_, 
										  double deltaF_, 
										  double deltaS_, 
										  size_t numDaysObserved_)
	: market(market_),
	  deltaP(deltaP_), 
	  deltaF(deltaF_),
	  deltaS(deltaS_),
	  numDaysObserved(numDaysObserved_)
{
	// Dimensions of observation and action spaces
	dimState = market.getDimState();
	dimAction = market.getDimAction();
	dimPastStates = numDaysObserved * dimState;
	dimObservation = dimPastStates + dimState + marketDimAction();
	
	// Initialize state cache variables 	
	pastStates.set_size(dimPastStates);
	currentState.set_size(dimState);
	(*this).initializePastStates();  
		
	// Initialize allocation cache variables
	currentAllocation.set_size(dimAction);
	(*this).initializeAllocation();
	newAllocation.set_size(dimAction);
}

AssetAllocationTask::initializePastStates()
{
	// Initialize past market states
	for(size_t i = 0; i < numDaysObserved; ++i)
	{
		pastState.rows(i * dimState, (i + 1) * dimState - 1) = market.getState()
		market.performAction()
	}

	// Initialize current market state
	currentState = market.getState();
}

AssetAllocationTask::initializeAllocation()
{
	currentAllocation.zeros();
	currentAllocation(0) = 1.0;
}

arma::vec AssetAllocationTask::getObservation () const
{
	arma::vec observation(dimObservation);

	// Copy past states
	observation.rows(0, dimPastStates-1) = pastStates;
	
	// Copy current states
	observation.rows(dimPastStates, dimPastStates + dimState - 1) = currentState;
	
	// Copy current allocation 
	observation.rows(dimPastStates + dimState, observation.size() - 1) = currentAllocation;
	
	return observation;
}

void AssetAllocationTask::performAction (arma::vec const &action)
{
	// Cache new allocation
	newAllocation = action;
	
	// Broadcast action to underlying environment
	market.performAction(action);
}

double AssetAllocationTask::getReward () const
{
	// Update past states with current state
	pastStates.rows(0, numDaysObserved * dimState - 1) = 
		pastState.rows(dimState, size(pastState));
	pastStates.rows(numDaysObserved * dimState, size(pastState)) = 
		currentState;

	// Observe new market state
	currentState = market.getState();

	// Compute portfolio return
	// pierpaolo - gio 09 giu 2016 16:47:48 CEST
	// TODO: Compute portfolio return
	
	double reward = 0.0;

	// Update allocation
	// pierpaolo - gio 09 giu 2016 16:48:22 CEST
	// TODO: Update portfolio allocation

	return reward;
}
	
void AssetAllocationTask::setEvaluationInterval (size_t startDate_,
												 size_t endDate_)
{
	market.setEvaluationInterval(startDate_, endDate_);
	(*this).initializePastStates();
	
	(*this).initializeAllocation();	
}

