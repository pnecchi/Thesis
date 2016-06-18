#include <thesis/AssetAllocationTask.h>
#include <armadillo>
#include <math.h>      /* log */
#include <limits>      /* numeric_limits */

void AssetAllocationTask::initializeStatesCache()
{
	// Initialize past market states
	arma::vec proxyAction(dimAction);
	for(size_t i = 0; i < numDaysObserved; ++i)
	{
        // Get market state
        pastStates.rows(i * dimState, (i + 1) * dimState - 1) = market.getState();

		// Move to the next time step
		market.performAction(proxyAction);
	}

	// Initialize current market state
    currentState = market.getState();
}

void AssetAllocationTask::initializeAllocationCache()
{
	currentAllocation.zeros();
}

AssetAllocationTask::AssetAllocationTask (MarketEnvironment const & market_,
                                          double riskFreeRate_,
										  double deltaP_,
										  double deltaF_,
										  double deltaS_,
										  size_t numDaysObserved_)
	: market(market_),
	  riskFreeRate(riskFreeRate_),
	  deltaP(deltaP_),
	  deltaF(deltaF_),
	  deltaS(deltaS_),
	  numDaysObserved(numDaysObserved_)
{
	// Dimensions of observation and action spaces
	dimState = market.getDimState();
	dimAction = market.getDimAction();
	dimPastStates = numDaysObserved * dimState;
	dimObservation = 1 + dimPastStates + dimState + dimAction;

	// Initialize state cache variables
	pastStates.set_size(dimPastStates);
	currentState.set_size(dimState);
	initializeStatesCache();

	// Initialize allocation cache variables
	currentAllocation.set_size(dimAction);
	newAllocation.set_size(dimAction);
	initializeAllocationCache();
}

arma::vec AssetAllocationTask::getObservation () const
{
	arma::vec observation(dimObservation);

    // Risk-free rate
    observation(0) = riskFreeRate;

	// Past states
	observation.rows(1, dimPastStates) = pastStates;

	// Current state
	observation.rows(dimPastStates + 1, dimPastStates + dimState) = currentState;

	// Current allocation
	observation.rows(dimPastStates + dimState + 1, observation.size() - 1) = currentAllocation;

	return observation;
}

void AssetAllocationTask::performAction (arma::vec const &action)
{
	// Cache new allocation
	newAllocation = action;

	// Broadcast action to underlying environment
	market.performAction(newAllocation);
}

double AssetAllocationTask::getReward ()
{
	// Update past states with current state
	pastStates.rows(0, (numDaysObserved - 1) * dimState - 1) =
		pastStates.rows(dimState, pastStates.size() - 1);
	pastStates.rows((numDaysObserved - 1) * dimState, pastStates.size() - 1) =
		currentState;

	// Observe new market state
	currentState = market.getState();

	// Compute portfolio simple return
	double portfolioSimpleReturn = computePortfolioSimpleReturn();

	// Update allocation weights
	currentAllocation = newAllocation % (1.0 + currentState) /
						(1.0 + portfolioSimpleReturn);

	// Return portfolio log-return
	return log(1.0 + portfolioSimpleReturn);
}

void AssetAllocationTask::reset()
{
    market.reset();
    initializeStatesCache();
    initializeAllocationCache();
}

void AssetAllocationTask::setEvaluationInterval (size_t startDate_,
												 size_t endDate_)
{
	market.setEvaluationInterval(startDate_, endDate_);
    reset();
}

double AssetAllocationTask::computePortfolioSimpleReturn () const
{
	// Proportional transaction costs
	double proportionTransactionCosts = deltaP *
		arma::sum(arma::abs(newAllocation - currentAllocation));

	// Fixed transaction costs
	double fixedTransactionCosts = 	deltaF *
		(!arma::approx_equal(currentAllocation, newAllocation, "absdiff",
							 std::numeric_limits<double>::epsilon()));

	// Short-selling fees
	double shortPositionsWeight = 0.0;
	for(size_t i = 0; i < newAllocation.size(); ++i)
		if (newAllocation(i) < 0.0)
			shortPositionsWeight += - newAllocation(i);
	double shortTransactionCosts = deltaS * shortPositionsWeight;

	// Trading profit & loss
	double tradingPL = riskFreeRate +
                       arma::dot(newAllocation, currentState - riskFreeRate);

	// Compute simple portfolio return
	double portfolioSimpleReturn = tradingPL
								 - proportionTransactionCosts
								 - fixedTransactionCosts
								 - shortTransactionCosts;
	return portfolioSimpleReturn;
}
