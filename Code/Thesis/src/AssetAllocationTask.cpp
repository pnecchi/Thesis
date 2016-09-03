#include <thesis/AssetAllocationTask.h>
#include <math.h>      /* log */
#include <limits>      /* numeric_limits */

void AssetAllocationTask::initializeStatesCache()
{
	// Initialize past market states
	arma::vec proxyAction(environmentPtr->getDimAction());
	for(size_t i = 0; i < numDaysObserved; ++i)
	{
        // Get market state
        pastStates.rows(i * dimState, (i + 1) * dimState - 1) =
            environmentPtr->getState();

		// Move to the next time step
		environmentPtr->performAction(proxyAction);
	}

	// Initialize current market state
    currentState = environmentPtr->getState();
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
	: Task(market_),
	  riskFreeRate(riskFreeRate_),
	  deltaP(deltaP_),
	  deltaF(deltaF_),
	  deltaS(deltaS_),
	  numDaysObserved(numDaysObserved_)
{
	// Dimensions of observation and action spaces
	dimState = environmentPtr->getDimState();
	dimPastStates = numDaysObserved * dimState;
	dimObservation = 1 + dimPastStates + dimState + environmentPtr->getDimAction();

	// Initialize state cache variables
	pastStates.set_size(dimPastStates);
	currentState.set_size(dimState);
	initializeStatesCache();

	// Initialize allocation cache variables
	currentAllocation.set_size(environmentPtr->getDimAction());
	newAllocation.set_size(environmentPtr->getDimAction());
	initializeAllocationCache();
}

AssetAllocationTask::AssetAllocationTask(AssetAllocationTask const &other_)
    : Task(*other_.environmentPtr),
      riskFreeRate(other_.riskFreeRate),
      deltaP(other_.deltaP),
      deltaF(other_.deltaF),
      deltaS(other_.deltaS),
      numDaysObserved(other_.numDaysObserved),
      dimState(other_.dimState),
      dimPastStates(other_.dimPastStates),
      dimObservation(other_.dimObservation),
      pastStates(other_.pastStates),
      currentState(other_.currentState),
      currentAllocation(other_.currentAllocation),
      newAllocation(other_.newAllocation)
{
    /* Nothing to do */
}

std::unique_ptr<Task> AssetAllocationTask::clone() const
{
    return std::unique_ptr<Task>(new AssetAllocationTask(*this));
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
	environmentPtr->performAction(newAllocation);
}

double AssetAllocationTask::getReward () const
{
	// Update past states with current state
	if (numDaysObserved > 1)
        pastStates.rows(0, (numDaysObserved - 1) * dimState - 1) =
            pastStates.rows(dimState, pastStates.size() - 1);
	pastStates.rows((numDaysObserved - 1) * dimState, pastStates.size() - 1) =
		currentState;

	// Observe new market state
	currentState = environmentPtr->getState();

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
    environmentPtr->reset();
    initializeStatesCache();
    initializeAllocationCache();
}

void AssetAllocationTask::setEvaluationInterval (size_t startDate_,
												 size_t endDate_)
{
	MarketEnvironment* marketEvironmentPtr =
        dynamic_cast<MarketEnvironment*>(environmentPtr.get());
	marketEvironmentPtr->setEvaluationInterval(startDate_, endDate_);
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
