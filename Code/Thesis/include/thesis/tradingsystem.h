//----------------------------------------------------------------------
// Description: Trading system class
// Author:      Pierpaolo Necchi
// Email:       pierpaolo.necchi@gmail.com
// Date:        sab 11 giu 2016 12:15:35 CEST
//----------------------------------------------------------------------

#ifndef TRADINGSTSTEM_H
#define TRADINGSTSTEM_H

#include <armadillo>
#include <memory>
#include <thesis/agent.h>

class TradingSystem : public Agent
{
public:
    // Standard constructor
	TradingSystem(Agent const &agent_, bool backtestMode = false)
        : agentPtr(agent_.clone()) {}

	// Standard destructor
	virtual ~TradingSystem() = default;

    // backtestMode: get and set
	bool getBacktestMode () const { return backtestMode; }
    void setBacktestMode(bool backtestMode_) { backtestMode = backtestMode_; }

    // Receive observation from the system
	virtual void receiveObservation (arma::vec const &observation);

	// Get action to perform on the environment
	virtual arma::vec getAction() const;

    // Receive reward from the environment
	virtual void receiveReward(double const reward_);

    // Learning step given previous experience
	virtual void learn();

    // TODO: implement method to output statistics

private:
	// Agent (wrapped)
	std::unique_ptr<Agent> agentPtr;

	// Statistics gatherer
	// pierpaolo - sab 11 giu 2016 12:21:42 CEST
	// TODO: Implement statistics gatherer class (cf. Joshi)
	// std::unique_ptr<Statistics> backtestStatisticsPtr;

	// Backtest flag
	bool backtestMode = false;

	// Cache variables
	mutable arma::vec actionCache;
	mutable double rewardCache;
};

#endif /* end of include guard: TRADINGSTSTEM_H */
