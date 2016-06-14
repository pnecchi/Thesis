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
#include <thesis/Agent.h>
#include <thesis/BacktestLog.h>

class TradingSystem : public Agent
{
public:
    // Standard constructor
	TradingSystem(Agent const &agent_,
                  bool backtestMode=false,
                  size_t numRecords=0ul);

    // Copy constructor
    TradingSystem(TradingSystem const &other_);

	// Standard destructor
	virtual ~TradingSystem() = default;

	// Virtual polymorphic clone
	virtual std::unique_ptr<Agent> clone() const;

    // backtestMode: get and set
	bool getBacktestMode() const { return backtestMode; }
    void setBacktestMode(bool backtestMode_) { backtestMode = backtestMode_; }

    // Receive observation from the system
	virtual void receiveObservation (arma::vec const &observation_);

    // Get action size
    virtual size_t getDimAction() const { return agentPtr->getDimAction(); }

	// Get action to perform on the environment
	virtual arma::vec getAction();

    // Receive reward from the environment
	virtual void receiveReward(double const reward_);

	// Receive next observation --> O_{t+1}
	void receiveNextObservation(arma::vec const &nextObservation_);

    // Learning step given previous experience
	virtual void learn();

    // TODO: implement method to output statistics

private:
	// Agent (wrapped)
	std::unique_ptr<Agent> agentPtr;

	// Backtest
	bool backtestMode = false;
    BacktestLog blog;

	// Cache variables
	mutable arma::vec actionCache;
	mutable double rewardCache;
};

#endif /* end of include guard: TRADINGSTSTEM_H */
