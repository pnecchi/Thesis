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
#include <thesis/learningagent.h>

class TradingSystem : public LearningAgent
{
public:
    // Standard constructor
	TradingSystem(std::unique_ptr<LearningAgent> const &learningAgentPtr_,
                  std::unique_ptr<Statistics> const &backtestStatistics_,
                  bool backtestMode = false);

	// Standard destructor
	virtual ~TradingSystem(){}

    // backtestMode: get and set
	bool getBacktestMode () const { return backtestMode; }
    void setBacktestMode(bool backtestMode_) { backtestMode = backtestMode_; }

    // Receive observation from the system
	virtual void receiveObservation (arma::vec const &observation_);

	// Get action to perform on the environment
	virtual void getAction (arma::vec &action_);

    // Receive reward from the environment
	virtual void receiveReward (double reward_);

    // Learning step given previous experience
	virtual void learn();

    // TODO: implement method to output statistics

private:
	// Learning agent (wrapped)
	std::unique_ptr<LearningAgent> learningAgentPtr;

	// Statistics gatherer
	// pierpaolo - sab 11 giu 2016 12:21:42 CEST
	// TODO: Implement statistics gatherer class (cf. Joshi)
	std::unique_ptr<Statistics> backtestStatisticsPtr;

	// Backtest flag
	bool backtestMode = false;
};

#endif /* end of include guard: TRADINGSTSTEM_H */
