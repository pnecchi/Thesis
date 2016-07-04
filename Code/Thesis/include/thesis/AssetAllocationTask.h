/*
 * Copyright (c) 2016 Pierpaolo Necchi
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ASSETALLOCATIONTASK_H
#define ASSETALLOCATIONTASK_H

#include <thesis/MarketEnvironment.h>
#include <armadillo>

/**
 * AssetAllocationTask implements the asset allocation task performed by an
 * investor in a financial market. It provides the interface between the market
 * environment and an agent, acting as a filter between the system state and the
 * observation received by the agent, which may be smaller (partially observable
 * setting) or larger (state augmentation).
 */

// TODO: Extract more complex features from the risky assets past log-returns.
// Write class responsible for feature engineering, e.g. technical indicators,
// deep auto-encoder, deep neural network for predicting risky-asset returns.

class AssetAllocationTask
{
    public:
        /**
         * Constructor.
         * Initialize an asset allocation task given the underlying market and
         * other task-related parameters.
         * \param market_ financial market environment
         * \param riskFreeRate_ risk-free rate available on the market
         * \param deltaP_ proportional transaction costs
         * \param deltaF_ fixed transaction costs
         * \param deltaS_ short-selling fees
         * \param numDaysObserved_ nb of days observed by the agent (today excl)
         */
        AssetAllocationTask (MarketEnvironment const & market_,
                             double riskFreeRate_,
                             double deltaP_,
                             double deltaF_,
                             double deltaS_,
                             size_t numDaysObserved_);

        //! Copy constructor.
        AssetAllocationTask(AssetAllocationTask const &other_) = default;

        //! Destructor.
        virtual ~AssetAllocationTask () = default;

        //! Get market risk-free rate.
        double getRiskFreeRate() const { return riskFreeRate; }

        //! Get proportional transaction cost fee.
        double getDeltaP () const { return deltaP; }

        //! Get fixed transaction cost fee.
        double getDeltaF () const { return deltaF; }

        //! Get short-selling fee.
        double getDeltaS () const { return deltaS; }

        //! Get number of days observed by the agent.
        size_t getNumDaysObserved () const { return numDaysObserved; }

        //! Get observation space size.
        size_t getDimObservation () const { return dimObservation; }

        //! Get action space size.
        size_t getDimAction () const { return dimAction; }

        /**
         * Provide state observation.
         * The agent observes the past numDaysObserved log-returns of the risky
         * assets, the risk-free rate and the current allocation.
         * \return observation of the system state.
         */
        arma::vec getObservation () const;

        /**
         * Perform action.
         * Select new portfolio allocation on the risky assets. It is assumed
         * that the portfolio weight on the risk-free asset is 1 - sum(u_i).
         * \param action portfolio allocation.
         */
        void performAction (arma::vec const &action);

        /**
         * Provide reward.
         * The agent receives the log-return of his portfolio as a feedback.
         * \return portfolio log-return
         */
        double getReward ();

        //! Reset asset allocation task to initial condition.
        void reset();

        //! Set evaluation interval for the allocation task
        void setEvaluationInterval (size_t startDate_, size_t endDate_);

    private:
        //-----------------//
        // Private Methods //
        //-----------------//

        //! Initialize state cache vector with the past log-returns.
        void initializeStatesCache ();

        /**
         * Initialize allocation cache vector.
         * The entire capital is initially invested in the risk-free asset.
         */
        void initializeAllocationCache ();

        /**
         * Compute the simple return for the portfolio allocation selected.
         * \return portfolio simple return.
         */
        double computePortfolioSimpleReturn () const;

        //-----------------//
        // Private Members //
        //-----------------//

        //! Underlying market environment.
        MarketEnvironment market;

        //! Risk-free rate.
        double riskFreeRate;

        //! Proportional transaction costs.
        double deltaP;

        //! Fixed transaction costs.
        double deltaF;

        //! Short-selling fees.
        double deltaS;

        //! Number of past days observed.
        size_t numDaysObserved;

        //! State space size.
        size_t dimState;

        //! Past states size.
        size_t dimPastStates;

        //! Observation space size.
        size_t dimObservation;

        //! Action space size.
        size_t dimAction;

        //! Past states cache vector.
        arma::vec pastStates;

        //! Current state cache vector.
        arma::vec currentState;

        //! Current allocations cache vector.
        arma::vec currentAllocation;

        //! New allocation cache vector.
        arma::vec newAllocation;
};

#endif /* end of include guard: ASSETALLOCATIONTASK_H */
