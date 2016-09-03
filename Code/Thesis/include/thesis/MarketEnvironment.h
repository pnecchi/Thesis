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

#ifndef MARKETENVIRONMENT_H
#define MARKETENVIRONMENT_H

#include <thesis/Environment.h>
#include <armadillo>
#include <vector>
#include <string>

/**
 * MarketEnvironment implements a simple financial market environment in which
 * the asset prices evolve according to historical time series. The market
 * consists of I+1 different stocks that are traded only at discrete times
 * {0, 1, ... }. The 0-th asset is by assumption a risk-less asset whose price
 * grows at a risk-free rate. A trading system will interact with the market in
 * the asset allocation task.
 */

class MarketEnvironment : public Environment
{
    public:
        /**
         * Constructor.
         * Initialize the financial market reading the historical log-return
         * series from an input file.
         * \param inputFilePath path to the input file
         * \param startDate_ initial time step
         * \param endDate_ final time step
         */
        MarketEnvironment(std::string inputFilePath);

        //! Default copy constructor.
        MarketEnvironment(MarketEnvironment const &market_);

        //! Virtual destructor.
        virtual ~MarketEnvironment() = default;

        /**
         * Get system state.
         * The function returns a vector of the risky assets log-returns for the
         * current time step.
         * \return current time step risky assets log-returns.
         */
        virtual arma::vec getState() const;

        /**
         * Perform Action on the system.
         * Select a portfolio allocation for the I risky assets. The system
         * dynamics is independent from the portfolio allocation selected by the
         * agent.
         * \param action portfolio allocation
         */
        virtual void performAction(arma::vec const &action);

        //!Get assets ticker symbols.
        std::vector<std::string> getAssetsSymbols() const { return assetsSymbols; }

        //! Get total number of days in the time series.
        size_t getNumDays() const { return numDays; }

        //! Get number of risky assets available on the market
        size_t getNumRiskyAssets() const { return numRiskyAssets; }

        //! Get dimension of the state space.
        virtual size_t getDimState() const { return dimState; }

        //! Get dimension of the action space.
        virtual size_t getDimAction() const { return dimAction; }

        //! Get initial time step.
        size_t getStartDate() const { return startDate; }

        //! Get current time step.
        size_t getCurrentDate() const { return currentDate; }

        //! Get final time step
        size_t getEndDate() const { return endDate; }

        //! Set initial time step.
        void setStartDate(size_t startDate_) { startDate = startDate_; }

        //! Set final time step.
        void setEndDate(size_t endDate_) { endDate = endDate_; }

        //! Set evaluation interval.
        void setEvaluationInterval(size_t startDate_, size_t endDate_);

        //! Reset market environment to initial condition.
        virtual void reset();

    private:
        //! Asset ticker symbols.
        std::vector<std::string> assetsSymbols;

        /**
         * Log-return time series.
         * The matrix is of size numRiskyAssets X numDays for faster slicing.
         */
        arma::mat assetsReturns;

        //! Total number of time steps.
        size_t numDays;

        //! Number of risky assets in the market.
        size_t numRiskyAssets;

        //! State space dimension
        size_t dimState;

        //! Action space dimension
        size_t dimAction;

        //! Initial time step.
        size_t startDate;

        //! Current time step.
        size_t currentDate;

        //! Final time step.
        size_t endDate;
};

#endif /* end of include guard: MARKETENVIRONMENT_H */
