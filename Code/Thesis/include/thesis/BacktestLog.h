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

#ifndef BACKTESTLOG_H
#define BACKTESTLOG_H

#include <armadillo>
#include <ostream>

/**
 * BacktestLog implements a simple data-structure that stores the relevant
 * information for the analysis of the backtest performances of the trading
 * strategy, i.e. the state of the system, the selected allocation and the
 * portfolio log-return.
 */

class BacktestLog
{
    friend std::ostream& operator<<(std::ostream &os, BacktestLog const &blog);

    public:
        /*!
         * Constructor
         * Initializes a backtest log given the sizes of the problem.
         * \param dimState_ size of the state space.
         * \param dimAction_ size of the action space.
         * \param numRecords_ number of records that will be stored.
         */
        BacktestLog(size_t dimState_, size_t dimAction_, size_t numRecords_);

        //! Copy constructor.
        BacktestLog(BacktestLog const &other_) = default;

        //! Destructor.
        virtual ~BacktestLog() = default;

        /*!
         * Insert new record in the log.
         * \param state_ system state.
         * \param action_ action selected by the agent.
         * \param reward_ portfolio log-return.
         */
        void insertRecord(arma::vec const &state_,
                          arma::vec const &action_,
                          double const reward_);

        /*!
         * Print the backtest log in a file at the given filepath.
         * \param filename path to the output file.
         */
        void save(std::string filename);

        //! Reset log to the initial state.
        void reset();

    private:
        //! Matrix storing the data column-wise.
        arma::mat history;

        //! Size of the state space.
        size_t dimState;

        //! Size of the action space.
        size_t dimAction;

        //! Current index.
        size_t currentIdx;
};

//! Output operator for the BacktestLog function.
std::ostream& operator<<(std::ostream &os, BacktestLog const &blog);

#endif // BACKTESTLOG_H
