#ifndef BACKTESTLOG_H
#define BACKTESTLOG_H

#include <armadillo>
#include <ostream>

class BacktestLog
{
    friend std::ostream& operator<<(std::ostream &os, BacktestLog const &blog);

    public:
        // Constructor
        BacktestLog(size_t dimAction, size_t numRecords):
            history(dimAction, numRecords), currentIdx(0ul) {}

        // Copy constructor
        BacktestLog(BacktestLog const &other_) = default;

        // Destructor
        virtual ~BacktestLog() = default;

        // Get number of records
        size_t getNumRecords() const { return history.n_cols; }

        // Add entry
        void insertRecord(arma::vec const &action_, double const reward_);

    private:
        arma::mat history;
        size_t currentIdx;
};

std::ostream& operator<<(std::ostream &os, BacktestLog const &blog);

#endif // BACKTESTLOG_H
