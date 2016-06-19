#ifndef BACKTESTLOG_H
#define BACKTESTLOG_H

#include <armadillo>
#include <ostream>

class BacktestLog
{
    friend std::ostream& operator<<(std::ostream &os, BacktestLog const &blog);

    public:
        // Constructor
        BacktestLog(size_t dimState_, size_t dimAction_, size_t numRecords_);

        // Copy constructor
        BacktestLog(BacktestLog const &other_) = default;

        // Destructor
        virtual ~BacktestLog() = default;

        // Add entry
        void insertRecord(arma::vec const &state_,
                          arma::vec const &action_,
                          double const reward_);

        // Print
        void save(std::string filename);

        // Reset
        void reset();

    private:
        arma::mat history;
        size_t dimState;
        size_t dimAction;
        size_t currentIdx;
};

std::ostream& operator<<(std::ostream &os, BacktestLog const &blog);

#endif // BACKTESTLOG_H
