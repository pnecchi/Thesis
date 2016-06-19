#include "thesis/BacktestLog.h"

std::ostream& operator<<(std::ostream &os, BacktestLog const &blog)
{
    os << blog.history.t() << std::endl;
    return os;
}

BacktestLog::BacktestLog(size_t dimState_, size_t dimAction_, size_t numRecords_)
    : dimState(dimState_),
      dimAction(dimAction_),
      history(dimState_ + dimAction_ + 2, numRecords_, arma::fill::zeros),
      currentIdx(0ul)
{
    /* Nothing to do */
}

void BacktestLog::insertRecord(arma::vec const &state_,
                               arma::vec const &action_,
                               double const reward_)
{
    history(arma::span(0, state_.size() - 1), currentIdx) = state_;
    history(state_.size(), currentIdx) = 1.0 - arma::sum(action_);
    history(arma::span(state_.size() + 1, history.n_rows - 2), currentIdx) = action_;
    history(history.n_rows-1, currentIdx) = reward_;
    ++currentIdx;
}

void BacktestLog::save(std::string filename)
{
    // Open file
    std::ofstream backtestFile;
    backtestFile.open(filename);

    // Write header
    std::ostringstream headerStream;
    for (size_t n = 1; n < dimState + 1; ++n)
        headerStream << "r_" << n << ",";
    for (size_t n = 0; n < dimAction + 1; ++n)
        headerStream << "a_" << n << ",";
    headerStream << "logReturn";
    std::string header = headerStream.str();
    backtestFile << header << std::endl;

    // Save matrix
    arma::mat historyTransposed = trans(history);
    historyTransposed.save(backtestFile, arma::csv_ascii);

    // Close file
    backtestFile.close();
}

void BacktestLog::reset()
{
    history.zeros();
    currentIdx = 0ul;
}
