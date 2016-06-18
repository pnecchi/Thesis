#include "thesis/BacktestLog.h"

std::ostream& operator<<(std::ostream &os, BacktestLog const &blog)
{
    os << blog.history.t() << std::endl;
    return os;
}

void BacktestLog::insertRecord(arma::vec const &action_, double const reward_)
{
    history.col(currentIdx).rows(0, history.n_rows-2) = action_;
    history(currentIdx, history.n_rows-1) = reward_;
}

void BacktestLog::print(std::ostream& os)
{
    history.print(os);
}
