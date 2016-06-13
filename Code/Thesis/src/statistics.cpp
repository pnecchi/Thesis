#include "thesis/statistics.h"

StatisticsAverage::StatisticsAverage()
    : runningSum(0.0), nResults(0)
{
    /* Nothing to do */
}

void std::unique_ptr<Statistics> StatisticsAverage::clone() const
{
    return std::unique_ptr<Statistics>(new StatisticsAverage(*this));
}

virtual void StatisticsAverage::dumpOneResult(double result)
{
    runningSum += result;
    nResults += 1;
}

virtual std::vector<std::vector<double>> getStatistics() const
{
    std::vector<std::vector<double>> results(1);
    results[0].resize(1);
    results[0][0] = runningSum / nResults;
    return results;
}

StatisticsEMA::StatisticsEMA(double decayRate_):
    : EMA(0.0), decayRate(decayRate_), learningRate(1.0 - decayRate), first(true)
{
    /* Nothing to do */
}

virtual void std::unique_ptr<Statistics> StatisticsEMA::clone() const
{
    return std::unique_ptr<Statistics>(new StatisticsEMA(*this));
}

virtual void StatisticsEMA::dumpOneResult(double result)
{
    if (first)
        EMA = result;
    else
        EMA = decayRate * EMA + learningRate * result;
}

virtual std::vector<std::vector<double>> StatisticsEMA::getStatistics() const
{
    std::vector<std::vector<double>> results(1);
    results[0].resize(1);
    results[0][0] = EMA;
    return results;
}
