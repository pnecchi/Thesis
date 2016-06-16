#include "thesis/Statistics.h"

StatisticsAverage::StatisticsAverage()
    : runningSum(0.0), nResults(0)
{
    /* Nothing to do */
}

std::unique_ptr<Statistics> StatisticsAverage::clone() const
{
    return std::unique_ptr<Statistics>(new StatisticsAverage(*this));
}

void StatisticsAverage::dumpOneResult(double result)
{
    runningSum += result;
    nResults += 1;
}

std::vector<std::vector<double>> StatisticsAverage::getStatistics() const
{
    std::vector<std::vector<double>> results(1);
    results[0].resize(1);
    results[0][0] = runningSum / nResults;
    return results;
}

StatisticsEMA::StatisticsEMA(double learningRate_)
    : EMA(0.0),
      learningRate(learningRate_)
{
    /* Nothing to do */
}

std::unique_ptr<Statistics> StatisticsEMA::clone() const
{
    return std::unique_ptr<Statistics>(new StatisticsEMA(*this));
}

void StatisticsEMA::dumpOneResult(double result)
{
    // TODO: check for first result dumped
    EMA = (1.0 - learningRate) * EMA + learningRate * result;
}

std::vector<std::vector<double>> StatisticsEMA::getStatistics() const
{
    std::vector<std::vector<double>> results(1);
    results[0].resize(1);
    results[0][0] = EMA;
    return results;
}
