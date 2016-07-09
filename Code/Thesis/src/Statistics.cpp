#include "thesis/Statistics.h"
#include <math.h>

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

void StatisticsAverage::reset()
{
    runningSum = 0.0;
    nResults = 0;
}

StatisticsEMA::StatisticsEMA(double learningRate_)
    : EMA(0.0),
      learningRate(learningRate_),
      first(true)
{
    /* Nothing to do */
}

std::unique_ptr<Statistics> StatisticsEMA::clone() const
{
    return std::unique_ptr<Statistics>(new StatisticsEMA(*this));
}

void StatisticsEMA::dumpOneResult(double result)
{
    if (first)
    {
        EMA = result;
        first = false;
    }
    else
        EMA = (1.0 - learningRate) * EMA + learningRate * result;
}

std::vector<std::vector<double>> StatisticsEMA::getStatistics() const
{
    std::vector<std::vector<double>> results(1);
    results[0].resize(1);
    results[0][0] = EMA;
    return results;
}

void StatisticsEMA::reset()
{
    EMA = 0.0;
}

std::unique_ptr<Statistics> StatisticsExperiment::clone() const
{
    return std::unique_ptr<Statistics>(new StatisticsExperiment(*this));
}

void StatisticsExperiment::dumpOneResult(double result)
{
    averageReward.dumpOneResult(result);
    averageSquareReward.dumpOneResult(result * result);
}

std::vector<std::vector<double>> StatisticsExperiment::getStatistics() const
{
    std::vector<std::vector<double>> result(1);
    result[0].resize(3);

    // Average
    result[0][0] = averageReward.getStatistics()[0][0];

    // Standard deviation
    result[0][1] = sqrt(averageSquareReward.getStatistics()[0][0] -
                        result[0][0] * result[0][0]);

    // Sharpe ratio
    result[0][2] = result[0][0] / result[0][1];
    return result;
}


void StatisticsExperiment::reset()
{
    averageReward.reset();
    averageSquareReward.reset();
}
