#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>
#include <memory>

/**
 * Statistics is an abstract base class that implements a generic interface for
 * a statistics gatherer, e.g. average, exponential moving average, convergence
 * table, etc. It is taken from "Joshi, M. - C++ Design Patterns for Derivatives
 * Pricing (2008)".
 */

class Statistics
{
    public:
        Statistics() = default;
        virtual ~Statistics() = default;

        // Statistics is clonable to allow virtual copy construction
        virtual std::unique_ptr<Statistics> clone() const = 0;

        virtual void dumpOneResult(double result) = 0;
        virtual std::vector<std::vector<double>> getStatistics() const = 0;
};


class StatisticsAverage : public Statistics
{
    public:
        StatisticsAverage();
        virtual std::unique_ptr<Statistics> clone() const;
        virtual void dumpOneResult(double result);
        virtual std::vector<std::vector<double>> getStatistics() const;
    private:
        double runningSum;
        size_t nResults;
};

/**
 * Exponential Moving Average (EMA) class.
 */

class StatisticsEMA : public Statistics
{
    public:
        StatisticsEMA(double learningRate_);
        virtual std::unique_ptr<Statistics> clone() const;
        virtual void dumpOneResult(double result);
        virtual std::vector<std::vector<double>> getStatistics() const;
        double getLearningRate() const { return learningRate; }
    private:
        double EMA;
        double learningRate;
};


class StatisticsExperiment : public Statistics
{
    public:
        StatisticsExperiment() = default;
        virtual std::unique_ptr<Statistics> clone() const;
        virtual void dumpOneResult(double result);
        virtual std::vector<std::vector<double>> getStatistics() const;

    private:
        StatisticsAverage averageReward;
        StatisticsAverage averageSquareReward;
};


#endif // STATISTICS_H
