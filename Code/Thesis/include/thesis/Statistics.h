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
        //! Virtual destructor.
        virtual ~Statistics() = default;

        /**
         * Clone method.
         * Statistics is clonable to allow polymorphic copy.
         */
        virtual std::unique_ptr<Statistics> clone() const = 0;

        /**
         * Update the statistics with a new value.
         * \param result value to integrate in the statistics.
         */
        virtual void dumpOneResult(double result) = 0;

        /**
         * Return statistics computed by the gatherer.
         * \return a vector of vectors of statistics.
         */
        virtual std::vector<std::vector<double>> getStatistics() const = 0;

        //! Reset statistics gatherer to initial conditions.
        virtual void reset() = 0;
};

/**
 * StatisticAverage is a statistics gatherer responsible for computing an
 * arithmetic average.
 */

class StatisticsAverage : public Statistics
{
    public:
        //! Constructor.
        StatisticsAverage();

        //! Copy constructor.
        StatisticsAverage(StatisticsAverage const &other_) = default;

        //! Destructor.
        virtual ~StatisticsAverage() = default;

        //! Clone method.
        virtual std::unique_ptr<Statistics> clone() const;

        /**
         * Update the arithmetic average with a new value.
         * \param result value to integrate in the arithmetic average.
         */
        virtual void dumpOneResult(double result);

        /**
         * Compute arithmetic average.
         * \return arithmetic average.
         */
        virtual std::vector<std::vector<double>> getStatistics() const;

        //! Reset statistics gatherer to initial conditions.
        virtual void reset();

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
        /**
         * Constructor.
         * \param learningRate_ learning rate for EMA, i.e. 1 - decayRate
         */
        StatisticsEMA(double learningRate_);

        //! Copy constructor
        StatisticsEMA(StatisticsEMA const &other_) = default;

        //! Destructor
        virtual ~StatisticsEMA() = default;

        //! Clone method for polymorphic copy
        virtual std::unique_ptr<Statistics> clone() const;

        /**
         * Update the EMA with a new value.
         * \param result value to integrate in the EMA.
         */
        virtual void dumpOneResult(double result);

        /**
         * Compute EMA.
         * \return EMA.
         */
        virtual std::vector<std::vector<double>> getStatistics() const;

        //! Reset statistics gatherer to initial conditions.
        virtual void reset();

    private:
        double EMA;
        double learningRate;
        bool first;
};

/**
 * StatisticsExperiment gathers the relevant statistics for an asset allocation
 * experiment, i.e. average reward, reward standard deviation and Sharpe ratio.
 */

class StatisticsExperiment : public Statistics
{
    public:
        //! Constructor.
        StatisticsExperiment() = default;

        //! Copy constructor.
        StatisticsExperiment(StatisticsExperiment const &other_) = default;

        //! Destructor.
        virtual ~StatisticsExperiment() = default;

        //! Clone method for polymorphic copy.
        virtual std::unique_ptr<Statistics> clone() const;

        /**
         * Update statistics with a new observation of the reward.
         * \param result reward observation
         */
        virtual void dumpOneResult(double result);

        /**
         * Compute experiment statistics.
         * \return average reward, standard deviation and Sharpe ratio.
         */
        virtual std::vector<std::vector<double>> getStatistics() const;

        //! Reset statistics gatherer to initial conditions.
        virtual void reset();

    private:
        StatisticsAverage averageReward;
        StatisticsAverage averageSquareReward;
};


#endif // STATISTICS_H
