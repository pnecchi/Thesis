#ifndef LEARNINGRATE_H
#define LEARNINGRATE_H

#include <math.h>  /* pow */

/**
 * LearningRate is an abstract class which implements a generic interface for a
 * learning rate which can be used in an iterative optimization procedure, e.g.
 * the gradient descent algorithm.
 */

class LearningRate
{
    public:
        //! Constructor.
        LearningRate() = default;

        //! Destructor.
        virtual ~LearningRate() = default;

        /**
         * Get learning rate current value.
         * \return learning rate
         */
        virtual double get() const = 0;

        /**
         * Update learning rate according to a certain schedule.
         */
        virtual double update() = 0;
};

/**
 * ConstantLearningRate implements a constant learning rate which can be used in
 * an iterative optimization procedure, e.g. the gradient descent algorithm.
 */

class ConstantLearningRate : public LearningRate
{
    public:
        /**
         * Constructor.
         * \param learningRate_ constant learning rate value.
         */
        ConstantLearningRate(double const learningRate_=0.1)
            : LearningRate(learningRate_) {}

        //! Destructor
        virtual ~ConstantLearningRate() = default;

        /**
         * Get learning rate current value.
         * \return learning rate
         */
        virtual double get() const { return learningRate; }

        /**
         * Update learning rate according to a certain schedule.
         */
        virtual double update() { /* Nothing to do */ }

    private:
        double learningRate;
};

/**
 * DecayingLearningRate implements an exponentially decaying learning of the
 * following form: alpha_k = C / n^exp. It can be used in an iterative
 * optimization procedure, e.g. the gradient descent algorithm. The parameters
 * should be chosen so as to satisfy the Robbins-Monro conditions.
 */

class DecayingLearningRate : public LearningRate
{
    public:
        /**
         * Constructor.
         * \param learningRate_ constant learning rate value.
         */
        DecayingLearningRate(double const c_=1.0, double const decayExp_=1.0)
            : c(c_), decayExp(decayExp_) {}

        //! Destructor
        virtual ~DecayingLearningRate() = default;

        /**
         * Get learning rate current value.
         * \return learning rate
         */
        virtual double get() const { return learningRate; }

        /**
         * Update learning rate according to a power decay schedule.
         */
        virtual double update()
        {
            ++currentIteration;
            learningRate = c / pow(currentIteration, decayExp);
        }

    private:
        double learningRate;
        double c;
        double decayExp;
        size_t currentIteration;
};

#endif // LEARNINGRATE_H
