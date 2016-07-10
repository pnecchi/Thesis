#ifndef LEARNINGRATE_H
#define LEARNINGRATE_H

#include <math.h>  /* pow */
#include <memory>  /* unique_ptr */

/**
 * LearningRate is an abstract class which implements a generic interface for a
 * learning rate which can be used in an iterative optimization procedure, e.g.
 * the gradient descent algorithm.
 */

class LearningRate
{
    public:
        //! Destructor.
        virtual ~LearningRate() = default;

        /**
         * Clone method.
         * the class is clonable to allow for polymorphic copy.
         * \return unique_ptr pointing to new LearningRate instance.
         */
        virtual std::unique_ptr<LearningRate> clone() const = 0;

        /**
         * Get learning rate current value.
         * \return learning rate
         */
        virtual double get() const = 0;

        /**
         * Update learning rate according to a certain schedule.
         */
        virtual void update() = 0;

        /**
         * Reset learning rate to initial conditions.
         */
        virtual void reset() = 0;
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
        ConstantLearningRate(double learningRate_=0.1)
            : learningRate(learningRate_) {}

        //! Destructor
        virtual ~ConstantLearningRate() = default;

        /**
         * Clone method.
         * the class is clonable to allow for polymorphic copy.
         * \return unique_ptr pointing to new ConstantLearningRate instance.
         */
        virtual std::unique_ptr<LearningRate> clone() const;

        /**
         * Get learning rate current value.
         * \return learning rate
         */
        virtual double get() const { return learningRate; }

        /**
         * Update learning rate according to a certain schedule.
         */
        virtual void update() { /* Nothing to do */ }

        /**
         * Reset learning rate to initial conditions.
         */
        virtual void reset() { /* Nothing to do */ }

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
            : c(c_), decayExp(decayExp_), currentIteration(1ul) {}

        //! Destructor
        virtual ~DecayingLearningRate() = default;

        /**
         * Clone method.
         * the class is clonable to allow for polymorphic copy.
         * \return unique_ptr pointing to new DecayingLearningRate instance.
         */
        virtual std::unique_ptr<LearningRate> clone() const;

        /**
         * Get learning rate current value.
         * \return learning rate
         */
        virtual double get() const { return learningRate; }

        /**
         * Update learning rate according to a power decay schedule.
         */
        virtual void update();

        /**
         * Reset learning rate to initial conditions.
         */
        virtual void reset();

    private:
        double learningRate;
        double c;
        double decayExp;
        size_t currentIteration;
};

#endif // LEARNINGRATE_H
