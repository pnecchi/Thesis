#include <thesis/LearningRate.h>

std::unique_ptr<LearningRate> ConstantLearningRate::clone() const
{
    return std::unique_ptr<LearningRate>(new ConstantLearningRate(*this));
}

std::unique_ptr<LearningRate> DecayingLearningRate::clone() const
{
    return std::unique_ptr<LearningRate>(new DecayingLearningRate(*this));
}

void DecayingLearningRate::update()
{
    ++currentIteration;
    learningRate = c / pow(currentIteration, decayExp);
}

void DecayingLearningRate::reset()
{
    currentIteration = 1ul;
    learningRate = c;
}
