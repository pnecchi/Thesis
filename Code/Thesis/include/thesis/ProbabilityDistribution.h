#ifndef PROBABILITYDISTRIBUTION_H
#define PROBABILITYDISTRIBUTION_H

#include <armadillo>
#include <memory>

class ProbabilityDistribution
{
    public:
        ProbabilityDistribution() = default;
        ProbabilityDistribution(ProbabilityDistribution const &other_) = default;
        virtual ~ProbabilityDistribution() = default;
        virtual std::unique_ptr<ProbabilityDistribution> clone() const = 0;
        virtual size_t getDimOutput() const = 0;
        virtual size_t getDimParameters() const = 0;
        virtual arma::vec getParameters() const = 0;
        virtual void setParameters(arma::vec const &parameters_) = 0;
        virtual arma::vec simulate() const = 0;
        virtual arma::vec likelihoodScore(arma::vec const &output_) const = 0;
        virtual void reset() = 0;
};

#endif // PROBABILITYDISTRIBUTION_H
