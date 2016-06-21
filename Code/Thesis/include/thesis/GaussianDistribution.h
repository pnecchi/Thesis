#ifndef GAUSSIANDISTRIBUTION_H
#define GAUSSIANDISTRIBUTION_H

#include <thesis/ProbabilityDistribution.h>
#include <memory>

class GaussianDistribution : public ProbabilityDistribution
{
    public:
        GaussianDistribution(size_t dimOutput_);
        GaussianDistribution(GaussianDistribution const &other) = default;
        virtual ~GaussianDistribution() = default;
        virtual std::unique_ptr<ProbabilityDistribution> clone() const;
        virtual size_t getDimOutput() const { return dimOutput; }
        virtual size_t getDimParameters() const { return dimParameters; }
        virtual arma::vec getParameters() const { return parameters; }
        virtual void setParameters(arma::vec const &parameters_) { parameters = parameters_; }
        virtual arma::vec simulate() const;
        virtual arma::vec likelihoodScore(arma::vec const &output_) const;
        virtual void reset();

    private:
        void initializeParameters();

        // Parameters: [mu_1; ... ; mu_D; sigma_1; ... ; sigma_D]
        arma::vec parameters;

        // Sizes
        size_t dimOutput;
        size_t dimParameters;

        // Random number generator
        mutable std::mt19937 generator;
};

#endif // GAUSSIANDISTRIBUTION_H
