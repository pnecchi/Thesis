#include "thesis/GaussianDistribution.h"

GaussianDistribution::GaussianDistribution(size_t dimOutput_)
    : dimOutput(dimOutput_),
      dimParameters(2 * dimOutput_),
      parameters(2 * dimOutput_),
      generator(16u)                    // eventually use random seed
{
    initializeParameters();
}

void GaussianDistribution::initializeParameters()
{
    parameters.rows(0, dimOutput - 1).zeros();
    parameters.rows(dimOutput, dimParameters - 1).ones();
}

std::unique_ptr<ProbabilityDistribution> GaussianDistribution::clone() const
{
    return std::unique_ptr<ProbabilityDistribution>(new GaussianDistribution(*this));
}

arma::vec GaussianDistribution::simulate() const
{
    arma::vec simulation(dimOutput);
    for (size_t i = 0; i < dimOutput; ++i)
    {
        std::normal_distribution<double> d(parameters[i], parameters[dimOutput+i]);
        simulation[i] = d(generator);
    }
    return simulation;
}

arma::vec GaussianDistribution::likelihoodScore(arma::vec const &output_) const
{
    arma::vec likScore(dimParameters);
    arma::vec delta = output_ - parameters.rows(0, dimOutput-1);
    arma::vec sigma = parameters.rows(dimOutput, dimParameters-1);
    arma::vec sigma2 = sigma % sigma;
    arma::vec sigma3 = sigma2 % sigma;
    likScore.rows(0, dimOutput-1) = delta / sigma2;
    likScore.rows(dimOutput, dimParameters-1) = delta % delta / sigma3 - 1.0 / sigma;
    return likScore;
}

void GaussianDistribution::reset()
{
    initializeParameters();
}
