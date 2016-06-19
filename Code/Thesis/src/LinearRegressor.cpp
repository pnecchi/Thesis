#include "thesis/LinearRegressor.h"

LinearRegressor::LinearRegressor(size_t dimInput_)
    : FunctionApproximator(dimInput_), parameters(dimInput_ + 1)
{
    initializeParameters();
}

void LinearRegressor::initializeParameters()
{
    parameters.randu();
    parameters -= 0.5;
    parameters *= 0.001;
}

std::unique_ptr<FunctionApproximator> LinearRegressor::clone() const
{
    return std::unique_ptr<FunctionApproximator>(new LinearRegressor(*this));
}

arma::vec LinearRegressor::getParameters() const
{
    return parameters;
}

void LinearRegressor::setParameters(arma::vec const &parameters_)
{
    parameters = parameters_;
}

double LinearRegressor::evaluate(arma::vec const &x) const
{
    return parameters(0) + arma::dot(parameters.rows(1, getDimParameters()-1), x);
}

arma::vec LinearRegressor::gradient(arma::vec const &x) const
{
    arma::vec grad(getDimParameters());
    grad(0) = 1.0;
    grad.rows(1, getDimParameters() - 1) = x;
    return grad;
}

void LinearRegressor::reset()
{
    initializeParameters();
}
