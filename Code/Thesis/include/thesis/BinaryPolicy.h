#ifndef BINARYPOLICY_H
#define BINARYPOLICY_H

#include <thesis/Policy.h>


class BinaryPolicy : public Policy
{
    public:
        BinaryPolicy(size_t dimObservation_);
        virtual ~BinaryPolicy() = default;
        virtual size_t getDimParameters() const { return dimParameters; }
        virtual arma::vec getParameters() const { return parameters; }
        virtual void setParameters(arma::vec const & parameters_) { parameters = parameters_; }
        virtual arma::vec getAction(arma::vec const & observation_) const;
        virtual void reset();

    private:
        void initializeParameters();
        virtual std::unique_ptr<Policy> cloneImpl() const;
        arma::vec parameters;
        size_t dimParameters;
};

#endif // BINARYPOLICY_H
