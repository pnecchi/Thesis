#ifndef LOGISTICPOLICY_H
#define LOGISTICPOLICY_H

#include <thesis/Policy.h>


class LogisticPolicy : public Policy
{
    public:
        LogisticPolicy(size_t dimObservation_);
        virtual ~LogisticPolicy() = default;
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

#endif // LOGISTICPOLICY_H
