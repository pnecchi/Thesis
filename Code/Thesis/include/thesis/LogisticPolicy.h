#ifndef LOGISTICPOLICY_H
#define LOGISTICPOLICY_H

#include <thesis/Policy.h>
#include <armadillo>        /* arma::vec */
#include <memory>           /* std::unique_ptr */
#include <limits>           /* std::numeric_limits<double> */

/*!
 * LogisticPolicy implements a deterministic parametric policy that produces
 * actions in (-1, 1) following
 *     a = tanh( theta' * observation )
 * This policy is used to implement a long-short trading system that invests
 * in a single risky asset and can be used as a controller for a PGPE policy.
 */

class LogisticPolicy : public Policy
{
    public:
        /*!
         * Constructor.
         * Initialize a BinaryPolicy object given the size of the observation.
         * \param dimObservation_ dimension of the observation space
         * \param paramMinValue_ parameters lower bound
         * \param paramMaxValue_ parameters upper bound
         */
        LogisticPolicy(size_t dimObservation_,
                       double paramMinValue_=std::numeric_limits<double>::min(),
                       double paramMaxValue_=std::numeric_limits<double>::max());

        //! Default copy constructor
        LogisticPolicy(LogisticPolicy const &other_) = default;

        //! Default destructor.
        virtual ~LogisticPolicy() = default;

        /*!
         * Get policy parameters size, i.e. size of the parameter vector
         * \return parameters size
         */
        virtual size_t getDimParameters() const { return dimParameters; }

        /*!
         * Get method for the policy parameters.
         * \return parameters stored in an arma::vector
         */
        virtual arma::vec getParameters() const { return parameters; }

        /*!
         * Set method for the policy parameters. The parameters bounds are enforced.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        virtual void setParameters(arma::vec const & parameters_);

        /*!
         * Given an observation, select an action accordind to the policy.
         * \param observation_ observation
         * \return action
         */
        virtual arma::vec getAction(arma::vec const & observation_) const;

        /*!
         * Reset policy to initial conditions.
         */
        virtual void reset();

    private:
        //! Virtual inner clone method
        virtual std::unique_ptr<Policy> cloneImpl() const;

        //! Initialize LogisticPolcy parameters
        void initializeParameters();

        //! Policy parameters
        arma::vec parameters;
        size_t dimParameters;

        /*!
         * Parameters bounds.
         * The parameters must lie in the interval [paramMinValue, paramMaxValue]
         * This constraint can be useful to avoid divergence when modifying the
         * parameters by gradient ascent in an optimization procedure.
         */
        double paramMinValue;
        double paramMaxValue;
};

#endif // LOGISTICPOLICY_H
