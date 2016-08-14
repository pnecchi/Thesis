#ifndef LONGSHORTPOLICY_H
#define LONGSHORTPOLICY_H

#include <thesis/Policy.h>

/*!
 * LongShortPolicy implements a deterministic parametric policy that produces
 * 2-dimensional actions in {-1, 1}, according to
 *     a_1 = sign( theta' * observation )
 *     a_2 = - a_1
 * This policy is used to implement a long-short trading system that invests in
 * opposite positions in two risky assets and can be used as a controller for a
 * PGPE policy.
 */

class LongShortPolicy: public Policy
{
    public:
        /*!
         * Constructor.
         * Initialize a LongShortPolicy object given the size of the observation.
         * \param dimObservation_ dimension of the observation space
         * \param paramMinValue_ parameters lower bound
         * \param paramMaxValue_ parameters upper bound
         */
        LongShortPolicy(size_t dimObservation_,
                        double paramMinValue_=std::numeric_limits<double>::min(),
                        double paramMaxValue_=std::numeric_limits<double>::max());

        //! Default copy constructor
        LongShortPolicy(LongShortPolicy const &other_) = default;

        //! Default destructor.
        virtual ~LongShortPolicy() = default;

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
        //! Initialize BinaryPolicy parameters
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

        //! Virtual inner clone method
        virtual std::unique_ptr<Policy> cloneImpl() const;
};

#endif // LONGSHORTPOLICY_H
