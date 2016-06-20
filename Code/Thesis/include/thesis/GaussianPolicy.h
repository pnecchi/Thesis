#ifndef GAUSSIANPOLICY_H
#define GAUSSIANPOLICY_H

#include <thesis/StochasticPolicy.h>
#include <armadillo>                  /* arma::vec */
#include <vector>                     /* std::vector */
#include <random>                     /* std::normal_distribution */



/*!
 * GaussianPolicy is a class that implements the Gaussian stochastic policy for
 * a continuous action space. For further information, cf. Wiering, M. and Van
 * Otterlo, M. - "Reinforcement learning." Adaptation, Learning, and
 * Optimization 12 (2012).
 */

class GaussianPolicy : public StochasticPolicy
{
    public:
        /*!
         * Constructor.
         * Initialize a Gaussian policy given the sizes of the observation
         * space and the dimension of the action space.
         * \param dimObservation_ dimension of the observation space
         * \param dimAction_ dimension of the action space
         */
        GaussianPolicy(size_t dimObservation_,
                       size_t dimAction_);

        //! Default destructor
        virtual ~GaussianPolicy() = default;

        //! Clone method for polymorphic clone
        std::unique_ptr<StochasticPolicy> clone() const
        {
            return checkedClone<StochasticPolicy>();
        }

        /*!
         * Get policy parameters size, i.e. size of the parameter vector
         * \return parameters size
         */
        virtual size_t getDimParameters() const { return dimParameters; }

        /*!
         * Get method for the policy parameters.
         * \return parameters stored in an arma::vector
         */
        virtual arma::vec getParameters() const;

        /*!
         * Set method for the policy parameters.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        virtual void setParameters(arma::vec const &parameters_);

        /*!
         * Given an observation, select an action accordind to the policy.
         * \param observation_ observation
         * \return action
         */
        virtual arma::vec getAction(arma::vec const &observation_) const;

        /*!
         * Evaluate the Likelihood score function at a given observation and
         * action.
         * \param observation_ observation
         * \param action_ action
         * \return likelihood score evaluated at observation_ and action_
         */
        virtual arma::vec likelihoodScore(arma::vec const &observation_,
                                          arma::vec const &action_) const;

        /*!
         * Reset policy to initial conditions.
         */
        virtual void reset();

    private:
        //! Virtual inner clone method
        virtual std::unique_ptr<Policy> cloneImpl() const;

        //! Initialize the Gaussian policy parameters
        void initializeParameters();

        //! Parameters size
        size_t dimParameters;

        //! Policy parameters: Theta = {psi, sigma}
        arma::vec parameters;
        arma::mat psiMat;      // mean parameters matrix (shares memory)

        /*!
         * Random number generator. Need to be mutable because the generator
         * state changes when simulating an action.
         */
        mutable std::mt19937 generator;
};

#endif // GAUSSIANPOLICY_H
