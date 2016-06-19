#ifndef CRITIC_H
#define CRITIC_H

#include <armadillo>
#include <memory>
#include <thesis/FunctionApproximator.h>

/*!
 * Critic implements the generic interface of a critic for a state-value
 * function. It is based on the FunctionApproximator hierarchy by composition,
 * so that different function approximators can be easily used as the core of
 * critic.
 */

class Critic
{
    public:
        /*!
         * Constructor.
         * Initialize a critic using a function approximator.
         * \param approximator_ function approximator
         */
        Critic(FunctionApproximator const &approximator_)
            : approximatorPtr(approximator_.clone()) {}

        /*!
         * Copy constructor.
         * \param rhs critic to copy
         */
        Critic(Critic const &rhs)
            : approximatorPtr(rhs.approximatorPtr->clone()) {}

        //! Default destructor
        virtual ~Critic() = default;

        /*!
         * Get method for the input dimention.
         * \return input size
         */
        size_t getDimInput() const { return approximatorPtr->getDimInput(); }

        /*!
         * Get method for the parameters dimension.
         * \return parameters size
         */
        size_t getDimParameters() const { return approximatorPtr->getDimParameters(); }

        /*!
         * Get method for the critic parameters.
         * \return parameters stored in an arma::vector
         */
        arma::vec getParameters() const
            { return approximatorPtr->getParameters(); }

        /*!
         * Set method for the critic parameters.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        void setParameters(arma::vec const &parameters_)
            { approximatorPtr->setParameters(parameters_); }

        /*!
         * Evaluate the critic for a given observation.
         * \param observation_ observation
         * \return evaluation of the critic for this observation
         */
        double evaluate(arma::vec &observation_) const
            { return approximatorPtr->evaluate(observation_); }

        /*!
         * Evaluate the critic's gradient wrt the parameters.
         * \param observation_ observation
         * \return evaluation of the critic's gradient for this observation
         */
        arma::vec gradient(arma::vec const &observation) const
            { return approximatorPtr->gradient(observation); }

        //! Reset critic to initial conditions
        void reset() { approximatorPtr->reset(); }

    private:
        //! Function approximator
        std::unique_ptr<FunctionApproximator> approximatorPtr;
};

#endif // CRITIC_H
