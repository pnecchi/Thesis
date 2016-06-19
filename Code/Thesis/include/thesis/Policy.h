#ifndef POLICY_H
#define POLICY_H

#include <armadillo>  /* arma::vec */
#include <memory>     /* std::unique_ptr */
#include <assert.h>   /* assert */

/**
 * Policy is a pure abstract class that provides a generic interface for a
 * parametric policy pi(s, a; theta). It can be used as the core of an actor who
 * selects its actions according to the policy.
 */

class Policy
{
    public:
        /*!
         * Constructor.
         * Initialize a Policy object given the sizes of the observation space
         * and of the action space.
         * \param dimObservation_ dimension of the observation space
         * \param dimAction_ dimension of the action space
         */
        Policy(size_t dimObservation_, size_t dimAction_)
            : dimObservation(dimObservation_), dimAction(dimAction_) {}

        //! Default destructor
        virtual ~Policy() = default;

        /*!
         * Clone method.
         * The class is polymorphically clonable, making it possible to write
         * copy constructors and assignment operators for classes that aggregate
         * objects of the ParametricFunction hierarchy by composition.
         * \return a unique pointer to a copy of the policy
         */
        std::unique_ptr<Policy> clone() const
        {
            return checkedClone<Policy>();
        }

        /*!
         * Get observation size, i.e. size of the observation vector.
         * \return observation size
         */
        size_t getDimObservation() const { return dimObservation; }

        /*!
         * Get action size, i.e. size of the action vector.
         * \return action size
         */
        size_t getDimAction() const { return dimAction; }

        /*!
         * Get policy parameters size, i.e. size of the parameter vector
         * \return parameters size
         */
        virtual size_t getDimParameters() const = 0;

        /*!
         * Get method for the policy parameters.
         * \return parameters stored in an arma::vector
         */
        virtual arma::vec getParameters() const = 0;

        /*!
         * Set method for the policy parameters.
         * \param parameters_ the new parameters stored in an arma::vector
         */
        virtual void setParameters(arma::vec const &parameters_) = 0;

        /*!
         * Given an observation, select an action accordind to the policy.
         * \param observation_ observation
         * \return action
         */
        virtual arma::vec getAction(arma::vec const & observation_) const = 0;

        /*!
         * Reset policy to initial conditions.
         */
        virtual void reset() = 0;

    protected:
        /*!
         * checkedClone method for converting the unique pointer to Policy
         * returned by checkImpl() to a unique pointer of a derived class.
         */
        template<class T>
        std::unique_ptr<T> checkedClone() const
        {
            auto p = cloneImpl();
            assert(typeid(*p) == typeid(*this) &&
                   "subclass doesn't properly override cloneImpl()");
            assert(nullptr != dynamic_cast<T*>(p.get()));
            return std::unique_ptr<T>(static_cast<T*>(p.release()));
        }

    private:
        /**
         * NVI (non-virtual interface idiom) for virtual clone
         * cf. http://stackoverflow.com/questions/37788255/clonable-class-hierarchy-and-unique-ptr/
         */
        virtual std::unique_ptr<Policy> cloneImpl() const=0;

        //! Observation size
        size_t dimObservation;

        //! Action size
        size_t dimAction;
};

#endif // POLICY_H
