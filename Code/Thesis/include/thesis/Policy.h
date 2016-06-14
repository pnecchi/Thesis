#ifndef POLICY_H
#define POLICY_H

#include <assert.h>     /* assert */

/**
 * Policy is a pure abstract class that provides a generic interface for a
 * parametric policy pi(s, a). It can be used as the core of an actor who
 * selects its actions according to the policy.
 */

class Policy
{
    public:
        // Default constructor
        Policy(size_t dimObservation_, size_t dimAction_)
            : dimObservation(dimObservation_), dimAction(dimAction_) {}

        // Default destructor */
        virtual ~Policy() = default;

        /**
         * The class is polymorphically clonable, making it possible to write
         * copy constructors and assignment operators for classes that aggregate
         * objects of the ParametricFunction hierarchy by composition.
         */
        std::unique_ptr<Policy> clone() const
        {
            return checkedClone<Policy>();
        }

        // Get policy sizes
        size_t getDimObservation() const { return dimObservation; }
        size_t getDimAction() const { return dimAction; }
        virtual size_t getDimParameters() const = 0;

        // Get and set parameters
        virtual arma::vec getParameters() const = 0;
        virtual void setParameters(arma::vec const &parameters) = 0;

        // Get action given an observation
        virtual arma::vec getAction(arma::vec const & observation) const = 0;

    protected:
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

        // Policy sizes
        size_t dimObservation;
        size_t dimAction;
};

#endif // POLICY_H
