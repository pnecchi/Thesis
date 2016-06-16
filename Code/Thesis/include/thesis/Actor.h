#ifndef ACTOR_H
#define ACTOR_H

#include <armadillo>
#include <memory>

/**
 * An Actor is the policy employed by an Agent for selecting an action given an
 * observation of the system. The Actor base class is an abstract base class
 * which defines the generic interface of an actor.
 */

class Actor
{
    public:
        //! Default constructor
        Actor() = default;

        //! Default copy constructor
        Actor(Actor const &rhs) = default;

        //! Default destructor
        virtual ~Actor() = default;

        //! Get action size
        virtual size_t getDimAction() const=0;

        /*!
         * Select action given an observation of the system.
         * \param observation_ observation of the system
         * \return an action
         */
        virtual arma::vec getAction(arma::vec const &observation_) const = 0;
};


#endif // ACTOR_H
