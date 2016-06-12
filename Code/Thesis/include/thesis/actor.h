#ifndef ACTOR_H
#define ACTOR_H

#include <armadillo>

/**
 * An Actor is the policy employed by an Agent for selecting an action given an
 * observation of the system. The Actor base class is an abstract base class
 * which defines the generic interface of an actor.
 */

class Actor
{
    public:
        // Default destructor
        virtual ~Actor() = default;

        // Get Action
        arma::vec getAction(arma::vec const &observation) const = 0;
};


#endif // ACTOR_H
