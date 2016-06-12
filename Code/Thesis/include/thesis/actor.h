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
    // Default constructor
    Actor(Module const &module_);

    // Default destructor
    virtual ~Actor() = default;

    // Get sizes
    size_t getDimInput() const { return dimInput; }
    size_t getDimOutput() const { return dimOutput; }

    // Get Action
    void selectAction(arma::vec const &observation, arma::vec &action);



private:
    Module module;

};


#endif // ACTOR_H
