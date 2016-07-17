/*
 * Copyright (c) 2016 Pierpaolo Necchi
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
