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

#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <thesis/Task.h>        /* Task */
#include <thesis/Agent.h>       /* Agent */
#include <memory>               /* std::unique_ptr */

/**
 * Generic interface for a RL experiment, which matches a task and an agend and
 * handles their interactions during the learning process.
 */

class Experiment
{
    public:

        /**
         * Constructor.
         * Initializes an RL experiment given a task and an agent.
         * @param task_ a task defining an MDP.
         * @param agent_ an agent which performs the task.
         */
        Experiment(Task const &task_,
                   Agent const &agent_);

        //! Copy constructor.
        Experiment(Experiment const &experiment_);

        //! Virtual destructor.
        virtual ~Experiment() = default;

        //! Clone method.
        virtual std::unique_ptr<Experiment> clone() const = 0;

        /**
         * Run the RL experiment, i.e. the learning process of the agent on the
         * given task.
         */
        virtual void run() = 0;

    protected:

        //! Task
        std::unique_ptr<Task> taskPtr;

        //! Agent
        std::unique_ptr<Agent> agentPtr;
};

#endif // EXPERIMENT_H
