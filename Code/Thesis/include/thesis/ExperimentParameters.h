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

#ifndef EXPERIMENTPARAMETERS_H
#define EXPERIMENTPARAMETERS_H

#include <string>

class ExperimentParameters
{
    friend std::ostream &operator<<(std::ostream &os,
                                    ExperimentParameters const & params);

    public:

        /*!
         * Default constructor.
         */
        ExperimentParameters();

        /*!
         * Constructor.
         * Read parameters from external file using getpot.
         */
        ExperimentParameters(std::string const &filename, bool verbose=false);

        //! Default destructor.
        virtual ~ExperimentParameters() = default;

        /*!
         * Market parameters
         */

        //! Risk-free rate
        double riskFreeRate;

        /*!
         * Asset allocation task parameters
         */

        //! Proportional transaction costs
        double deltaP;

        //! Fixed transaction costs
        double deltaF;

        //! Short-selling fees
        double deltaS;

        //! Number of past days observed by the agent
        size_t numDaysObserved;

        /*!
         * Agent parameters
         */

        //! TD(lambda) parameter
        double lambda;

        //! Actor learning rate
        double alphaConstActor;
        double alphaExpActor;

        //! Critic learning rate
        double alphaConstCritic;
        double alphaExpCritic;

        //! Baseline learning rate
        double alphaConstBaseline;
        double alphaExpBaseline;

        /*!
         * Experiment parameters
         */

        //! Number of experiments
        size_t numExperiments;

        //! Number of epochs
        size_t numEpochs;

        //! Number of training steps
        size_t numTrainingSteps;

        //! Number of test steps
        size_t numTestSteps;
};

/*!
 * Print parameters to ostream.
 */
std::ostream& operator<<(std::ostream &os,
                         ExperimentParameters const & params);

#endif // EXPERIMENTPARAMETERS_H
