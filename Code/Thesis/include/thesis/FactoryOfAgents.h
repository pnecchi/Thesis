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

#ifndef FACTORYOFAGENTS_H
#define FACTORYOFAGENTS_H

#include <memory>
#include <thesis/LearningRate.h>
#include <thesis/Agent.h>
#include <thesis/AracAgent.h>
#include <thesis/ArAgent.h>
#include <thesis/ArrsacAgent.h>
#include <thesis/NpgpeAgent.h>
#include <thesis/RiskSensitiveNpgpeAgent.h>

/**
 * Simple factory class for creating different types of agent. The class is a
 * singleton and exploits Meyers' trick.
 */

class FactoryOfAgents
{
    public:
        /*!
         * instance method for creating a Singleton using Meyers' trick.
         * @return a reference to the unique instance of a FactoryOfAgents object.
         */
        static FactoryOfAgents& instance(size_t const &dimObservation_,
                                         LearningRate const &baselineLearningRate_,
                                         LearningRate const &criticLearningRate_,
                                         LearningRate const &actorLearningRate_,
                                         double const &lambda_);

        /*!
         * make method for creating an agent of the given type.
         * @param type of the Agent to create
         * @return unique pointer to the instance of the agent
         */
        std::unique_ptr<Agent> make(std::string const &agentId) const;

    private:
        //! Standard constructor
        FactoryOfAgents() = default;

        //! Constructor
        FactoryOfAgents(size_t const &dimObservation_,
                        LearningRate const &baselineLearningRate_,
                        LearningRate const &criticLearningRate_,
                        LearningRate const &actorLearningRate_,
                        double const &lambda_);

        FactoryOfAgents(FactoryOfAgents const &)=delete;
        FactoryOfAgents& operator=(FactoryOfAgents const &)=delete;

        //! Default destructor
        virtual ~FactoryOfAgents() = default;

        //! Builder for ARAC agent
        std::unique_ptr<ARACAgent> makeARACAgent() const;

        //! Builder for PGPE Agent
        std::unique_ptr<ARACAgent> makePGPEAgent() const;

        //! Builder for NPGPE agent
        std::unique_ptr<NPGPEAgent> makeNPGPEAgent() const;

        //! Builder for RSARAC agent
        std::unique_ptr<ARRSACAgent> makeRSARACAgent() const;

        //! Builder for RSPGPE agent
        std::unique_ptr<ARRSACAgent> makeRSPGPEAgent() const;

        //! Builder for RSNPGPE agent
        std::unique_ptr<RiskSensitiveNPGPEAgent> makeRSNPGPEAgent() const;

        size_t dimObservation;
        std::unique_ptr<LearningRate> baselineLearningRatePtr;
        std::unique_ptr<LearningRate> criticLearningRatePtr;
        std::unique_ptr<LearningRate> actorLearningRatePtr;
        double lambda;
};

#endif // FACTORYOFAGENTS_H
