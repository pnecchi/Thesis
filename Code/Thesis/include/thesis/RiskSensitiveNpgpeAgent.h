#ifndef RISKSENSITIVENPGPEAGENT_H
#define RISKSENSITIVENPGPEAGENT_H

#include <thesis/Agent.h>
#include <thesis/Policy.h>
#include <thesis/Statistics.h>
#include <thesis/LearningRate.h>
#include <memory>

/*!
 * RiskSensitiveNPGPEAgent implements a risk-sensitive natural PGPE agent based
 * on a deterministic controller and a gaussian probability distribution for the
 * controller parameters. For further information on Risk-Sensitive NPGPE, refer
 * thesis report.
 */

class RiskSensitiveNPGPEAgent : public Agent
{
    public:

        /*!
         * Constructor.
         * Initialize aRiskSensitiveNPGPEAgent given a deterministic policy.
         * \param policy_ deterministic controller.
         * \param learningRate learning rate object.
         * \param discount_ discount factor
         */
        RiskSensitiveNPGPEAgent(Policy const &policy_,
                                LearningRate const &learningRate_,
                                double discountFactor_);

        /*!
         * Copy constructor.
         * \param other_ RiskSensitiveNPGPEAgent to copy.
         */
        RiskSensitiveNPGPEAgent(RiskSensitiveNPGPEAgent const &other_);

        //! Default destructor.
        virtual ~RiskSensitiveNPGPEAgent() = default;

        //! Clone method for virtual copy constructor
        virtual std::unique_ptr<Agent> clone() const;

        // Get action size
        virtual size_t getDimAction() const { return policyPtr->getDimAction(); }

        // Receive observation of the system state --> O_t
        virtual void receiveObservation(arma::vec const &observation_)
            { observation = observation_; }

        // Get action to be performed on the system --> A_t
        virtual arma::vec getAction();

        // Receive reward from the system --> R_{t+1}
        virtual void receiveReward(double reward_)
            { reward = reward_; }

        // Receive next observation --> O_{t+1}
        virtual void receiveNextObservation(arma::vec const &nextObservation_) {}

        // Learning step given previous experience
        virtual void learn();

        // New epoch
        virtual void newEpoch();

        // Reset
        virtual void reset();

    private:
        void initializeParameters();

        //! Deterministic controller
        std::unique_ptr<Policy> policyPtr;

        //! Hyperparameters
        arma::vec mean;
        arma::mat choleskyFactor;  // Sigma = choleskyFactor * choleskyFactor'

        //! Parameters distribution
        mutable std::mt19937 generator;
        mutable std::normal_distribution<double> gaussianDistr;
        arma::vec xi;

        //! Baseline
        StatisticsEMA rewardBaseline;
        StatisticsEMA squareRewardBaseline;

        //! Gradient cache
        arma::vec gradientMean;
        arma::mat gradientChol;

        //! Learning rate
        std::unique_ptr<LearningRate> learningRatePtr;

        //! Discount factor
        double discountFactor;

        // Cache variables
        arma::vec observation;
        arma::vec action;
        double reward;
};

#endif // RISKSENSITIVENPGPEAGENT_H
