#ifndef NPGPEAGENT_H
#define NPGPEAGENT_H

#include <thesis/Agent.h>
#include <thesis/Policy.h>
#include <thesis/Statistics.h>
#include <memory>

/*!
 * NPGPEAgent implements a Natural PGPE agent based on a deterministic
 * controller and a gaussian probability distribution for the controller
 * parameters. For further information on NPGPE, refer to "Miyamae et Al. -
 * Natural Policy Gradient Methods with Parameter-based Exploration for Control
 * Tasks (2010)".
 */

class NPGPEAgent : public Agent
{
    public:

        /*!
         * Constructor.
         * Initialize an NPGPEAgent given a deterministic policy.
         * \param policy_ deterministic controller.
         * \param alpha_ learning rate.
         * \param discount_ discount factor
         */
        NPGPEAgent(Policy const &policy_,
                   double alpha_,
                   double discountFactor_);

        /*!
         * Copy constructor.
         * \param other_ NPGPEAgent to copy.
         */
        NPGPEAgent(NPGPEAgent const &other_);

        //! Default destructor.
        virtual ~NPGPEAgent() = default;

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
        StatisticsEMA baseline;

        //! Gradient cache
        arma::vec gradientMean;
        arma::mat gradientChol;

        //! Learning rate
        double alpha;

        //! Discount factor
        double discountFactor;

        // Cache variables
        arma::vec observation;
        arma::vec action;
        double reward;
};

#endif // NPGPEAGENT_H
