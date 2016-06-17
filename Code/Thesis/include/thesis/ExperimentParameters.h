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
         * File paths
         */

        //! Input filepath
        std::string inputDataPath;

        //! Output filepath
        std::string outputDataPath;

        //! Debug filepath
        std::string debugDataPath;

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
         * ARRSAC agent parameters
         */

        //! Actor learning rate
        double alphaActor;

        //! Critic learning rate
        double alphaCritic;

        //! Baseline learning rate
        double alphaBaseline;

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
