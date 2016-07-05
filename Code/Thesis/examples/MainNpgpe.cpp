#include <iostream>
#include <string>
#include <armadillo>
#include <thesis/ExperimentParameters.h>
#include <thesis/MarketEnvironment.h>
#include <thesis/AssetAllocationTask.h>
#include <thesis/LinearRegressor.h>
#include <thesis/Critic.h>
#include <thesis/BoltzmannPolicy.h>
#include <thesis/GaussianPolicy.h>
#include <thesis/LogisticPolicy.h>
#include <thesis/BinaryPolicy.h>
#include <thesis/LogisticPolicy.h>
#include <thesis/GaussianDistribution.h>
#include <thesis/PgpePolicy.h>
#include <thesis/NpgpePolicy.h>
#include <thesis/StochasticActor.h>
#include <thesis/ArrsacAgent.h>
#include <thesis/AracAgent.h>
#include <thesis/NpgpeAgent.h>
#include <thesis/AssetAllocationExperiment.h>

int main()
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "-        Algorithmic Asset Allocation        -" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    // 0) Parameters
    std::cout << "0) Read parameters" << std::endl;
    std::string parametersFilepath  = "../../../Data/Parameters/ParametersNpgpe.pot";
    const ExperimentParameters params(parametersFilepath, true);

    // Copy parameters
    std::string inputDataPath = params.inputDataPath;
    std::string outputDataPath = params.outputDataPath;
    std::string debugDataPath = params.debugDataPath;
    double riskFreeRate = params.riskFreeRate;
    double deltaP = params.deltaP;
    double deltaF = params.deltaF;
    double deltaS = params.deltaS;
    size_t numDaysObserved = params.numDaysObserved;
    double alphaActor = params.alphaActor;
    size_t numExperiments = params.numExperiments;
    size_t numEpochs = params.numEpochs;
    size_t numTrainingSteps = params.numTrainingSteps;
    size_t numTestSteps = params.numTestSteps;

    // 1) Initialization
    std::cout << std::endl << "1) Initialization" << std::endl;

	// Market
	std::cout << ".. Market environment - ";
	MarketEnvironment market(inputDataPath);
    size_t startDate = 0;
	size_t endDate = numDaysObserved + numTrainingSteps + numTestSteps - 1;
	market.setEvaluationInterval(startDate, endDate);
    std::cout << "done" << std::endl;

    // Asset allocation task
    std::cout << ".. Asset allocation task - ";
	AssetAllocationTask task(market,
                             riskFreeRate,
                             deltaP,
                             deltaF,
                             deltaP,
                             numDaysObserved);
    std::cout << "done" << std::endl;

    // PGPE Binary policy
    std::cout << ".. Policy - ";
    BinaryPolicy controller(task.getDimObservation());
    std::cout << "done" << std::endl;

    // NPGPE Agent
    std::cout << ".. NPGPE Agent - ";
    NPGPEAgent agent(controller,
                     alphaActor,
                     0.95);

    std::cout << "done" << std::endl;

    // Asset allocation experiment
    std::cout << ".. Asset allocation experiment - ";
    AssetAllocationExperiment experiment(task,
                                         agent,
                                         numExperiments,
                                         numEpochs,
                                         numTrainingSteps,
                                         numTestSteps);
    std::cout << "done" << std::endl;

    // 2) Run experiment
    std::cout << std::endl << "2) Experiment" << std::endl;
    experiment.run();

	return 0;
}
