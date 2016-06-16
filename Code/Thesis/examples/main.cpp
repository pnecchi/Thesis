#include <iostream>
#include <string>
#include <armadillo>
#include <thesis/MarketEnvironment.h>
#include <thesis/AssetAllocationTask.h>
#include <thesis/LinearRegressor.h>
#include <thesis/Critic.h>
#include <thesis/BoltzmannExplorationPolicy.h>
#include <thesis/StochasticActor.h>

int main(int argc, char *argv[])
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "-        Algorithmic Asset Allocation        -" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

	// Market parameters
	std::string inputFilePath = "../../../Data/Input/synthetic.csv";
	double riskFreeRate = 0.0;
	size_t startDate = 0;
	size_t endDate = 10;

	// Task parameters
	double deltaP = 0.0005;
	double deltaF = 0.0;
	double deltaS = 0.0;
	size_t numDaysObserved = 2;

	// Market
	std::cout << ">> Initialize market environment" << std::endl;
	MarketEnvironment market(inputFilePath, riskFreeRate, startDate, endDate);

    // Asset allocation task
    std::cout << ">> Initialize asset allocation task" << std::endl;
	AssetAllocationTask task(market, deltaP, deltaF, deltaP, numDaysObserved);

	// State-value function critic
	std::cout << ">> Initialize linear regressor" << std::endl;
	LinearRegressor linearReg(task.getDimObservation());

    // Initialize critic
    std::cout << ">> Initialize critic" << std::endl;
    Critic criticV(linearReg);

    // Boltzmann Exploration Policy
    std::cout << ">> Initialize Boltzmann stochastic policy" << std::endl;
    std::vector<double> possibleAction {-1.0, 0.0, 1.0};
    BoltzmannExplorationPolicy policy(task.getDimObservation(), possibleAction);

    // Stochastic Actor
    std::cout << ">> Initialize actor" << std::endl;
    StochasticActor actor(policy);

    arma::vec observation(task.getDimObservation());
    arma::vec action(task.getDimAction());
    arma::vec likScore(actor.getDimParameters());
    double reward;

    std::cout << ">> 1st interaction" << std::endl;
    std::cout << "Observation: " << std::endl;
    observation = task.getObservation();
    observation.print(std::cout);
    std::cout << "Action:" << std::endl;
    action = actor.getAction(observation);
    action.print(std::cout);
    task.performAction(action);
    reward = task.getReward();
    std::cout << "Reward: " << reward << std::endl;
    std::cout << "Critic: " << criticV.evaluate(observation) << std::endl;
    likScore = actor.likelihoodScore(observation, action);
    std::cout << "Likelihood score:" << std::endl;
    likScore.print(std::cout);

    std::cout << ">> 2nd interaction" << std::endl;
    std::cout << "Observation: " << std::endl;
    observation = task.getObservation();
    observation.print(std::cout);
    std::cout << "Action:" << std::endl;
    action = actor.getAction(observation);
    action.print(std::cout);
    task.performAction(action);
    reward = task.getReward();
    std::cout << "Reward: " << reward << std::endl;
    std::cout << "Critic: " << criticV.evaluate(observation) << std::endl;
    likScore = actor.likelihoodScore(observation, action);
    std::cout << "Likelihood score:" << std::endl;
    likScore.print(std::cout);

	return 0;
}
