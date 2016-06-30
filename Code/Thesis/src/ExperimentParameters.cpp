#include "thesis/ExperimentParameters.h"
#include <iostream>
#include <fstream>
#include <getpot.h>

ExperimentParameters::ExperimentParameters()
    : inputDataPath("../../../Data/Input/synthetic.csv"),
      outputDataPath("../../../Data/Output/"),
      debugDataPath("../../../Data/Debug/"),
      riskFreeRate(0.0),
      deltaP(0.0),
      deltaF(0.0),
      deltaS(0.0),
      numDaysObserved(2),
      lambda(0.5),
      alphaActor(0.005),
      alphaCritic(0.01),
      alphaBaseline(0.05),
      numExperiments(1),
      numEpochs(100),
      numTrainingSteps(1000),
      numTestSteps(100)
{
    /* Nothing to do */
}

ExperimentParameters::ExperimentParameters(std::string const &filename, bool verbose)
    : ExperimentParameters()  // Initialize parameters with default values
{
    // Check if target file exists and is readable
    std::ifstream check(filename);
    if(!check)
    {
        std::cerr << "ERROR: Parameter file " << filename << " does not exist" << std::endl;
        std::cerr << "Reverting to default values." << std::endl;
        if (verbose)
            std::cout << (*this);
        check.close();
    }
    else
    {
        check.close();

        // Read parameters from file using GetPot
        GetPot ifile(filename.c_str());
        inputDataPath = ifile("inputDataPath", inputDataPath.c_str());
        outputDataPath = ifile("outputDataPath", outputDataPath.c_str());
        debugDataPath = ifile("debugDataPath", debugDataPath.c_str());
        riskFreeRate = ifile("riskFreeRate", riskFreeRate);
        deltaP = ifile("deltaP", deltaP);
        deltaF = ifile("deltaF", deltaF);
        deltaS = ifile("deltaS", deltaS);
        numDaysObserved = ifile("numDaysObserved", static_cast<int>(numDaysObserved));
        lambda = ifile("lambda", lambda);
        alphaActor = ifile("alphaActor", alphaActor);
        alphaCritic = ifile("alphaCritic", alphaCritic);
        alphaBaseline = ifile("alphaBaseline", alphaBaseline);
        numExperiments = ifile("numExperiments", static_cast<int>(numExperiments));
        numEpochs = ifile("numEpochs", static_cast<int>(numEpochs));
        numTrainingSteps = ifile("numTrainingSteps", static_cast<int>(numTrainingSteps));
        numTestSteps = ifile("numTestSteps", static_cast<int>(numTestSteps));

        if (verbose)
        {
            std::cout << (*this);
        }
    }
}

std::ostream &operator<<(std::ostream &os, ExperimentParameters const &params)
{
    std::cout << ".. inputDataPath:    " << params.inputDataPath << std::endl;
    std::cout << ".. outputDataPath:   " << params.outputDataPath << std::endl;
    std::cout << ".. debugDataPath:    " << params.debugDataPath << std::endl;
    std::cout << ".. riskFreeRate:     " << params.riskFreeRate << std::endl;
    std::cout << ".. deltaP:           " << params.deltaP << std::endl;
    std::cout << ".. deltaF:           " << params.deltaF << std::endl;
    std::cout << ".. deltaS:           " << params.deltaS << std::endl;
    std::cout << ".. numDaysObserved:  " << params.numDaysObserved << std::endl;
    std::cout << ".. lambda:           " << params.lambda << std::endl;
    std::cout << ".. alphaActor:       " << params.alphaActor << std::endl;
    std::cout << ".. alphaCritic:      " << params.alphaCritic << std::endl;
    std::cout << ".. alphaBaseline:    " << params.alphaBaseline << std::endl;
    std::cout << ".. numExperiments:   " << params.numExperiments << std::endl;
    std::cout << ".. numEpochs:        " << params.numEpochs << std::endl;
    std::cout << ".. numTrainingSteps: " << params.numTrainingSteps << std::endl;
    std::cout << ".. numTestSteps:     " << params.numTestSteps << std::endl;
}


