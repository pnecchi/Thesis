#!/usr/bin/env python
# -*- coding: utf-8 -*-

#===============================================================================

#-------------------------------------------------------------------------------
# The experiment parameters can be set by modifying the variables below
#-------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------|
# Working Directories:                                                                               |
#  * paramBaseDir: the directory in which the parameters.pot file will be written                    |
#  * inputBaseDir: the directory in which the C++ program will look for the input file               |
#  * outputBaseDir: the base directory in which the C++ program will write the output files          |
#  * debugBaseDir: the base directory in which the C++ program will write the debug files            |
#  * postProcessingDir: the base directory in which the output of the postprocessing will be written |
#----------------------------------------------------------------------------------------------------|

thesisBaseDir     = '/home/pierpaolo/Documents/University/6_Anno_Poli/7_Thesis/'
paramBaseDir      = thesisBaseDir + 'Data/Parameters/'
inputBaseDir      = thesisBaseDir + 'Data/Input/'
outputBaseDir     = thesisBaseDir + 'Data/Output/'
debugBaseDir      = thesisBaseDir + 'Data/Debug/'
postProcessingDir = thesisBaseDir + 'Code/Postprocessing/'

#------------------------------------------------------------------------------|
# Experiment parameters                                                        |
#  * riskFreeRate: risk-free rate available on the market                      |
#  * deltaP: proportional transaction costs                                    |
#  * deltaF: fixed transaction costs                                           |
#  * deltaS: short-selling fee                                                 |
#  * numDaysObserved: number of past days observed by the agent                |
#  * lambda: decay factor for policy gradient algorithms                       |
#  * alphaConstActor: constant factor for the actor's decaying learning rate   |
#  * alphaExpActor: exponent for the actor's decaying learning rate            |
#  * alphaConstCritic: constant factor for the critic's decaying learning rate |
#  * alphaExpCritic: exponent for the critic's decaying learning rate          |
#  * alphaConstBaseline: constant factor for the baseline's learning rate      |
#  * alphaExpBaseline: exponent for the baseline's learning rate               |
#  * numExperiments: number of independent experiments to run                  |
#  * numEpochs: number of training epochs for each experiment                  |
#  * numTrainingSteps: number of time-steps used for training                  |
#  * numTestSteps: number of time-steps used for backtest                      |
#------------------------------------------------------------------------------|

params = {'riskFreeRate'      : 0.0,
          'deltaP'            : 0.0000,
          'deltaF'            : 0.0,
          'deltaS'            : 0.0050,
          'numDaysObserved'   : 5,
          'lambda'            : 0.1,
          'alphaConstActor'   : 0.1,
          'alphaExpActor'     : 0.8,
          'alphaConstCritic'  : 0.1,
          'alphaExpCritic'    : 0.7,
          'alphaConstBaseline': 0.1,
          'alphaExpBaseline'  : 0.6,
          'numExperiments'    : 10,
          'numEpochs'         : 1000,
          'numTrainingSteps'  : 7000,
          'numTestSteps'      : 2000}


#------------------------------------------------------------------------------------------------------|
# Framework selection:                                                                                 |
#  * riskSensitive: if True (resp. False), use risk-sensitive (resp. risk-neutral) learning algorithms |
#  * synthetic: if True (resp. False) use synthetic (resp. historical) asset returns                   |
#  * multiAsset: if True (resp. False) run the multi-asset (resp. single asset) case                   |
#------------------------------------------------------------------------------------------------------|

riskSensitive = True
synthetic     = True
multiAsset    = False

if not synthetic and multiAsset:
    raise ValueError('ERROR: multi asset case not implemented for historical data.')

#===============================================================================

#-------------------------------------------------------------------------------
# Do not modify the code below this line
#-------------------------------------------------------------------------------

#---------#
# Imports #
#---------#

import sys
import os
import errno

#-------------------#
# utility functions #
#-------------------#

def createDirectory(dirPath):
    """ Create directory at a given path (absolute).

    Args:
        dirPath (str): absolute path for new directory.
    """
    if not os.path.exists(os.path.expanduser(dirPath)):
        try:
            os.makedirs(os.path.expanduser(dirPath))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


#--------------------------------------#
# Input, Output and Debug destinations #
#--------------------------------------#

if synthetic:
    if not multiAsset:
        inputFile = 'synthetic.csv'
    else:
        inputFile = 'cointegrated.csv'
else:
    if not multiAsset:
        inputFile = 'historical_single.csv'
    else:
        inputFile = 'historical_multi.csv'

inputFilePath = inputBaseDir + inputFile
print inputFilePath

experimentCode = ('Multi_' if multiAsset else 'Single_') + \
                 ('Synth_' if synthetic else 'Hist_') + \
                 ('RS_' if riskSensitive else 'RN_') + \
                 'P' + str(int(params['deltaP'] * 10000)) + '_' + \
                 'F' + str(int(params['deltaF'] * 10000)) + '_' + \
                 'S' + str(int(params['deltaS'] * 10000)) + '_' + \
                 'N' + str(params['numDaysObserved'])

outputDir = outputBaseDir + experimentCode + '/'
debugDir  = debugBaseDir + experimentCode + '/'

#-------------------------------#
# Write parameters to .pot file #
#-------------------------------#

parametersFile = paramBaseDir + experimentCode + '.pot'
with open(os.path.expanduser(parametersFile), 'w+') as f:
    for key, value in params.items():
        f.write('%s = %s\n' % (key, value))

#----------------------------------#
# Define list of algorithms to run #
#----------------------------------#

if riskSensitive:
    algorithmsList = ['RSPGPE', 'RSNPGPE']
else:
    algorithmsList = ['ARAC', 'PGPE', 'NPGPE']

#-------------------------#
# Run learning algorithms #
#-------------------------#

if multiAsset:
    execPath = thesisBaseDir + 'Code/Thesis/examples/main_multiple'

    # TODO: Remove this line when other algorithms will be implemented as well
    algorithmsList = ['PGPE']
else:
    execPath = thesisBaseDir + 'Code/Thesis/examples/main_thesis'

for algo in algorithmsList:

    outputDirAlgo = outputDir + algo + '/'
    createDirectory(outputDirAlgo)

    debugDirAlgo  = debugDir + algo + '/'
    createDirectory(debugDirAlgo)

    os.system(execPath +
              " -a " + algo +
              " -p " + parametersFile +
              " -i " + inputFilePath +
              " -o " + outputDirAlgo +
              " -d " + debugDirAlgo)

#-------------------------#
# Postprocessing analysis #
#-------------------------#

sys.path.insert(0, "../Code/Postprocessing/")
from postprocessing import postprocessing

postprocessing(debugDir, outputDir)










