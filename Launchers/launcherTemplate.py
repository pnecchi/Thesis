#!/usr/bin/env python
# -*- coding: utf-8 -*-

#---------#
# Imports #
#---------#

import os
import subprocess
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

#---------------------#
# Working Directories #
#---------------------#

thesisBaseDir     = '/home/pierpaolo/Documents/University/6_Anno_Poli/7_Thesis/'
paramBaseDir      = thesisBaseDir + 'Data/Parameters/'
inputBaseDir      = thesisBaseDir + 'Data/Input/'
outputBaseDir     = thesisBaseDir + 'Data/Output/'
debugBaseDir      = thesisBaseDir + 'Data/Debug/'
postProcessingDir = thesisBaseDir + 'Code/Postprocessing/'

#-----------------------#
# Experiment parameters #
#-----------------------#

params = {'riskFreeRate'      : 0.0,
          'deltaP'            : 0.0,
          'deltaF'            : 0.0,
          'deltaS'            : 0.0,
          'numDaysObserved'   : 5,
          'lambda'            : 0.9,
          'alphaConstActor'   : 0.02,
          'alphaExpActor'     : 0.75,
          'alphaConstCritic'  : 0.1,
          'alphaExpCritic'    : 0.7,
          'alphaConstBaseline': 0.2,
          'alphaExpBaseline'  : 0.65,
          'numExperiments'    : 10,
          'numEpochs'         : 500,
          'numTrainingSteps'  : 1000,
          'numTestSteps'      : 200}

riskSensitive = False
synthetic     = True
multiAsset    = False

#--------------------------------------#
# Input, Output and Debug destinations #
#--------------------------------------#

inputFile = inputBaseDir + 'synthetic.csv'

experimentCode = ('Multi_' if multiAsset else 'Single_' + \
                  'Synth_' if synthetic else 'Hist_' + \
                  'RS_' if riskSensitive else 'RN_') + \
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
    algorithmsList = ['RSARAC', 'RSPGPE', 'RSNPGPE']
else:
    algorithmsList = ['ARAC', 'PGPE', 'NPGPE']

#-------------------------#
# Run learning algorithms #
#-------------------------#

execPath = thesisBaseDir + 'Code/Thesis/examples/main'

for algo in algorithmsList:

    outputDirAlgo = outputDir + algo + '/'
    createDirectory(outputDirAlgo)

    debugDirAlgo  = debugDir + algo + '/'
    createDirectory(debugDirAlgo)

    os.system(execPath +
              " -a " + algo +
              " -p " + parametersFile +
              " -i " + inputFile +
              " -o " + outputDirAlgo +
              " -d " + debugDirAlgo)

    # subprocess.call([execPath,
                     # "-a", algo,
                     # "-p", parametersFile,
                     # "-i", inputFile,
                     # "-o", outputDirAlgo,
                     # "-d", debugDirAlgo], shell=True)











