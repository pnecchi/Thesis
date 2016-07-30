# -*- coding: utf-8 -*-

################################################################################
# Description: Python script to analyze the results of the asset allocation exp.
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 24 lug 2016 21:31:25 BST
################################################################################

#--------#
# Import #
#--------#

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
import errno
import ffn

# General settings
matplotlib.style.use('seaborn-colorblind')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 10),
          'figure.facecolor': 'white',
          'figure.edgecolor': 'black',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)


#-----------------------#
# Algorithms considered #
#-----------------------#

algorithms = set(['ARAC', 'PGPE', 'NPGPE', 'RSARAC', 'RSPGPE', 'RSNPGPE'])


#-------------------#
# Utility functions #
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


#-----------#
# Functions #
#-----------#

def analyzeTransactionCostsImpact():

    propCostValues = [0, 1, 5, 10, 20, 50]

    dfTransactionCost = pd.DataFrame(index=propCostValues)

    filesList = ['~/Documents/University/6_Anno_Poli/7_Thesis/Data/Output/Single_Synth_RS_P' +
                 v + '_F0_S0_N5/Statistics/backtest.csv' for v in propCostValues]



    for v, f in zip(propCostValues, filesList):

        dfStat = pd.read_csv(os.path.expanduser(f))




