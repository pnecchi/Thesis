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
import matplotlib

# General settings
matplotlib.style.use('seaborn-colorblind')


#-----------#
# Functions #
#-----------#

def analyzeConvergence(filesList, algorithmName):
    """ Aggregate the convergence information of a series of independent
    experiments of a certain learning algorithms.

    Args:
        filesList (list of str): list of the files of convergence information

    Returns:
        dfReward (pd.DataFrame): dataframe containing the aggregate average reward
        dfStddev (pd.DataFrame): datraframe containing the aggregate standard dev
        dfSharpe (pd.DataFrame): dataframe containing the aggreagate Sharpe ratio
    """

    # Initialize output dataframes
    temp = pd.read_csv(filesList[0], index_col=0)
    c1 = algorithmName
    c2 = algorithmName + '_delta'
    dfReward = pd.DataFrame(index=temp.index, columns=[c1, c2])
    dfStddev = pd.DataFrame(index=temp.index, columns=[c1, c2])
    dfSharpe = pd.DataFrame(index=temp.index, columns=[c1, c2])

    # Temporary variables
    rSum   = np.zeros(len(temp))
    r2Sum  = np.zeros(len(temp))
    sSum   = np.zeros(len(temp))
    s2Sum  = np.zeros(len(temp))
    shSum  = np.zeros(len(temp))
    sh2Sum = np.zeros(len(temp))

    # For all the files
    for f in filesList:
        df = pd.read_csv(filesList[0], index_col=0)
        rSum += df['average'].values
        r2Sum += df['average'].values * df['average'].values
        sSum += df['stdev'].values
        s2Sum += df['stdev'].values * df['stdev'].values
        shSum += df['sharpe'].values
        sh2Sum += df['sharpe'].values * df['sharpe'].values

    # Compute statistics
    nExperiments = len(filesList)
    meanReward = rSum / float(nExperiments)
    meanStddev = sSum / float(nExperiments)
    meanSharpe = shSum / float(nExperiments)
    deltaReward = r2Sum / float(nExperiments) - meanReward * meanReward
    deltaStddev = np.sqrt(s2Sum / float(nExperiments) - meanStddev * meanStddev)
    deltaSharpe = np.sqrt(sh2Sum / float(nExperiments) - meanSharpe * meanSharpe)
    dfReward[c1] = meanReward
    dfStddev[c1] = meanStddev
    dfSharpe[c1] = meanSharpe
    dfReward[c2] = deltaReward
    dfStddev[c2] = deltaStddev
    dfSharpe[c2] = deltaStddev

    print meanReward
    print deltaReward

    print dfReward
    print dfStddev
    print dfSharpe

    # Return
    return dfReward, dfStddev, dfSharpe


def compareAlgorithmConvergence(debugDir):

    dfReward = pd.DataFrame()
    dfStddev = pd.DataFrame()
    dfSharpe = pd.DataFrame()

    algorithmsList = []

    for subdir, dirs, files in os.walk(debugDir):

        # Retrieve algorithm name
        algorithmName = subdir[::-1].split('/', 1)[0][::-1]
        if len(algorithmName) < 7:
            algorithmsList += [algorithmName]

        # Retrieve debug files for the current algorithm
        filesList = [os.path.join(subdir, f) for f in files]

        if len(filesList) > 0:
            # Compute aggregate convergence statistics for the current algorithm
            dfRewardAlgo, dfStddevAlgo, dfSharpeAlgo = analyzeConvergence(filesList, algorithmName)

            # Merge results
            dfReward = pd.concat([dfReward, dfRewardAlgo], axis=1)
            dfStddev = pd.concat([dfStddev, dfStddevAlgo], axis=1)
            dfSharpe = pd.concat([dfSharpe, dfSharpeAlgo], axis=1)

    fig, ax = plt.subplots()
    dfReward[algorithmsList].plot(lw=3, ax=ax)

    algorithmsListDelta = [algo + '_delta' for algo in algorithmsList]
    dfRewardUpperBound = pd.DataFrame(dfReward[algorithmsList].values + 2.0 * dfReward[algorithmsListDelta].values,
                                      columns=algorithmsList, index=dfReward.index)
    dfRewardLowerBound = pd.DataFrame(dfReward[algorithmsList].values - 2.0 * dfReward[algorithmsListDelta].values,
                                      columns=algorithmsList, index=dfReward.index)

    dfRewardUpperBound.plot(lw=2, ls='--', ax=ax)
    dfRewardLowerBound.plot(lw=2, ls='--', ax=ax)
    plt.grid(True)
    plt.show()


compareAlgorithmConvergence(os.path.expanduser('~/Documents/University/6_Anno_Poli/7_Thesis/Data/Debug/Single_Synth_RN_P0_F0_S0_N5'))



