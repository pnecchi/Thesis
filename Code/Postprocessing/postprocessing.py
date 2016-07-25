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
    temp = pd.read_csv(os.path.expanduser(filesList[0]), index_col=0)
    dfRewardExp = pd.DataFrame(index=temp.index)
    dfStddevExp = pd.DataFrame(index=temp.index)
    dfSharpeExp = pd.DataFrame(index=temp.index)

    # For all the files
    for f in filesList:
        expName = f[::-1].split('/', 1)[0][::-1][:-4]
        df = pd.read_csv(os.path.expanduser(f), index_col=0)
        dfRewardExp[expName] = df['average']
        dfStddevExp[expName] = df['stdev']
        dfSharpeExp[expName] = df['sharpe']

    # Compute mean and stddev across experiments
    c1 = algorithmName
    c2 = algorithmName + '_delta'
    dfReward = pd.DataFrame(index=temp.index, columns=[c1, c2])
    dfStddev = pd.DataFrame(index=temp.index, columns=[c1, c2])
    dfSharpe = pd.DataFrame(index=temp.index, columns=[c1, c2])

    dfReward[c1] = dfRewardExp.mean(axis=1)
    dfReward[c2] = dfRewardExp.std(axis=1)
    dfStddev[c1] = dfStddevExp.mean(axis=1)
    dfStddev[c2] = dfStddevExp.std(axis=1)
    dfSharpe[c1] = dfSharpeExp.mean(axis=1)
    dfSharpe[c2] = dfSharpeExp.std(axis=1)

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

    algorithmsListDelta = [algo + '_delta' for algo in algorithmsList]
    colorsList = ['steelblue', 'darkorange', 'seagreen']

    fig = plt.figure(figsize=(15,5), facecolor='white', edgecolor='black')

    # Average reward
    ax1 = fig.add_subplot(131)
    dfReward[algorithmsList].plot(lw=3, color=colorsList, ax=ax1)
    dfRewardUpperBound = pd.DataFrame(dfReward[algorithmsList].values + 2.0 * dfReward[algorithmsListDelta].values,
                                      columns=algorithmsList, index=dfReward.index)
    dfRewardLowerBound = pd.DataFrame(dfReward[algorithmsList].values - 2.0 * dfReward[algorithmsListDelta].values,
                                      columns=algorithmsListDelta, index=dfReward.index)
    dfRewardUpperBound.plot(lw=2, ls='--', color=colorsList, ax=ax1)
    dfRewardLowerBound.plot(lw=2, ls='--', color=colorsList, ax=ax1)

    ax1.set_ylabel('Average Reward')
    ax1.set_xlabel('Training Epoch')
    ax1.legend(algorithmsList, loc='upper left')
    plt.grid(True)

    # Reward standard deviation
    ax2 = fig.add_subplot(132)
    dfStddev[algorithmsList].plot(lw=3, color=colorsList, legend=False, ax=ax2)
    dfStddevUpperBound = pd.DataFrame(dfStddev[algorithmsList].values + 2.0 * dfStddev[algorithmsListDelta].values,
                                      columns=algorithmsList, index=dfStddev.index)
    dfStddevLowerBound = pd.DataFrame(dfStddev[algorithmsList].values - 2.0 * dfStddev[algorithmsListDelta].values,
                                      columns=algorithmsListDelta, index=dfStddev.index)
    dfStddevUpperBound.plot(lw=2, ls='--', color=colorsList, legend=False, ax=ax2)
    dfStddevLowerBound.plot(lw=2, ls='--', color=colorsList, legend=False, ax=ax2)
    ax2.set_title('Convergence of Learning Process', fontsize=18)
    ax2.set_ylabel('Reward Standard Deviation')
    ax2.set_xlabel('Training Epoch')
    plt.grid(True)

    # Sharpe ratio
    ax3 = fig.add_subplot(133)
    dfSharpe[algorithmsList].plot(lw=3, color=colorsList, legend=False, ax=ax3)
    dfSharpeUpperBound = pd.DataFrame(dfSharpe[algorithmsList].values + 2.0 * dfSharpe[algorithmsListDelta].values,
                                      columns=algorithmsList, index=dfSharpe.index)
    dfSharpeLowerBound = pd.DataFrame(dfSharpe[algorithmsList].values - 2.0 * dfSharpe[algorithmsListDelta].values,
                                      columns=algorithmsListDelta, index=dfSharpe.index)
    dfSharpeUpperBound.plot(lw=2, ls='--', color=colorsList, legend=False, ax=ax3)
    dfSharpeLowerBound.plot(lw=2, ls='--', color=colorsList, legend=False, ax=ax3)
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_xlabel('Training Epoch')
    plt.grid(True)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

compareAlgorithmConvergence(os.path.expanduser('~/Documents/University/6_Anno_Poli/7_Thesis/Data/Debug/Single_Synth_RN_P0_F0_S0_N5'))



