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

# Colors used
colors = ['black',
          'dimgrey',
          'steelblue',
          'lightsteelblue']

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


def compareAlgorithmConvergence(debugDir, imagesDir=None):
    """ Compare the convergence properties of several learning algorithms. The
    function produces images and csv summaries of the analysis in the given
    directories.

    Args:
        outputDir (str): output directory.
        imagesDir (str): images directory.
    """

    dfReward = pd.DataFrame()
    dfStddev = pd.DataFrame()
    dfSharpe = pd.DataFrame()

    algorithmsList = []

    for subdir, dirs, files in os.walk(debugDir):

        # Retrieve algorithm name
        algorithmName = subdir[::-1].split('/', 1)[0][::-1]

        if algorithmName not in algorithms:
            continue
        else:
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

    fig = plt.figure(figsize=(20,10), facecolor='white', edgecolor='black')

    # Average reward
    ax1 = fig.add_subplot(131)
    (10000.0 * dfReward[algorithmsList]).plot(lw=3, color=colors[1:], ax=ax1)
    dfRewardUpperBound = pd.DataFrame(1e4 * (dfReward[algorithmsList].values + 2.0 * dfReward[algorithmsListDelta].values),
                                      columns=algorithmsList, index=dfReward.index)
    dfRewardLowerBound = pd.DataFrame(1e4 * (dfReward[algorithmsList].values - 2.0 * dfReward[algorithmsListDelta].values),
                                      columns=algorithmsListDelta, index=dfReward.index)
    dfRewardUpperBound.plot(lw=2, ls='--', color=colors[1:], ax=ax1)
    dfRewardLowerBound.plot(lw=2, ls='--', color=colors[1:], ax=ax1)

    ax1.set_ylabel('Daily Average Reward [bps]')
    ax1.set_xlabel('Training Epoch')
    ax1.legend(algorithmsList, loc='upper left')
    plt.grid(True)

    # Reward standard deviation
    ax2 = fig.add_subplot(132)
    (1e4 * dfStddev[algorithmsList]).plot(lw=3, color=colors[1:], legend=False, ax=ax2)
    dfStddevUpperBound = pd.DataFrame(1e4 * (dfStddev[algorithmsList].values + 2.0 * dfStddev[algorithmsListDelta].values),
                                      columns=algorithmsList, index=dfStddev.index)
    dfStddevLowerBound = pd.DataFrame(1e4 * (dfStddev[algorithmsList].values - 2.0 * dfStddev[algorithmsListDelta].values),
                                      columns=algorithmsListDelta, index=dfStddev.index)
    dfStddevUpperBound.plot(lw=2, ls='--', color=colors[1:], legend=False, ax=ax2)
    dfStddevLowerBound.plot(lw=2, ls='--', color=colors[1:], legend=False, ax=ax2)
    ax2.set_title('Convergence of Learning Process', fontsize=18)
    ax2.set_ylabel('Daily Reward Standard Deviation [bps]')
    ax2.set_xlabel('Training Epoch')
    plt.grid(True)

    # Sharpe ratio
    ax3 = fig.add_subplot(133)
    (np.sqrt(252) * dfSharpe[algorithmsList]).plot(lw=3, color=colors[1:], legend=False, ax=ax3)
    dfSharpeUpperBound = pd.DataFrame(np.sqrt(252) * (dfSharpe[algorithmsList].values + 2.0 * dfSharpe[algorithmsListDelta].values),
                                      columns=algorithmsList, index=dfSharpe.index)
    dfSharpeLowerBound = pd.DataFrame(np.sqrt(252) * (dfSharpe[algorithmsList].values - 2.0 * dfSharpe[algorithmsListDelta].values),
                                      columns=algorithmsListDelta, index=dfSharpe.index)
    dfSharpeUpperBound.plot(lw=2, ls='--', color=colors[1:], legend=False, ax=ax3)
    dfSharpeLowerBound.plot(lw=2, ls='--', color=colors[1:], legend=False, ax=ax3)
    ax3.set_ylabel('Annualized Sharpe Ratio')
    ax3.set_xlabel('Training Epoch')
    plt.grid(True)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    if imagesDir is not None:
        createDirectory(imagesDir)
        plt.savefig(imagesDir + 'convergence.eps', format='eps', dpi=1200)
    else:
        plt.show()


def analyzePerformance(filesList, algorithmName):
    """ Aggregate the performances of a series of independent experiments of a
        certain learning algorithms.

    Args:
        filesList (list of str): list of the files of convergence information
        algorithmName (str):     name of the learning algorithm

    Returns:
        dfBuyHold    (pd.DataFrame): dataframe containg the cumulative returns of the buy & hold strat
        dfCumProfit  (pd.DataFrame): dataframe containing the aggregate cumulative profits
        dfStatistics (pd.DataFrame): dataframe containing the backtest statistics
    """

    # Initialize output dataframes
    temp = pd.read_csv(os.path.expanduser(filesList[0]))
    dfAssetReturn   = pd.Series(temp['r_1'], index=np.arange(1, len(temp)+1))
    dfAllocationExp = pd.DataFrame(index=temp.index)
    dfLogReturnExp  = pd.DataFrame(index=np.arange(1, len(temp)+1))

    # For all the files
    for f in filesList:
        expName = f[::-1].split('/', 1)[0][::-1][:-4].encode("utf-8")
        df = pd.read_csv(os.path.expanduser(f))
        df.set_index(np.arange(1, len(temp)+1), inplace=True)
        dfAllocationExp[expName] = df['a_1']
        dfLogReturnExp[expName]  = df['logReturn']

    # Compute cumulative profits
    dfBuyHold   = pd.Series(100.0 * (np.cumprod(dfAssetReturn.values + 1.0) - 1.0), index=dfAssetReturn.index)
    dfCumProfit = pd.DataFrame(100.0 * (np.exp(dfLogReturnExp.cumsum()) - 1.0), index=dfLogReturnExp.index, columns=dfLogReturnExp.columns)
    dfBuyHold.loc[0]= 0.0
    dfCumProfit.loc[0, :] = 0.0
    dfBuyHold.sort_index(inplace=True)
    dfCumProfit.sort_index(inplace=True)

    # Price dataframe for computing strategy statistics
    dfPricesExp = pd.DataFrame(index=pd.date_range(start='01/01/2000', periods=len(dfBuyHold)),
                               columns=dfCumProfit.columns)
    dfPricesExp['Buy and Hold'] = 100.0 * (1.0 + dfBuyHold.values / 100.0)
    dfPricesExp[dfCumProfit.columns] = 100.0 * (1.0 + dfCumProfit.values / 100.0)

    # Compute aggregate information
    dfPerf = pd.DataFrame(index=dfCumProfit.index, columns=[algorithmName, algorithmName + '_delta'])
    dfPerf[algorithmName] = dfCumProfit.mean(axis=1)
    dfPerf[algorithmName + '_delta'] = dfCumProfit.std(axis=1)

    # Compute strategies stats
    dfStatistics = dfPricesExp.calc_stats()

    # Compute frequency of reallocation
    reallocationFrequency = (dfAllocationExp.diff().dropna() != 0).mean(axis=0).mean(axis=0)
    shortFrequency = (dfAllocationExp < 0).mean(axis=0).mean(axis=0)

    return dfBuyHold, dfPerf, dfStatistics, reallocationFrequency, shortFrequency


def compareAlgorithmPerformance(outputDir, imagesDir):
    """ Compare the backtest performances of several learning algorithms. The
    function produces images and csv summaries of the analysis in the given
    directories.

    Args:
        outputDir (str): output directory.
        imagesDir (str): images directory.
    """
    dfPerf = pd.DataFrame()
    dfStat = pd.DataFrame(index=['Total Return', 'Daily Sharpe', 'Monthly Sharpe', 'Yearly Sharpe', 'Max Drawdown',
                                 'Avg Drawdown', 'Avg Up Month', 'Avg Down Month', 'Win Year %', 'Win 12m %'])

    algorithmsList = []

    for subdir, dirs, files in os.walk(outputDir):

        # Retrieve algorithm name
        algorithmName = subdir[::-1].split('/', 1)[0][::-1]

        if algorithmName not in algorithms:
            continue
        else:
            algorithmsList += [algorithmName]

        # Retrieve debug files for the current algorithm
        filesList = [os.path.join(subdir, f) for f in files]

        if len(filesList) > 0:
            # Compute aggregate performance statistics
            dfBuyHold, dfPerfAlgo, dfStatAlgo, reallocationFreqAlgo, shortFreqAlgo = \
                analyzePerformance(filesList, algorithmName)

            # Merge results
            dfPerf = pd.concat([dfPerf, dfPerfAlgo], axis=1)

            # Write backtest statistics to .csv file
            createDirectory(outputDir + 'Statistics/')
            dfStatAlgo.to_csv(path = os.path.expanduser(outputDir + 'Statistics/backtest' + algorithmName + '.csv'))

            # Extract statistics
            dfStatRed = extractBacktestStatistics(dfStatAlgo)
            dfStat['Buy and Hold'] = dfStatRed['Buy and Hold']
            dfStat.loc['Reallocation Freq', 'Buy and Hold'] = 0.0
            dfStat.loc['Short Freq', 'Buy and Hold'] = 0.0
            dfStat[algorithmName]  = dfStatRed.ix[:, 1:].mean(axis=1)
            dfStat.loc['Reallocation Freq', algorithmName] = reallocationFreqAlgo
            dfStat.loc['Short Freq', algorithmName] = shortFreqAlgo

    # Plot performance
    algorithmsListDelta = [algo + '_delta' for algo in algorithmsList]
    colorsList = ['steelblue', 'darkorange', 'seagreen']
    plt.figure()
    ax = dfBuyHold.plot(title='Performance of Learning Algorithms', color=colors[0], lw=3)
    dfPerf[algorithmsList].plot(title='Performance of Learning Algorithms', color=colors[1:], lw=3, ax=ax)
    dfPerfLowerBound = pd.DataFrame(dfPerf[algorithmsList].values - 2.0 * dfPerf[algorithmsListDelta].values,
                                    columns=algorithmsList, index=dfPerf.index)
    dfPerfUpperBound = pd.DataFrame(dfPerf[algorithmsList].values + 2.0 * dfPerf[algorithmsListDelta].values,
                                    columns=algorithmsList, index=dfPerf.index)
    dfPerfLowerBound.plot(ax=ax, color=colors[1:], ls='--', lw=2)
    dfPerfUpperBound.plot(ax=ax, color=colors[1:], ls='--', lw=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Profit [%]')
    ax.legend(['Buy and Hold'] + algorithmsList, loc='upper left')
    plt.grid(True)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    if imagesDir is not None:
        createDirectory(imagesDir)
        plt.savefig(imagesDir + 'performance.eps', format='eps', dpi=1200)
    else:
        plt.show()

    # Aggregate statistics
    dfStat.to_csv(outputDir + 'Statistics/backtest.csv')


def extractBacktestStatistics(dfStatAlgo):
    """ Extract backtest statistics from ffn stats object and store them in a
    pandas DataFrame, for easier aggregation.

    Args:
        dfStatAlgo (ffn.stats): ffn stats container.
    """
    dfStatRed = pd.DataFrame()
    for exp, stat in dfStatAlgo.iteritems():
        dfStatRed.loc['Total Return', exp]   = stat.total_return
        dfStatRed.loc['Daily Sharpe', exp]   = stat.daily_sharpe
        dfStatRed.loc['Monthly Sharpe', exp] = stat.monthly_sharpe
        dfStatRed.loc['Yearly Sharpe', exp]  = stat.yearly_sharpe
        dfStatRed.loc['Max Drawdown', exp]   = stat.max_drawdown
        dfStatRed.loc['Avg Drawdown', exp]   = stat.avg_drawdown
        dfStatRed.loc['Avg Up Month', exp]   = stat.avg_up_month
        dfStatRed.loc['Avg Down Month', exp] = stat.avg_down_month
        dfStatRed.loc['Win Year %', exp]     = stat.win_year_perc
        dfStatRed.loc['Win 12m %', exp]      = stat.twelve_month_win_perc
    return dfStatRed


def postprocessing(debugDir, outputDir):
    """ Postprocessing wrapper function.

    Args:
        debugDir (str): path to debug directory.
        outputDir (str): path to output directory.
    """

    # Compare algorithms convergence
    compareAlgorithmConvergence(os.path.expanduser(debugDir), os.path.expanduser(outputDir + 'Images/'))

    # Compare algorithms performances
    compareAlgorithmPerformance(os.path.expanduser(outputDir), os.path.expanduser(outputDir + 'Images/'))


if __name__ == "__main__":
    postprocessing('~/Documents/University/6_Anno_Poli/7_Thesis/Data/Debug/Single_Synth_RN_P50_F0_S0_N5/',
                   '~/Documents/University/6_Anno_Poli/7_Thesis/Data/Output/Single_Synth_RN_P50_F0_S0_N5/')

