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
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
import errno


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
    statsConsidered = ['Reallocation Freq', 'Total Return', 'Daily Sharpe']

    dfBuyHold = pd.DataFrame(index=propCostValues, columns=statsConsidered)
    dfTransactionCostPGPE = pd.DataFrame(index=propCostValues, columns=statsConsidered)
    dfTransactionCostNPGPE = pd.DataFrame(index=propCostValues, columns=statsConsidered)

    filesList = ['~/Documents/University/6_Anno_Poli/7_Thesis/Data/Output/Single_Synth_RN_P' +
                 str(v) + '_F0_S0_N5/Statistics/backtest.csv' for v in propCostValues]

    outputDir = os.path.expanduser('~/Documents/University/6_Anno_Poli/7_Thesis/Data/Output/ImpactTransactionCosts/')

    for v, f in zip(propCostValues, filesList):

        dfStat = pd.read_csv(os.path.expanduser(f), index_col=0)

        if v == propCostValues[0]:
            dfBuyHold.ix[:, statsConsidered] = dfStat.loc[statsConsidered, 'Buy and Hold'].values

        dfTransactionCostPGPE.loc[v, statsConsidered] = dfStat.loc[statsConsidered, 'PGPE'].values
        dfTransactionCostNPGPE.loc[v, statsConsidered] = dfStat.loc[statsConsidered, 'NPGPE'].values

    # Change units of statistics
    dfBuyHold['Reallocation Freq']              = 100.0 * dfBuyHold['Reallocation Freq']
    dfBuyHold['Total Return']                   = 100.0 * dfBuyHold['Total Return']
    dfTransactionCostPGPE['Reallocation Freq']  = 100.0 * dfTransactionCostPGPE['Reallocation Freq']
    dfTransactionCostPGPE['Total Return']       = 100.0 * dfTransactionCostPGPE['Total Return']
    dfTransactionCostNPGPE['Reallocation Freq'] = 100.0 * dfTransactionCostNPGPE['Reallocation Freq']
    dfTransactionCostNPGPE['Total Return']      = 100.0 * dfTransactionCostNPGPE['Total Return']

    # Plot results
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    ax1 = dfBuyHold['Reallocation Freq'].plot(color='gray', lw=3, label='Buy and Hold')
    dfTransactionCostPGPE['Reallocation Freq'].plot(color='seagreen', lw=3, label='PGPE', ax=ax1)
    dfTransactionCostNPGPE['Reallocation Freq'].plot(color='darkorange', lw=3, label='NPGPE', ax=ax1)
    ax1.set_xlabel('Proportional Transaction Costs [bps]')
    ax1.set_ylabel('Reallocation Frequency [%]')
    ax1.grid(True)
    ax1.legend(['Buy and Hold', 'PGPE', 'NPGPE'], loc='upper right')
    plt.subplot(1, 3, 2)
    ax2 = dfBuyHold['Total Return'].plot(color='gray', lw=3, label='Buy and Hold')
    dfTransactionCostPGPE['Total Return'].plot(color='seagreen', lw=3, label='PGPE', ax=ax2)
    dfTransactionCostNPGPE['Total Return'].plot(color='darkorange', lw=3, label='NPGPE', ax=ax2)
    ax2.set_xlabel('Proportional Transaction Costs [bps]')
    ax2.set_ylabel('Cumulative Profit [%]')
    ax2.grid(True)
    plt.subplot(1, 3, 3)
    ax3 = dfBuyHold['Daily Sharpe'].plot(color='gray', lw=3, label='Buy and Hold')
    dfTransactionCostPGPE['Daily Sharpe'].plot(color='seagreen', lw=3, label='PGPE', ax=ax3)
    dfTransactionCostNPGPE['Daily Sharpe'].plot(color='darkorange', lw=3, label='NPGPE', ax=ax3)
    ax3.set_xlabel('Proportional Transaction Costs [bps]')
    ax3.set_ylabel('Annualized Sharpe Ratio')
    ax3.grid(True)
    plt.tight_layout()

    if outputDir is not None:
        createDirectory(outputDir)
        plt.savefig(outputDir + 'impact_transaction_costs.eps', format='eps', dpi=1200)
    else:
        plt.show()

    # Print dataframes to output directory
    dfBuyHold.to_csv(outputDir + 'impact_transaction_costs_BH.csv')
    dfTransactionCostPGPE.to_csv(outputDir + 'impact_transaction_costs_PGPE.csv')
    dfTransactionCostNPGPE.to_csv(outputDir + 'impact_transaction_costs_NPGPE.csv')


def analyzeShortSellingFeesImpact():

    shortSellingFeeValues = [0, 1, 5, 10, 20, 50]
    statsConsidered = ['Short Freq', 'Total Return', 'Daily Sharpe']

    dfBuyHold = pd.DataFrame(index=shortSellingFeeValues, columns=statsConsidered)
    dfTransactionCostPGPE = pd.DataFrame(index=shortSellingFeeValues, columns=statsConsidered)
    dfTransactionCostNPGPE = pd.DataFrame(index=shortSellingFeeValues, columns=statsConsidered)

    filesList = ['~/Documents/University/6_Anno_Poli/7_Thesis/Data/Output/Single_Synth_RN_P0_F0_S' +
                 str(v) + '_N5/Statistics/backtest.csv' for v in shortSellingFeeValues]

    outputDir = os.path.expanduser('~/Documents/University/6_Anno_Poli/7_Thesis/Data/Output/ImpactShortSellingFee/')

    for v, f in zip(shortSellingFeeValues, filesList):

        dfStat = pd.read_csv(os.path.expanduser(f), index_col=0)

        if v == shortSellingFeeValues[0]:
            dfBuyHold.ix[:, statsConsidered] = dfStat.loc[statsConsidered, 'Buy and Hold'].values

        dfTransactionCostPGPE.loc[v, statsConsidered] = dfStat.loc[statsConsidered, 'PGPE'].values
        dfTransactionCostNPGPE.loc[v, statsConsidered] = dfStat.loc[statsConsidered, 'NPGPE'].values

    # Change units of statistics
    dfBuyHold['Short Freq']                = 100.0 * dfBuyHold['Short Freq']
    dfBuyHold['Total Return']              = 100.0 * dfBuyHold['Total Return']
    dfTransactionCostPGPE['Short Freq']    = 100.0 * dfTransactionCostPGPE['Short Freq']
    dfTransactionCostPGPE['Total Return']  = 100.0 * dfTransactionCostPGPE['Total Return']
    dfTransactionCostNPGPE['Short Freq']   = 100.0 * dfTransactionCostNPGPE['Short Freq']
    dfTransactionCostNPGPE['Total Return'] = 100.0 * dfTransactionCostNPGPE['Total Return']

    # Plot results
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    ax1 = dfBuyHold['Short Freq'].plot(color='gray', lw=3, label='Buy and Hold')
    dfTransactionCostPGPE['Short Freq'].plot(color='seagreen', lw=3, label='PGPE', ax=ax1)
    dfTransactionCostNPGPE['Short Freq'].plot(color='darkorange', lw=3, label='NPGPE', ax=ax1)
    ax1.set_xlabel('Short-Selling Fees [bps]')
    ax1.set_ylabel('Short-Selling Frequency [%]')
    ax1.grid(True)
    ax1.legend(['Buy and Hold', 'PGPE', 'NPGPE'], loc='upper right')
    plt.subplot(1, 3, 2)
    ax2 = dfBuyHold['Total Return'].plot(color='gray', lw=3, label='Buy and Hold')
    dfTransactionCostPGPE['Total Return'].plot(color='seagreen', lw=3, label='PGPE', ax=ax2)
    dfTransactionCostNPGPE['Total Return'].plot(color='darkorange', lw=3, label='NPGPE', ax=ax2)
    ax2.set_xlabel('Short-Selling Fees [bps]')
    ax2.set_ylabel('Cumulative Profit [%]')
    ax2.grid(True)
    plt.subplot(1, 3, 3)
    ax3 = dfBuyHold['Daily Sharpe'].plot(color='gray', lw=3, label='Buy and Hold')
    dfTransactionCostPGPE['Daily Sharpe'].plot(color='seagreen', lw=3, label='PGPE', ax=ax3)
    dfTransactionCostNPGPE['Daily Sharpe'].plot(color='darkorange', lw=3, label='NPGPE', ax=ax3)
    ax3.set_xlabel('Short-Selling Fees [bps]')
    ax3.set_ylabel('Annualized Sharpe Ratio')
    ax3.grid(True)
    plt.tight_layout()

    if outputDir is not None:
        createDirectory(outputDir)
        plt.savefig(outputDir + 'impact_short_selling_fees.eps', format='eps', dpi=1200)
    else:
        plt.show()

    # Print dataframes to output directory
    dfBuyHold.to_csv(outputDir + 'impact_short_selling_fees_BH.csv')
    dfTransactionCostPGPE.to_csv(outputDir + 'impact_short_selling_fees_PGPE.csv')
    dfTransactionCostNPGPE.to_csv(outputDir + 'impact_short_selling_fees_NPGPE.csv')

if __name__ == '__main__':
    analyzeTransactionCostsImpact()
    analyzeShortSellingFeesImpact()

