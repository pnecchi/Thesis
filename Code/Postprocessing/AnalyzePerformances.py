################################################################################
# Description: Script that analyzes the backtest performances of a trading
#              strategy.
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        dom 19 giu 2016 11:55:24 CEST
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ffn
import matplotlib
matplotlib.style.use('seaborn-colorblind')

##############
# Parameters #
##############

inputDir = '../../Data/Debug/'
outputDir = '../../Data/Output/'
nExperiments = 5

##########################################
# Plot cumulative returns and allocation #
##########################################

# Read backtest data
df = pd.read_csv(inputDir + 'backtestExperiment0.csv')

# Initialize returns dataframe
dfPrices = pd.DataFrame(index = pd.date_range(start='01/01/2010',
                                              periods=len(df)+1))

# Compute buy-hold cumulative profits
assetSimpleReturns = df['r_1'].values
assetCumReturns = np.zeros(len(assetSimpleReturns) + 1)
assetCumReturns[0] = 0.0
assetCumReturns[1:] = 100.0 * (np.cumprod(1.0 + assetSimpleReturns) - 1.0)
dfPrices['Buy and Hold'] = 100.0 * (1.0 + assetCumReturns/100.0)

# Extract allocation in the risky asset
alloc = df['a_1'].values

fig = plt.figure(figsize=(22,15), facecolor='white', edgecolor='black')

ax1 = fig.add_subplot(211)
ax1.set_title('Backtest Performances of Trading Strategy')
ax1.plot(np.arange(len(assetCumReturns)), assetCumReturns, label='Buy-Hold', lw=2)
ax1.plot(np.arange(len(assetCumReturns)), np.zeros(assetCumReturns.shape), lw=1, c='black')

# Compute mean cumulative returns from various experiments
cumReturn = np.zeros(len(assetSimpleReturns) + 1)
sumCumReturn = np.zeros(cumReturn.shape)
sumSquaresCumReturn = np.zeros(cumReturn.shape)
sumAlloc = np.zeros(cumReturn.shape)
sumSquaresAlloc = np.zeros(cumReturn.shape)

for i in xrange(nExperiments):

    # Read i-th experiment results
    df = pd.read_csv(inputDir + 'backtestExperiment' + str(i) + '.csv')

    # Compute cumulative profit
    logReturns = df['logReturn'].values
    cumReturn[0] = 0.0
    cumReturn[1:] = 100.0 * (np.exp(np.cumsum(logReturns)) - 1.0)
    dfPrices['Experiment ' + str(i)] = 100.0 * (1.0 + cumReturn/100.0)
    sumCumReturn += cumReturn
    sumSquaresCumReturn += cumReturn * cumReturn

    # Plot cumulative profit for this experiment
    ax1.plot(np.arange(len(cumReturn)), cumReturn, lw=1, ls='--', c='grey')

meanCumReturn = sumCumReturn / float(nExperiments)
stddevCumReturn = np.sqrt(sumSquaresCumReturn / float(nExperiments)
                          - meanCumReturn * meanCumReturn)

dfPrices['Average Experiment'] = 100 * (1.0 + meanCumReturn / 100.0)

ax1.plot(np.arange(len(meanCumReturn)), meanCumReturn, lw=2, label='NPGPE', c='orangered')
ax1.plot(np.arange(len(meanCumReturn)), meanCumReturn + 2.0 * stddevCumReturn, lw=2, ls=':', c='orangered')
ax1.plot(np.arange(len(meanCumReturn)), meanCumReturn - 2.0 * stddevCumReturn, lw=2, ls=':', c='orangered')

ax1.set_ylabel('Cumulative Returns')
ax1.grid()
ax1.legend(loc=2)

ax2 = fig.add_subplot(212)
ax2.plot(np.arange(len(alloc)), alloc, lw=2)
ax2.set_ylabel('Allocation')
ax2.set_xlabel('Time Step')
ax2.set_ylim((-1.1, 1.1))

plt.show()

# Compute strategies stats
stats = dfPrices.calc_stats()
stats.display()
stats.to_csv(path = outputDir + 'backtest.csv')
