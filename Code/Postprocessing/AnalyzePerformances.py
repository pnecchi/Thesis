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
import matplotlib
matplotlib.style.use('seaborn-colorblind')

##############
# Parameters #
##############

inputDir = '../../Data/Debug/'
nExperiments = 5

########################
# Visualize allocation #
########################

# Read backtest data
df = pd.read_csv(inputDir + 'backtestExperiment0.csv')

# Compute buy-hold cumulative profits
assetSimpleReturns = df['r_1'].values
assetCumReturns = 100.0 * np.cumprod(1.0 + assetSimpleReturns)

# Extract allocation in the risky asset
alloc = df['a_1'].values

fig = plt.figure(figsize=(15,15), facecolor='white', edgecolor='black')

ax1 = fig.add_subplot(211)
ax1.set_title('Backtest Performances of Trading Strategy')
ax1.plot(np.arange(len(assetCumReturns)), assetCumReturns, label='Buy-Hold', lw=2)
ax1.plot(np.arange(len(assetCumReturns)), 100.0 * np.ones(assetCumReturns.shape), lw=1, c='black')

# Compute mean cumulative returns from various experiments
sumCumReturn = np.zeros(alloc.shape)
sumSquaresCumReturn = np.zeros(alloc.shape)

for i in xrange(nExperiments):

    # Read i-th experiment results
    df = pd.read_csv(inputDir + 'backtestExperiment' + str(i) + '.csv')

    # Compute cumulative profit
    logReturns = df['logReturn'].values
    cumReturn = 100.0 * np.exp(np.cumsum(logReturns))
    sumCumReturn += cumReturn
    sumSquaresCumReturn += cumReturn * cumReturn

    # Plot cumulative profit for this experiment
    ax1.plot(np.arange(len(cumReturn)), cumReturn, lw=1, ls='--', c='grey')

meanCumReturn = sumCumReturn / float(nExperiments)
stddevCumReturn = np.sqrt(sumSquaresCumReturn / float(nExperiments)
                          - meanCumReturn * meanCumReturn)

ax1.plot(np.arange(len(meanCumReturn)), meanCumReturn, lw=2, label='ARRSAC', c='orangered')
ax1.plot(np.arange(len(meanCumReturn)), meanCumReturn + stddevCumReturn, lw=2, ls=':', c='orangered')
ax1.plot(np.arange(len(meanCumReturn)), meanCumReturn - stddevCumReturn, lw=2, ls=':', c='orangered')

ax1.set_ylabel('Cumulative Returns')
ax1.grid()
ax1.legend(loc=2)

ax2 = fig.add_subplot(212)
ax2.plot(np.arange(len(alloc)), alloc, lw=2)
ax2.set_ylabel('Allocation')
ax2.set_xlabel('Time Step')
ax2.set_ylim((1.1 * alloc.min(), alloc.max() * 1.1))

plt.show()
