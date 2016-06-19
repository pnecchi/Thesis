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

# Compute cumulative profit
logReturns = df['logReturn'].values
cumReturn = 100.0 * np.exp(np.cumsum(logReturns))

fig = plt.figure(figsize=(15,15), facecolor='white', edgecolor='black')

ax1 = fig.add_subplot(211)
ax1.set_title('Backtest Performances of Trading Strategy')
ax1.plot(np.arange(len(cumReturn)), assetCumReturns, label='Buy-Hold', lw=2)
ax1.plot(np.arange(len(cumReturn)), cumReturn, label='ARRSAC', lw=2)
ax1.plot(np.arange(len(cumReturn)), 100.0 * np.ones(cumReturn.shape), lw=2)
ax1.set_ylabel('Cumulative Returns')
ax1.grid()
ax1.legend(loc=2)

ax2 = fig.add_subplot(212)
ax2.plot(np.arange(len(alloc)), alloc, lw=2)
ax2.set_ylabel('Allocation')
ax2.set_xlabel('Time Step')

plt.show()
