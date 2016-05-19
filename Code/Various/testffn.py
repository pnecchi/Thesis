################################################################################
# Description: Test of ffn and bt financial libraries
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        lun 16 mag 2016 19:35:39 CEST
################################################################################

import ffn                       # Financial data manipulation
import bt                        # Backtesting
import matplotlib.pyplot as plt  # Plotting

# Parameters
tickers = ['GS', 'JPM']
startDate = '2000-01-01'
endDate = '2015-01-01'

# Retrieve time series
data = ffn.get(tickers,
               start=startDate,
               end=endDate,
               common_dates=False,
               forward_fill=True)

# Compute logreturns
returns = data.to_log_returns().dropna()

# Compute performance
perf = data.calc_stats()
perf.plot()
plt.show()
print perf.display()

# Create the strategy
s = bt.Strategy('s1', [bt.algos.RunMonthly(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighEqually(),
                       bt.algos.Rebalance()])

test = bt.Backtest(s, data)
res = bt.run(test)

res.plot(lw=1.5)
plt.show()
print res.display()


