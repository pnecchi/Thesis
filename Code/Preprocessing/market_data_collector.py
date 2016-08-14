################################################################################
# Description: Read financial time series from csv files
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        gio 05 mag 2016 18:06:22 CEST
################################################################################

import os
import csv
import pandas as pd
from pandas.io.data import DataReader


class MarketDataCollector(object):
    """Class that collects historical data from yahoo finance, preprocesses them and dumps them in .csv file. """

    # Data directory
    DATA_DIR = '~/Documents/University/6_Anno_Poli/7_Thesis/Data/Input/'

    def __init__(self, assets, start=pd.to_datetime('1990/01/01'), end=pd.to_datetime('today')):
        """Constructor.

        Parameters
        ----------
            assets: list of str
                list of bloomberg tickers
            start: pd.datetime
                start date [default: 1900/01/01]
            end: pd.datetime
                end date [default: today]
        """
        self.assets = assets
        self.start  = start
        self.end    = end
        self.df     = None

    def run(self):
        """Collects data from Yahoo finance and preprocesses them."""

        # Get reference dates from SP500
        df_ref = DataReader('SPY', 'yahoo', self.start, self.end)['Adj Close']
        df_ref.sort_index(inplace=True)

        if 'SPY' in assets:
            self.df = df_ref
            self.df.rename({'AdjClose': 'SPY'})
        else:
            self.df = pd.DataFrame(index=df_ref.index)

        # Retrieve AdjClose price for other assets
        for asset in self.assets:
            df_asset = DataReader(asset, 'yahoo', self.start, self.end)['Adj Close']
            df_asset.rename(asset, inplace=True)
            self.df = self.df.join(df_asset, how='left')

        # Fill NaN
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)

        # Compute assets simple returns
        self.df = (self.df / self.df.shift(1) - 1.0).ix[1:, :]

        # Reset indices
        self.df.reset_index(drop=True, inplace=True)

    def to_csv(self, filename='historical.csv'):
        """Dumps data in .csv file in the data directory.

        Parameters
        ----------
            filename: str
                Output file name [default: historical.csv]
        """
        with open(os.path.expanduser(self.DATA_DIR + filename), 'w') as f:
            a = csv.writer(f, delimiter=',', lineterminator='\n')
            a.writerows([[self.df.shape[0], self.df.shape[1]]])
            self.df.to_csv(f, index=False)


if __name__ == "__main__":
    start_date = pd.to_datetime('2000/01/01')
    assets = ['BMPS.MI']
    collector = MarketDataCollector(assets, start_date)
    collector.run()
    collector.to_csv()
