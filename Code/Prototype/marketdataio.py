################################################################################
# Description: Read financial time series from csv files
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        gio 05 mag 2016 18:06:22 CEST
################################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def symbol_to_path(symbol, base_dir='data'):
    """ Return CSV file path given ticker symbol.

    :symbol: ticker symbol of target asset
    :base_dir: path to input data directory
    :returns: path to CSV file for target ticker symbol

    """
    return os.path.join(base_dir, '{}.csv'.format(str(symbol)))


def read_stock_prices(reference_symbol, symbols, dates, base_dir='data'):
    """ Read stock data (adjusted close) for given symbols from CSV files

    :reference_symbol: reference ticker symbol, e.g. SPY
    :symbols: list of ticker symbols to read
    :dates:   list of datetime objects
    :base_dir: path to input data directory
    :returns: pandas dataframe containing the target time series

    """
    # Create empty dataframe with selected dates as indices
    df = pd.DataFrame(index=dates)

    # Insert reference symbol in symbols list
    symbols.insert(0, reference_symbol)

    # Read all price series
    for symbol in symbols:

        # Read price series in temporary dataframe
        df_temp = pd.read_csv(symbol_to_path(symbol, base_dir),
                              index_col='Date', parse_dates=True,
                              usecols=['Date', 'Adj Close'],
                              na_values=['nan']).sort_index()

        # Rename adjusted close column to symbol
        df_temp = df_temp.rename(columns={'Adj Close': symbol})

        # Join dataframes
        df = df.join(df_temp, how='inner'
                     if (symbol == reference_symbol) else 'left')

    # Fill na (forward/backward)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df


def plot_stock_prices(df, dates, symbols, title='Stock Prices'):
    """ Plot stock prices contained in df

    :df: stock price DataFrame
    :title: plot title

    """
    ax = df.ix[dates, symbols].plot(title=title)
    ax.set_xlabel('Dates')
    ax.set_ylabel('Prices')
    plt.show()


def compute_daily_returns(df):
    """compute daily returns of stocks in the dataframe

    :df: dataframe containing the stocks prices
    :returns: dataframe containing the stocks daily returns

    """
    daily_returns = df / df.shift(1) - 1
    return daily_returns.ix[1:]


if __name__ == "__main__":
    # Directories
    input_data_dir = '../../Data/Input/'
    output_data_dir = '../../Data/Output/'

    # Target Dates
    start_date = '1990-01-01'
    end_date = '2015-03-20'
    dates = pd.date_range(start_date, end_date)

    # Target symbols
    reference_symbol = 'SPY'
    symbols = []

    # Portfolio allocations
    ptf_weights = np.array([0.4, 0.4, 0.1, 0.1])

    # Read prices
    df = read_stock_prices(reference_symbol, symbols, dates, input_data_dir)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)

    # Write daily returns to file
    daily_returns.to_csv(input_data_dir + 'daily_returns.csv')
