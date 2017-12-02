# Freddy @DC, uWaterloo, ON, Canada
# Nov 12, 2017

import numpy as np
import pandas as pd
import csv
import time
from datetime import datetime
import sys
import os
# plot
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.finance import candlestick_ohlc

# yahoo data
#from pandas_datareader import data as pdr
#import fix_yahoo_finance as yf


#yf.pdr_override() # <== that's all it takes :-)


'''
def get_data(abbv, start, end):
    data = pdr.get_data_yahoo(abbv, start=start, end=end)
    #data['Date'] = data.index
    data.reset_index(inplace=True,drop=False)
    df = data[['Date', 'Open', 'High', 'Low', 'Close']]
    df["Date"] = df["Date"].apply(mdates.datestr2num)
    return df
'''

def data_plot(data):
    fig, ax = plt.subplots()
    #plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4, axisbg='#07000d')
    candlestick_ohlc(ax, data.values, width=.6, colorup='#53c156', colordown='#ff1717')
    #ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    fig.autofmt_xdate()
    plt.xticks(rotation=45)
    plt.ylabel('Stock Price')
    plt.xlabel('Date')
    fig.set_size_inches(8,8);
    #plt.close()
    return fig

def main():
    data = get_data(sys.argv[1], sys.argv[2], sys.argv[3])
    #print(data.head())
    fig = data_plot(data)
    #fig.show()
    #os.system('sleep 3s')
    fig.savefig('./data/imgs/' + sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3], dpi=200)


if __name__ == '__main__':
    main()


