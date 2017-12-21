# Freddy @Blair House, Waterloo, ON, Canada
# Dec 20, 2017

import numpy as np
import pandas as pd
import csv
import time
from datetime import datetime
import sys
import os
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.finance import candlestick_ohlc
import warnings
warnings.filterwarnings("ignore")
from data_gen import data_plot
from utils import *



def main():
    data = pd.read_csv(sys.argv[1], parse_dates=True, dayfirst=True)# argv[1]: stock_symbol.txt
    # some preprocessing
    data = data[['Open','High','Low','Close','Volume']].astype('float32')     
    
    data_len = len(data)
    counter = 0
    
    for k in range(0,int(data_len/23)+1):
        if(counter + 23 >= data_len): break
        data_cur = data[counter:counter+23]
        data_close = data_cur['Close']
        labels = []
        for i in range(0,23):
            # draw day [0,1,...,19], predict day [20,21,22] 
            # positive label 1, negative label 0
            last_4 = data_close[19:23].values
            labels_raw = last_4[1:4] - last_4[0]
            labels = list(map(one_hot2,labels_raw))
        counter = counter + 23
        
        # save data
        
        data_cur = data_cur[0:20].transpose() # only save the first 20 days
        
        output_dir = sys.argv[2] # root dir of the output data

        mode = 'train' # 70% training set
        if(k%10 == 0): mode = 'validation' # 10% validation set
        elif(k%10 == 4 or k%10 == 7): mode = 'test' # 20% testing set
        
        matrix_file_name = output_dir + '/matrix_'+mode+'.csv'
        
        new_label = [[labels[0],labels[1],labels[2]]]
        df = pd.DataFrame(new_label, columns=['pred_1', 'pred_2', 'pred_3'])
        label_file_name = output_dir+'/label_'+mode+'.csv'

        #write
        if(df.shape[0]==1 and df.shape[1]==3 and data_cur.shape[0]==5 and data_cur.shape[1]==20):
            df.to_csv(label_file_name, mode='a', header=False)
            data_cur.to_csv(matrix_file_name, mode='a', header=False)
        else:
            print('data shape error:')
            print(data_cur.shape)
            print(df.shape)

if __name__ == '__main__':
    main()


