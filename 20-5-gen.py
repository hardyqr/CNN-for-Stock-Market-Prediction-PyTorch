# Freddy @Blair House, uWaterloo, ON, Canada
# Nov 13, 2017

import numpy as np
import pandas as pd
import csv
import time
from datetime import datetime
import sys
import os
from data_gen import get_data, data_plot



def main():
    data = get_data(sys.argv[1], sys.argv[2], sys.argv[3])
    #print(data.head())
    fig = data_plot(data)
    fig.set_size_inches(20,20);
    #fig.show()
    #os.system('sleep 3s')
    fig.savefig('./data/imgs/' + sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3], dpi=200)


if __name__ == '__main__':
    main()


