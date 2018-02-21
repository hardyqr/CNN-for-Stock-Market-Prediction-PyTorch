
import os
import sys
from tqdm import *


# commandformat
# python3 main-gen.py -m ../Desktop/Stocks 10 ~/Desktop/stock_matrix_data


flags = sys.argv[1]

files = os.listdir(sys.argv[2])

iter_num = int(sys.argv[3])




def ten_three_img_gen():
    counter = 0
    for f in tqdm(files):
        if('.txt' not in f): continue
        if((len(sys.argv) == 3) and (counter > iter_num)): break
        os.system('python3 ./10-3-gen.py '+sys.argv[1]+'/'+f)
        print('generating: %s, iteration: %d' % (f, counter))
        counter = counter + 1

def twenty_three_matrix_gen():
    counter = 0
    for f in tqdm(files):
        if('.txt' not in f): continue
        if(counter > iter_num): break
        os.system('python3 ./matrix_gen.py '+sys.argv[2]+'/'+f+' '+sys.argv[4]) 
        # argv3 output root dir
        print('generating: %s, iteration: %d' % (f, counter))
        counter = counter + 1


if('-i' in flags):
    ten_three_img_gen()
elif('-m' in flags):
    twenty_three_matrix_gen()
