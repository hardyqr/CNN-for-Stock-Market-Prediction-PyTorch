
import os
import sys
from tqdm import *

files = os.listdir(sys.argv[1])

iter_num = int(sys.argv[2])

counter = 0


for f in tqdm(files):
    if('.txt' not in f): continue
    if((len(sys.argv) == 3) and (counter > iter_num)): break
    os.system('python3 ./10-3-gen.py '+sys.argv[1]+'/'+f)
    print('generating: %s, iteration: %d') % (f, counter)
    counter = counter + 1

