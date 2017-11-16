
import os
import sys
from tqdm import *

files = os.listdir(sys.argv[1])
for f in tqdm(files):
    if('.txt' not in f): continue
    os.system('python3 ./10-3-gen.py '+sys.argv[1]+ f)
