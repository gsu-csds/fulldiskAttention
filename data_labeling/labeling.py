# Necessary Imports

from datetime import datetime as dt
from datetime import date,timedelta
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import os, sys, csv
# pd.set_option('display.max_columns', 50000)
# pd.set_option('display.width', 50000)

#Retreive all magnetograms that exists in our directory and save it as a csv
def check_files():
    source = '/data/hmi_jpgs_512/'
    with open('data_labels/totalfiles_jpg_512.csv','w',newline='',encoding='utf-8-sig') as f:
        w = csv.writer(f)
        for path, subdirs, files in os.walk(source):
            for name in files:
                w.writerow([os.path.join(path, name)])
# check_files()