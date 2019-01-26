import pyUSRP.libUSRP2 as u
import numpy as np
import matplotlib.pyplot as pl
import os
import glob
import vispy
from scipy import signal
from vispy.plot import Fig

list_of_files = glob.glob('USRP*.h5')
latest_file = (max(list_of_files, key=os.path.getctime)).split(".")[0]
print "Opening " + latest_file

info = u.get_rx_info(latest_file)
if info['wave_type'][0] == 'NOISE':
    u.plot_all_pfb(latest_file, decimation = None, low_pass = None, backend = 'matplotlib', size = (12,8), output_filename = "last_file", start_time = None, end_time = None, auto_open = True)
else:
    u.plot_raw_data(latest_file, channel_list=None,mode='PM',size = (12,8), output_filename = "last_file", decimation = None,backend = 'matplotlib',end_time = 1)# 




