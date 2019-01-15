import libUSRP2 as u
import numpy as np
import matplotlib.pyplot as pl
import os
import glob
import vispy
from scipy import signal
from joblib import Parallel, delayed
folder_name = raw_input("folder name? ")
os.chdir(folder_name)

list_of_files = glob.glob('USRP*.h5')
latest_file = [x.split(".")[0] for x in (sorted(list_of_files, key=os.path.getctime))]
ch_list = [0]
print "Opening " + str(latest_file)
'''
Results = Parallel(n_jobs=-1,verbose=1 )(
        delayed(u.calculate_noise)(
            filename, welch = 100, clip = 0.5,dbc = True, rotate = True
        ) for filename in latest_file
    )
'''
my_info = []
for filename in latest_file:
    info = len(u.get_rx_info(filename)['wave_type'])
    my_info.append("n_chan: %d"%info)
    u.calculate_noise(filename, welch = 10, clip = 0.3,dbc = False, rotate = False)

u.plot_raw_data(latest_file, channel_list=ch_list,mode='PM', output_filename = folder_name, end_time = 0.1, decimation = None,backend = 'matplotlib',size = (10,8), add_info = my_info)#
u.plot_noise_spec(latest_file, channel_list = ch_list, max_frequency = None, backend = 'matplotlib', cryostat_attenuation = 0, auto_open = True, output_filename = folder_name+"_spec", add_info = my_info, size = (10,8))

u.plot_raw_data(latest_file, channel_list=ch_list,mode='PM', output_filename = folder_name, end_time = 0.1, decimation = None,backend = 'plotly',size = (10,8), add_info = my_info)#
u.plot_noise_spec(latest_file, channel_list = ch_list, max_frequency = None, backend = 'plotly', cryostat_attenuation = 0, auto_open = True, output_filename = folder_name+"_spec", add_info = my_info, size = (10,8))



