import sys
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print "Cannot find the pyUSRP package"
import numpy as np
import os
import glob
import argparse


'''
Results = Parallel(n_jobs=-1,verbose=1 )(
        delayed(u.calculate_noise)(
            filename, welch = 100, clip = 0.5,dbc = True, rotate = True
        ) for filename in latest_file
    )
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot the latest file')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")

    args = parser.parse_args()

    os.chdir(args.folder)

    list_of_files = glob.glob('USRP*.h5')
    latest_file = [x.split(".")[0] for x in (sorted(list_of_files, key=os.path.getctime))][0]
    ch_list = [0]
    print "Opening " + str(latest_file)

    u.plot_raw_data(latest_file, channel_list=ch_list,mode='IQ', output_filename = latest_file[0], end_time = 0.1, decimation = None,backend = 'matplotlib',size = (10,8))#
    u.plot_raw_data(latest_file, channel_list=ch_list,mode='IQ', output_filename = latest_file[0], end_time = 0.1, decimation = None,backend = 'plotly',size = (10,8))
