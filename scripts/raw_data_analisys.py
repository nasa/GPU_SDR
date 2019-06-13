import sys,os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
import scipy.signal as signal
import glob
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print "Cannot find the pyUSRP package"

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Put some tones in output and store unprocessed data')

    parser.add_argument('--folder', '-F', help='Name of the folder in which the data have been stored. Move to this folder before everything', type=str, default = "data")
    parser.add_argument('--file', '-f', help='Name of the file containing the data. Default is last saved', type=str)
    parser.add_argument('--samples', '-s', help='Number of samples to plot ion the timestream', type=int, default = 1e4)
    parser.add_argument('--decimation', '-d', help='Decimation factor applied to data', type=int, default = None)
    args = parser.parse_args()
    decimation = args.decimation
    os.chdir(args.folder)
    snap_len = args.samples
    if args.file is None:
        list_of_files = glob.glob('USRP_Noise*.h5')
        filename = max(list_of_files, key=os.path.getctime)
    else:
        filename = args.file

    Z = u.openH5file(filename)[0]
    info = u.get_tx_info(filename, ant=None)
    #get a chunk of data to avoid capacitor charging
    Z = Z[int(1e6):int(3e6)]

    #plot a psd
    print "Reportd rate is %d Msps"%(info['rate']/1e6)
    psd_img, ax = pl.subplots( nrows=1, ncols=1 )
    psd_img.set_size_inches(18.5, 10.5)
    ax.psd(Z, Fs = info['rate'], NFFT = int(1e6), detrend = 'linear', scale_by_freq = True)
    psd_img.savefig("raw_data_psd.png")

    tone_list = info['freq']

    #analyze the signal
    tones_DC = []
    ii =0
    for tone in tone_list:
        demodulation_signal = np.conj(np.exp(1.j*(np.pi*2.*tone/info['rate']*np.arange(len(Z) + np.pi*0.25*tone/info['rate']))))
        #psd_img, ax = pl.subplots( nrows=1, ncols=1 )
        #psd_img.set_size_inches(18.5, 10.5)
        #ax.psd(demodulation_signal, Fs = info['rate'], NFFT = int(1e6), detrend = 'linear', scale_by_freq = True)
        #psd_img.savefig("signal_dem_psd.png")

        #this is to patch a wierd error
        l = min(len(demodulation_signal),len(Z)) -1
        res = demodulation_signal[:l]*Z[:l]
        if decimation is not None:
            res = signal.decimate(res, decimation, ftype='fir')[100:-100]
            snap_len -= 200
            rate = info['rate']/decimation
        else:
            rate = info['rate']

        tones_DC.append(res)

        psd_img, ax = pl.subplots( nrows=1, ncols=1 )
        psd_img.set_size_inches(18.5, 10.5)
        psd_img.suptitle("Channel %.2fMHz"%(tone/1e6))
        ax.psd(res, Fs = rate, NFFT = int(1e6), detrend = 'none', scale_by_freq = True)
        psd_img.savefig("signal_DC_psd.png")

        psd_img, ax = pl.subplots( nrows=1, ncols=1 )
        psd_img.set_size_inches(18.5, 10.5)
        time_axis = np.arange(int(snap_len))*1e6/rate
        ax.plot(time_axis,np.abs(res[:int(snap_len)]),label = "abs")
        psd_img.suptitle("Channel %.2fMHz"%(tone/1e6))
        pl.xlabel('Time $\mu s$')
        pl.ylabel('ADCu $\pm 1$')
        #ax.plot(np.angle(res[:int(1e5)]),label = "phi")
        pl.legend()
        psd_img.savefig("signal_timestream_%d.png"%ii)
        ii+=1
