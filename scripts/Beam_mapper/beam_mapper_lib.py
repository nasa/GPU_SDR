import numpy as np
import sys,os,csv
import time
import h5py
from scipy import signal
###############################################################
# FUNCTIONS COPIED FROM PYUSRP TO MAKE THIS MODULE STANDALONE #
###############################################################
def bound_open(filename):
    '''
    Return pointer to file. It's user responsability to call the close() method.
    '''
    try:
        filename = format_filename(filename)
        f = h5py.File(filename,'r')
    except IOError as msg:
        print_error("Cannot open the specified file: "+str(msg))
        f = None
    return f

def print_warning(message):
    '''
    Print a yellow warning label before message.
    :param message: the warning message.
    :return: None
    '''
    print "\033[40;33mWARNING\033[0m: " + str(message) + "."


def print_error(message):
    '''
    Print a red error label before message.
    :param message: the error message.
    :return: None
    '''
    print "\033[1;31mERROR\033[0m: " + str(message) + "."


def print_debug(message):
    '''
    Print the message in italic grey.
    :param message: the debug message.
    :return: None
    '''
    print "\033[3;2;37m" + str(message) + "\033[0m"

def format_filename(filename):
    return os.path.splitext(filename)[0]+".h5"

def find_nearest(array, value):
    '''
    Utility function to find the nearest value in an array.
    :param array: array of numbers.
    :param value: value to find.
    :return: closest index.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)


###################################################
#           END OF COPIED FUNCTIONS               #
###################################################

def extract_frequecy(data, fs, freq, half_span, welch = None):
    '''
    Perform fft and extract frequency db amplitude.
    :return db amplitude value.
    '''
    L = len(data)
    if welch == None:
        welch = L
    else:
        welch = int(L / welch)

    Frequencies, ampl = signal.welch(data, nperseg=welch, fs=fs, detrend='linear', scaling='density')
    ampl = 10 * np.log10(ampl)
    idx = find_nearest(Frequencies, freq)
    idx_up = find_nearest(Frequencies, freq+half_span)
    idx_down = find_nearest(Frequencies, freq-half_span)

    if idx_up == idx or idx_down == idx:
        half_span = Frequencies[3] - Frequencies[2]
        idx_up = find_nearest(Frequencies, freq+half_span)
        idx_down = find_nearest(Frequencies, freq-half_span)
        print_warning("Frequency half_span too low. extending to: %.1f Hz" % span)

    return np.max(ampl[idx_down:idx_up])

def build_time_axis(filename, front_end = 'A_RX2', verbose = True):
    filename = format_filename(filename)
    if not check_beam_embedded(filename):
        err_msg = "Cannot check time axis: beam data not embedded. consider running the function embed_beam_data()"
        print_error(err_msg)
        raise ValueError(err_msg)
    beam_data = get_beam_data(filename)
    beam_data_start = beam_data['ti'][0]
    beam_data_end = max(beam_data['tf'])
    f = bound_open(filename)
    data_t_start = f['raw_data0'][front_end]['data'].attrs.get("start_epoch")
    data_rate = f['raw_data0'][front_end].attrs.get("rate")/f['raw_data0'][front_end].attrs.get("fft_tones")
    if f['raw_data0'][front_end].attrs.get("decim")!=0:
        data_rate /= f['raw_data0'][front_end].attrs.get("decim")

    if verbose:
        if data_rate < 1e6:
            print_debug("Effective data rate: %.2f ksps"%(data_rate/1e3))
        else:
            print_debug("Effective data rate: %.1f Msps"%(data_rate/1e6))

    data_len = len(f['raw_data0'][front_end]['data'][0])
    data_t_end = data_len*data_rate + data_t_start
    print data_len*data_rate
    print beam_data_end-beam_data_start

    f.close()
    delta_init = beam_data_start - data_t_start
    delta_end = data_t_end - beam_data_end

    if verbose:
        print_debug("Resonators data start %.1f seconds before the beam movement"%delta_init)
        print_debug("Beam movement ends %.1f seconds before the end of acquisition"%delta_end)

    return



def check_beam_embedded(filename):
    '''
    Check if the h5 file containing noise data has the beam data embedded as group.
    :return boolean resulting
    '''
    f = bound_open(filename)
    try:
        data = f['beam_data']
        ret = True
    except ValueError:
        ret = False
    except KeyError:
        ret = False

    f.close()
    return ret

def read_beam_csv(csv_filename):
    '''
    Read the csv containing the beam data and return a dictionary with the data.
    :return dictionary containing the information of each bem step with keys:
        - ti: initial step time [s]
        - tf: final step time [s]
        - x: x position [inch]
        - y: y position [inch]
    '''
    ti = []
    tf = []
    x = []
    y = []
    col_name = True
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_number = 0
        for row in csv_reader:
            line_number +=1
            if col_name:
                print_debug('Column names are %s'%{", ".join(row)})
                col_name = False
            else:
                try:
                    ti.append(float(row[0]))
                    tf.append(float(row[1]))
                    x.append(float(row[2]))
                    y.append(float(row[3]))
                except ValueError:
                    err_msg = "Invalid value at line %d"%line_number
                    print_error(err_msg)
                    raise ValueError(err_msg)
                except IndexError:
                    err_msg = "Missing value at line %d"%line_number
                    print_error(err_msg)
                    raise ValueError(err_msg)

    return {'ti':ti,'tf':tf,'x':x,'y':y}

def get_beam_data(filename):
    '''
    Read the beam map data stored in a h5 noise file.
    :return dictionaty containing the data.with keys:
        - ti: initial step time [s]
        - tf: final step time [s]
        - x: x position [inch]
        - y: y position [inch]
    '''
    filename = format_filename(filename)
    if not check_beam_embedded(filename):
        err_msg = "Beam map data not embedded in the noise file"
        print_error(err_msg)
        raise ValueError(err_msg)

    f = bound_open(filename)
    grp = f['beam_data']
    ret = {
        'ti': np.asarray(grp['t_init']),
        'tf': np.asarray(grp['t_final']),
        'x': np.asarray(grp['x_pos']),
        'y': np.asarray(grp['y_pos'])
    }
    f.close()
    return ret

def embed_beam_data(csv_filename, noise_filename, verbose = True):
    '''
    Embed beam map data in the noise file.
    '''
    if verbose: print("Embedding beam map data from file \'%s\' in the h5 noise file \'%s\'"
        % (csv_filename,noise_filename))
    noise_filename = format_filename(noise_filename)
    data = read_beam_csv(csv_filename)
    fv = h5py.File(noise_filename,'r+')
    try:
        beam_group = fv.create_group("beam_data")
    except ValueError:
        del fv["beam_data"]
        beam_group = fv.create_group("beam_data")
        if verbose: print_warning("Overwriting beam data group")

    beam_group.create_dataset("t_init", data = data['ti'])
    beam_group["t_init"].attrs.create(name="unit", data="s epoch")
    beam_group.create_dataset("t_final", data = data['tf'])
    beam_group["t_final"].attrs.create(name="unit", data="s epoch")
    beam_group.create_dataset("x_pos", data = data['x'])
    beam_group["x_pos"].attrs.create(name="unit", data="inch")
    beam_group.create_dataset("y_pos", data = data['y'])
    beam_group["y_pos"].attrs.create(name="unit", data="inch")

    fv.close()
    if verbose: print_debug("Embedding complete")

    return
