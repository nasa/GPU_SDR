import numpy as np
import sys,os,csv
import time
import h5py
from scipy import signal
from joblib import Parallel,delayed
import matplotlib.pyplot as pl
from scipy import optimize
from atpbar import atpbar
from matplotlib.lines import Line2D

###############################################################
# FUNCTIONS PORTED FROM PYUSRP TO MAKE THIS MODULE STANDALONE #
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

def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x,y):
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        g = height*np.exp(
            -(((center_x-xp)/width_x)**2+
              ((center_y-yp)/width_y)**2)/2.)
        return g
    return rotgauss

def moments(data):
    '''
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments. modified from scipy-cookbook to match our data
    '''

    (x,y) = np.unravel_index(np.argmax(data, axis=None), data.shape)
    width_x = len(data)/20
    width_y = len(data[0])/15
    height = np.max(data)
    return height, x, y, width_x, width_y, 0

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    '''
    bounds = (
        np.asarray([0.9*params[0], 0.8*params[1], 0.8*params[2], 0.5*params[3], 0.5*params[4], -np.pi]),
        np.asarray([1.1*params[0], 1.2*params[1], 1.2*params[2], 1.5*params[3], 1.5*params[4], +np.pi])
    )
    '''
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p = optimize.least_squares(errorfunction, x0 = params)

    return p['x']


###################################################
#           END OF PORTED FUNCTIONS               #
###################################################

def build_time_axis(filename, front_end = 'A_RX2', verbose = True):
    '''
    Return the time axis in the epoch coord for the resonator data. Also check timing consistency of beam map data and resonator data.
    '''
    filename = format_filename(filename)
    if not check_beam_embedded(filename):
        err_msg = "Cannot check time axis: beam data not embedded. consider running the function embed_beam_data()"
        print_error(err_msg)
        raise ValueError(err_msg)
    beam_data = get_beam_data(filename)
    beam_data_start = beam_data['ti'][0]
    beam_data_end = max(beam_data['tf'])
    print np.argmax(beam_data['tf'])
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

    data_len = int(f['raw_data0'][front_end]['data'].attrs.get("samples"))
    data_t_end = data_len/data_rate + data_t_start


    f.close()
    delta_init = beam_data_start - data_t_start
    delta_end = data_t_end - beam_data_end

    if verbose:
        if delta_init > 0:
            print_debug("Resonators data start %.1f seconds before the beam movement"%delta_init)
        else:
            err_msg ="Resonator data acquisition starts after beam movement"
            print_error(err_msg)
            raise ValueError(err_msg)
        if delta_end > 0:
            print_debug("Beam movement ends %.1f seconds before the end of acquisition"%delta_end)
        else:
            err_msg = "Resonator data acquisition ends before beam movement by %.2f seconds"%delta_end
            print_error(err_msg)
            raise ValueError(err_msg)

    return np.asarray([data_t_start + x*(1./data_rate) for x in range(data_len)])


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


def extract_frequecy(data, step,time_axis,beam_data, fs, freq, half_span, welch = None, delay = 0):
    '''
    Perform fft and extract frequency db amplitude.
    :return db amplitude value.
    '''
    delay/=1000.
    start = find_nearest(time_axis,beam_data['ti'][step]-delay)
    end = find_nearest(time_axis,beam_data['tf'][step]-delay)
    #print "delats: init: %.3f end %.3f"%(time_axis[start] - (beam_data['ti'][step]-delay) , time_axis[end] - (beam_data['tf'][step]-delay))
    #data = data_[start:end]
    L = len(data[start:end])
    if welch == None:
        welch = L
    else:
        welch = int(L / welch)

    Frequencies, ampl = signal.welch(np.abs(data[start:end]), nperseg=welch, fs=fs, detrend='linear', scaling='density')
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


def build_map(filename, freq, half_span, front_end = 'A_RX2', verbose = True, welch = None):
    '''
    Calculate the beam map and write it in the file containing the data.
    '''
    if verbose: print "Calculating beam map..."
    time_axis = build_time_axis(filename, front_end, verbose)
    beam_data = get_beam_data(filename)
    nx = len(np.unique(beam_data['x']))
    ny = len(np.unique(beam_data['y']))
    if verbose: print_debug("Detected motor steps: %d x %d"%(nx,ny))
    # get the pointer to the data: should work in low ram configurations
    filename = format_filename(filename)
    fp = bound_open(filename)
    data = fp['raw_data0'][front_end]['data'][:]

    tone_freq = np.asarray(fp['raw_data0'][front_end].attrs.get("freq")) + fp['raw_data0'][front_end].attrs.get("rf")
    n_chan = len(fp['raw_data0'][front_end]['data'])
    n_step = len(beam_data['ti'])
    data_rate = fp['raw_data0'][front_end].attrs.get("rate")/fp['raw_data0'][front_end].attrs.get("fft_tones")
    if fp['raw_data0'][front_end].attrs.get("decim")!=0:
        data_rate /= fp['raw_data0'][front_end].attrs.get("decim")


    def aligned(a, alignment=32):
        if (a.ctypes.data % alignment) == 0:
            return a

        extra = alignment / a.itemsize
        buf = np.empty(a.size + extra, dtype=a.dtype)
        ofs = (-buf.ctypes.data % alignment) / a.itemsize
        aa = buf[ofs:ofs+a.size].reshape(a.shape)
        np.copyto(aa, a)
        assert (aa.ctypes.data % alignment) == 0
        return aa

    data = np.asarray([aligned(ch) for ch in data])

    if verbose: print_debug("Calculating ffts...")
    beam_map = Parallel(n_jobs=24, verbose=False, require='sharedmem')(
        delayed(extract_frequecy)(
    #beam_map = [extract_frequecy(
            data = data[ch],
            time_axis = time_axis,
            beam_data = beam_data,
            step = step,
            fs = data_rate,
            freq = freq,
            half_span = half_span,
            welch = welch,
            delay = 0
        )  for ch in atpbar(range(n_chan), name='channels') for step in atpbar(range(n_step), name='motor steps')
    )
    #]
    fp.close()
    if verbose: print_debug("Writing beam map to file...")
    fv = h5py.File(filename,'r+')
    try:
        beam_group = fv.create_group("beam_map")
    except ValueError:
        del fv["beam_map"]
        beam_group = fv.create_group("beam_map")
        if verbose: print_warning("Overwriting beam map group")

    beam_group.attrs.create(name="freq", data=freq)
    beam_group.attrs.create(name="half_span", data=half_span)
    beam_group.attrs.create(name="n_chan", data=n_chan)
    beam_group.attrs.create(name="nx", data=nx)
    beam_group.attrs.create(name="ny", data=ny)
    beam_group.attrs.create(name="tone_freq", data=tone_freq)

    for ch in range(n_chan):
        channel_data = beam_map[int(ch*(nx*ny)):int((ch+1)*(nx*ny))]
        w = zip(channel_data, beam_data['x'],beam_data['y'])
        w.sort(key = lambda l: (l[1],l[2]), reverse=True)
        channel_data = zip(*w)[0]
        beam_group.create_dataset("channel_%d"%ch, data = channel_data)
    fv.close()

def get_full_beam_map_data(filename):
    '''
    Returen beam map data from a h5 file. The file has to be previoulsy analyzed with the function build_map().
    '''
    beam_data = get_beam_data(filename)
    filename = format_filename(filename)
    fp = bound_open(filename)
    try:
        g = fp['beam_map']
    except ValueError:
        err_msg = "Cannot find beam_map group. Has this file been analyzed?"
        print_error(err_msg)
        raise ValueError(err_msg)
    except KeyError:
        err_msg = "Cannot find beam_map group. Has this file been analyzed?"
        print_error(err_msg)
        raise ValueError(err_msg)

    ret = {
        'n_chan':g.attrs.get('n_chan'),
        'nx':g.attrs.get('nx'),
        'ny':g.attrs.get('ny'),
        'x':beam_data['x'],
        'y':beam_data['y'],
        'data': [g['channel_%d'%i][:] for i in range(g.attrs.get('n_chan'))],
        'freq':g.attrs.get('freq'),
        'half_span':g.attrs.get('half_span'),
        'tone_freq':g.attrs.get('tone_freq')
    }

    fp.close()

    return ret

def plot_beam_map(filename, cmap = 'Greys', levels = None):
    '''
    Plot the beam map stored on the h5 file.
    '''
    print "Plotting beam map data..."

    beam_data = get_full_beam_map_data(filename)

    try:
        os.mkdir("figures")
    except OSError:
        pass

    os.chdir("figures")


    #reconstruct axis
    xvec = np.linspace(min(beam_data['x']),max(beam_data['x']),beam_data['nx'])
    yvec = np.linspace(min(beam_data['y']),max(beam_data['y']),beam_data['ny'])
    step_x = xvec[1]-xvec[0]
    step_y = yvec[1]-yvec[0]
    sx = 14
    sy = sx*float(step_y)/float(step_x)
    print_debug("aspect ratio: %.2f"%(float(step_y)/float(step_x)))
    X,Y=np.meshgrid(yvec,xvec)

    if levels == None:
        levels = [-110,-90,-85,]

    linewidths = np.linspace(0.5,1.5,len(levels))
    centers = []
    for i in atpbar(range(beam_data['n_chan']), name='channels plot'):
        fig, ax = pl.subplots(figsize=(sx,sy))

        Z = np.reshape(beam_data['data'][i],(beam_data['nx'],beam_data['ny']))
        g = pl.pcolormesh(X,Y,Z, cmap = cmap, alpha = 0.7)
        #fitting
        params = fitgaussian(10**(Z/20))
        fit = gaussian(*params)
        fit_data = fit(*np.indices(Z.shape))
        #rint params
        fit_level = [np.max(fit_data)*0.64]
        pl.contour(X, Y, fit_data, colors='red', levels = fit_level)
        #"gaussian fit:\nCenter: %.2f %.2f"%(params[0],params[1])
        (y_max,x_max) = np.unravel_index(np.argmax(fit_data, axis=None), fit_data.shape)
        x_max = xvec[min(x_max,len(xvec)-1)]
        y_max = yvec[min(y_max,len(yvec)-1)]
        #x_max = params[1]
        #y_max = params[2]
        centers.append((x_max,y_max))
        
        fit_label = "2D gaussian fit 64%"
        fit_label+="\nCenter X: %.2f Y: %.2f [inch]"%(x_max,y_max)
        fit_label+="\nAsymmetry $|1-x/y|$: %.2f"%(np.abs(1-params[3]/params[4]))
        fit_label+="\nRotation: $%.1f^o$"%params[5]
        colors = ['black', 'red']
        lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in colors]
        labels = ['Beam map level', fit_label]
        pl.legend(lines, labels)

        contours = pl.contour(X, Y, Z, colors='black',
            levels = levels, linestyles = 'solid', antialiased = True, linewidths = linewidths)
        pl.clabel(contours, inline=False, fontsize=10)

        cbar = pl.colorbar(g)
        cbar.set_label( "$%d\pm%.1f Hz$ line magnitude [dBm]"%(beam_data['freq'],beam_data['half_span']), rotation=270, labelpad=30)


        pl.scatter(
            #xvec[int(params[1])] * np.cos(np.radians(params[5])),
            #yvec[int(params[2])] * np.sin(np.radians(params[5])),
            [x_max],
            [y_max],
            color = 'red'
        )

        pl.xlabel('X position [Inches]')
        pl.ylabel('Y posiiton [Inches]')
        pl.title("Beam map: channel %.2fMHz"%(beam_data['tone_freq'][i]/1e6))
        pl.savefig("channel%d.png"%(i))

        pl.close(fig)
    print_debug("Printing resonators map...")
    centers = zip(*centers)
    fig, ax = pl.subplots(figsize=(20,20))
    #pl.scatter(centers[0], centers[1], marker = '+')
    pl.title("Resonators position")
    for i in range(len(centers[0])):
        ax.text(centers[0][i], centers[1][i], "%d"%i)
        #ax.annotate("%.2fMHz"%(beam_data['tone_freq'][i]/1e6),(centers[0][i], centers[1][i]))
        #ax.annotate("%d"%i,(centers[0][i], centers[1][i]))
        pl.scatter([min(xvec),max(xvec)],[min(yvec),max(yvec)],alpha=0,label =  "%d, %.2fMHz"%(i,beam_data['tone_freq'][i]/1e6))
    pl.xlabel('X position [Inches]')
    pl.ylabel('Y posiiton [Inches]')
    pl.legend(ncol = 4,bbox_to_anchor=(1.04,1), loc="upper left")
    pl.grid()
    pl.savefig("channel_map.png",bbox_inches="tight")

    pl.close(fig)
    os.chdir('..')
