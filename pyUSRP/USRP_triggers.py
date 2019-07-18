########################################################################################
##                                                                                    ##
##  THIS LIBRARY IS PART OF THE SOFTWARE DEVELOPED BY THE JET PROPULSION LABORATORY   ##
##  IN THE CONTEXT OF THE GPU ACCELERATED FLEXIBLE RADIOFREQUENCY READOUT PROJECT     ##
##                                                                                    ##
########################################################################################
from .USRP_low_level import *
import numpy as np
from scipy import signal
class trigger_template(object):
    '''
    Example class for developing a trigger.
    The triggering method has to be passed as an argument of the Packets_to_file function and has to respect the directions given in the Trigger section of this documentation.
    The user is responible for the initialization of the object.
    The internal variable trigger_coltrol determines if the trigger dataset bookeep whenever the trigger method returns metadata['length']>0 or if it's controlled by the user.
    In case the trigger_control is not set on \'AUTO\' the user must take care of expanding the dataset before writing.
    '''

    def __init__(self, rate):
        self.trigger_control = "MANUAL" # OR MANUAL
        if (self.trigger_control != "AUTO") and (self.trigger_control != "MANUAL"):
            err_msg = "Trigger_control in the trigger class can only have MANUAL or AUTO value, not \'%s\'"%str(self.trigger_control)
            print_error(err_msg)
            raise ValueError(err_msg)

        #------------------------------
        self.stored_data = np.empty([0,])
        self.time_index = 0
        self.rate = rate

    def dataset_init(self, antenna_group):
        '''
        This function is called on file creation an is used to create additional datasets inside the hdf5 file.
        In order to access the datasets created here in the trigger function make them member of the class:

        >>> self.custom_dataset = antenna_group.create_dataset("amazing_dataset", shape = (0,), dtype=np.dtype(np.int64), maxshape=(None,), chunks=True)

        Note: There is no need to bookeep when (at what index) the trigger is called as this is already taken care of in the trigger dataset.
        :param antenna_group is the antenna group containing the 'error','data' and triggering datasets.
        '''

        self.trigger_group = antenna_group['trigger']

        return
    def write_trigger(self, data):

        current_len_trigger = len(self.trigger)
        self.trigger.resize(current_len_trigger+1,0)
        self.trigger[current_len_trigger] = data

    def trigger(self, data, metadata):
        '''
        Triggering function.
        Make modification to the data and metadata accordingly and return them.

        :param data: the data packet from the GPU server
        :param metadata: the metadata packet from the GPU server

        :return same as argument but with modified content.

        Note: the order of data at this stage follows the example ch0_t0, ch1_t0, ch0_t1, ch1_t1, ch0_t2, ch1_t2...
        '''

        n_chan = metadata['channels']
        self.time_index += metadata['length']/n_chan
        self.stored_data = np.concatenate((self.stored_data,data)) ##accumulating data
        if len(self.stored_data) >= 10*self.rate: ##if data is long enough
            n_samples = len(self.stored_data) / n_chan ##number of samples per channel
            reshaped_data = np.reshape(self.stored_data, (n_samples, n_chan)).T
            srate = self.rate
            hits = np.zeros(n_samples, dtype=bool) ##initially all false.
            for x in range(0, n_chan):
                current = reshaped_data[x]
                med = np.median(current)
                stddev = np.std(current)
                lo = med - 10*stddev
                hi = med + 10*stddev
                mask = np.logical_or(current<lo, current>hi)
                hits = np.logical_or(hits, mask)
            ##now hits has the indices of all the glitches across all the chs.
            hit_indices = np.nonzero(hits)[0]
            indices_diffs = np.ediff1d(hit_indices)
            count = 0
            for y in range(0, len(indices_diffs)):
                if indices_diffs[y] < (0.001*srate): ##if points are less than .001 sec apart
                    hit_indices = np.delete(hit_indices, count+1)
                else:
                    count += 1
            ##now hit_indices only contains one marker per glitch.
            if len(hit_indices) != 0: ##if this detects a glitch
                num = int(srate * 0.002) ##half of number of points saved. (0.004 sec range total saved)
                res = np.empty([n_chan, 0])
                total_time = np.arange(n_samples)/srate
                times = np.array([])
                for z in range(0, len(hit_indices)): ##find data around glitches
                    i = hit_indices[z]
                    chopped = reshaped_data[0:n_chan, (i-num):(i+num)]
                    res = np.concatenate((res.T, chopped.T)).T
                    times = np.concatenate((times, total_time[(i-num):(i+num)]))
                res = np.reshape(res.T, (res.size,))
                metadata['length'] = len(res)
                self.stored_data = np.array([])
                #self.write_trigger(self.time_index) this is a bug
                return res, metadata#, times ####added times output for testing purposes
            else: ##if no glitches detected
                self.stored_data = np.array([])
                metadata['length'] = 0
                return np.array([]), metadata
                ###return piece of timestream
        else: ##if data is not long enough.
            metadata['length'] = 0
            return np.array([]), metadata


class deriv_test(trigger_template):
    '''
    Just a test I wrote for the triggers.
    There is a bug somewhere an it's way too slow for a long acquisition to be sustainable.
    '''
    def __init__(self):
        trigger_template.__init__(self)
        self.stored_data = np.array([])
        self.threshold = 1.1

    def trigger(self, data, metadata):
        n_chan = metadata['channels']

        # Accumulate data
        self.stored_data = np.concatenate((self.stored_data,data))

        # Reach a condition
        if len(self.stored_data) >= 3 * metadata['length']:

            # do some analysis
            samples_per_channel = 3*metadata['length']/n_chan
            formatted_data = np.gradient( np.reshape(self.stored_data, (samples_per_channel, n_chan)).T, axis = 1)
            per_channel_average = np.abs(np.mean(formatted_data,1))
            x = sum([sum(np.abs(formatted_data[i])>(self.threshold*per_channel_average[i])) for i in range(len(formatted_data))])
            if x > 1:

                ret = self.stored_data
                metadata['length'] = len(self.stored_data)
                self.stored_data = np.array([])

                return ret, metadata
            else:
                self.stored_data = np.array([])
                metadata['length'] = 0
                return [[],],metadata
        else:
            metadata['length'] = 0
            return [[],],metadata
