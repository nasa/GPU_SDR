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

    def __init__(self):
        self.trigger_control = "AUTO" # OR MANUAL
        if (self.trigger_control != "AUTO") and (self.trigger_control != "MANUAL"):
            err_msg = "Trigger_control in the trigger class can only have MANUAL or AUTO value, not \'%s\'"%str(self.trigger_control)
            print_error(err_msg)
            raise ValueError(err_msg)

    def dataset_init(self, antenna_group):
        '''
        This function is called on file creation an is used to create additional datasets inside the hdf5 file.
        In order to access the datasets created here in the trigger function make them member of the class:

        >>> self.custom_dataset = antenna_group.create_dataset("amazing_dataset", shape = (0,), dtype=np.dtype(np.int64), maxshape=(None,), chunks=True)

        Note: There is no need to bookeep when (at what index) the trigger is called as this is already taken care of in the trigger dataset.
        :param antenna_group is the antenna group containing the 'error','data' and triggering datasets.
        '''

        return

    def trigger(self, data, metadata):
        '''
        Triggering function.
        Make modification to the data and metadata accordingly and return them.

        :param data: the data packet from the GPU server
        :param metadata: the metadata packet from the GPU server

        :return same as argument but with modified content.

        Note: the order of data at this stage follows the example ch0_t0, ch1_t0, ch0_t1, ch1_t1, ch0_t2, ch1_t2...
        '''
        return data, metadata

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
