########################################################################################
##                                                                                    ##
##  THIS LIBRARY IS PART OF THE SOFTWARE DEVELOPED BY THE JET PROPULSION LABORATORY   ##
##  IN THE CONTEXT OF THE GPU ACCELERATED FLEXIBLE RADIOFREQUENCY READOUT PROJECT     ##
##                                                                                    ##
########################################################################################

import numpy as np
import scipy.signal as signal
import signal as Signal
import h5py
import sys
import struct
import json
import os
import socket
import Queue
from Queue import Empty
from threading import Thread, Condition
import multiprocessing
from joblib import Parallel, delayed
from subprocess import call
import time
import gc
import datetime

# plotly stuff
from plotly.graph_objs import Scatter, Layout
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import colorlover as cl

# matplotlib stuff
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches

# needed to print the data acquisition process
import progressbar

# import submodules
from USRP_low_level import *
from USRP_files import *


def reinit_data_socket():
    '''
    Reinitialize the data network socket.
    :return: None
    '''
    global USRP_data_socket
    USRP_data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    USRP_data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def reinit_async_socket():
    '''
    Reinitialize the command network socket.
    :return: None
    '''
    global USRP_socket
    USRP_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    USRP_socket.settimeout(1)
    USRP_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def clean_data_queue(USRP_data_queue=USRP_data_queue):
    '''
    Clean the USRP_data_queue from residual elements. returns the number of element found in the queue.

    Returns:
        - Integer number of packets removed from the queue.
    '''
    print_debug("Cleaning data queue... ")
    residual_packets = 0
    while (True):
        try:
            meta_data, data = USRP_data_queue.get(timeout=0.1)
            residual_packets += 1
        except Empty:
            break
    print_debug("Queue cleaned of " + str(residual_packets) + " packets.")
    return residual_packets


def Packets_to_file(parameters, timeout=None, filename=None, dpc_expected=None, push_queue = None, trigger = None, **kwargs):
    '''
    Consume the USRP_data_queue and writes an H5 file on disk.

    :param parameters: global_parameter object containing the informations used to drive the GPU server.
    :param timeout: time after which the function stops and tries to stop the server.
    :param filename: eventual filename. Default is datetime.
    :param dpc_expected: number of sample per channel expected. if given display a percentage progressbar.
    :param push_queue: external queue where to push data and metadata
    :param trigger: trigger class (see section on trigger function for deteails)

    :return filename or empty string if something went wrong

    Note:
        - if the \"End of measurement\" async signal is received from the GPU server the timeout mode becomes active.
    '''

    global dynamic_alloc_warning
    push_queue_warning = False

    def write_ext_H5_packet(metadata, data, h5fp, index, trigger = None):
        '''
        Write a single packet inside an already opened and formatted H5 file as an ordered dataset.

        Arguments:
            - metadata: the metadata describing the packet directly coming from the GPU sevrer.
            - data: the data to be written inside the dataset.
            - dataset: file pointer to the h5 file. extensible dataset has to be already created.
            - index: dictionary containg the accumulated length of the dataset.
            - trigger: trigger class. (see trigger section for more info)

        Notes:
            - The way this function write the packets inside the h5 file is strictly related to the metadata type in decribed in USRP_server_setting.hpp as RX_wrapper struct.
        '''

        global dynamic_alloc_warning
        dev_name = "raw_data" + str(int(metadata['usrp_number']))
        group_name = metadata['front_end_code']
        samples_per_channel = metadata['length'] / metadata['channels']
        dataset = h5fp[dev_name][group_name]["data"]
        errors = h5fp[dev_name][group_name]["errors"]
        data_shape = np.shape(dataset)
        data_start = index
        data_end = data_start + samples_per_channel
        if ((trigger is not None) and (metadata['length']>0)):
            if trigger.trigger_control == "AUTO":
                trigger_dataset = h5fp[dev_name][group_name]["trigger"]
                current_len_trigger = len(trigger_dataset)
                trigger_dataset.resize(current_len_trigger+1,0)
                trigger_dataset[current_len_trigger] = index
        try:
            if data_shape[0] < metadata['channels']:
                print_warning("Main dataset in H5 file not initialized.")
                dataset.resize(metadata['channels'], 0)

            if data_end > data_shape[1]:
                if dynamic_alloc_warning:
                    print_warning("Main dataset in H5 file not correctly sized. Dynamically extending dataset...")
                    # print_debug("File writing thread is dynamically extending datasets.")
                    dynamic_alloc_warning = False
                dataset.resize(data_end, 1)
            packet = np.reshape(data, (samples_per_channel,metadata['channels'])).T
            dataset[:, data_start:data_end] = packet
            dataset.attrs.__setitem__("samples", data_end)
            if data_start == 0:
                dataset.attrs.__setitem__("start_epoch", time.time())

            if metadata['errors'] != 0:
                print_warning("The server encounterd an error")
                err_shape = np.shape(errors)
                err_len = err_shape[1]
                if err_shape[0] == 0:
                    errors.resize(2, 0)
                errors.resize(err_len + 1, 1)
                errors[:, err_len] = [data_start, data_end]
        except RuntimeError as err:
            print_error("A packet has not been written because of a problem: " + str(err))

    def write_single_H5_packet(metadata, data, h5fp):
        '''
        Write a single packet inside an already opened and formatted H5 file as an ordered dataset.

        Arguments:
            - metadata: the metadata describing the packet directly coming from the GPU sevrer.
            - data: the data to be written inside the dataset.
            - h5fp: already opened, with wite permission and group created h5 file pointer.

        Returns:
            - Nothing

        Notes:
            - The way this function write the packets inside the h5 file is strictly related to the metadata type in decribed in USRP_server_setting.hpp as RX_wrapper struct.
        '''

        dev_name = "raw_data" + str(int(metadata['usrp_number']))
        group_name = metadata['front_end_code']
        dataset_name = "dataset_" + str(int(metadata['packet_number']))
        try:
            ds = h5fp[dev_name][group_name].create_dataset(
                dataset_name,
                data=np.reshape(data, (metadata['channels'], metadata['length'] / metadata['channels']))
                # compression = H5PY_compression
            )
            ds.attrs.create(name="errors", data=metadata['errors'])
            if metadata['errors'] != 0:
                print_warning("The server encounterd a transmission error: " + str(metadata['errors']))
        except RuntimeError as err:
            print_error("A packet has not been written because of a problem: " + str(err))

    def create_h5_file(filename):
        '''
        Tries to open a h5 file without overwriting files with the same name. If the file already exists rename it and then create the file.

        Arguments:
            - String containing the name of the file.

        Returns:
            - Pointer to rhe opened file in write mode.
        '''
        filename = filename.split(".")[0]

        try:
            h5file = h5py.File(filename + ".h5", 'r')
            h5file.close()
        except IOError:
            try:
                h5file = h5py.File(filename + ".h5", 'w')
                return h5file
            except IOError as msg:
                print_error("Cannot create the file " + filename + ".h5:")
                print msg
                return ""
        else:
            print_warning(
                "Filename " + filename + ".h5 is already present in the folder, adding old(#)_ to the filename")
            count = 0
            while True:
                new_filename = "old(" + str(int(count)) + ")_" + filename + ".h5"
                try:
                    test = h5py.File(new_filename, 'r')
                    tets.close()
                except IOError:
                    os.rename(filename + ".h5", new_filename)
                    return open_h5_file(filename)
                else:
                    count += 1

    global USRP_data_queue, END_OF_MEASURE, EOM_cond, CLIENT_STATUS
    more_sample_than_expected_WARNING = True
    accumulated_timeout = 0
    sleep_time = 0.1

    acquisition_end_flag = False

    # this variable discriminate between a timeout condition generated
    # on purpose to wait the queue and one reached because of an error
    legit_off = False

    if filename == None:
        filename = "USRP_DATA_" + get_timestamp()
        print "Writing data on disk with filename: \"" + filename + ".h5\""

    H5_file_pointer = create_h5_file(str(filename))
    Param_to_H5(H5_file_pointer, parameters, trigger, **kwargs)

    allowed_counters = ['A_RX2','B_RX2']
    spc_acc = {}
    for fr_counter in allowed_counters:
        if parameters.parameters[fr_counter] != 'OFF': spc_acc[fr_counter] = 0

    CLIENT_STATUS["measure_running_now"] = True
    if dpc_expected is not None:
        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=dpc_expected)
    else:
        widgets = ['', progressbar.Counter('Samples per channel received: %(value)05d'),
               ' Client time elapsed: ', progressbar.Timer(), '']
        bar = progressbar.ProgressBar(widgets=widgets)
    data_warning = True
    bar.start()
    while (not acquisition_end_flag):
        try:
            meta_data, data = USRP_data_queue.get(timeout=0.1)
            # USRP_data_queue.task_done()
            accumulated_timeout = 0
            if meta_data == None:
                acquisition_end_flag = True
            else:
                # write_single_H5_packet(meta_data, data, H5_file_pointer)
                if trigger is not None:
                    data, meta_data = trigger.trigger(data, meta_data)
                write_ext_H5_packet(meta_data, data, H5_file_pointer, spc_acc[meta_data['front_end_code']], trigger = trigger)
                if push_queue is not None:
                    if not push_queue_warning:
                        try:
                            push_queue.put((meta_data, data))
                        except:
                            print_warning("Cannot push packets into external queue: %s"%str(sys.exc_info()[0]))
                            push_queue_warning = True

                spc_acc[meta_data['front_end_code']] += meta_data['length'] / meta_data['channels']
                try:
                    #print "max expected: %d total received %d"%(dpc_expected, spc_acc)
                    bar.update(spc_acc[meta_data['front_end_code']])
                except:
                    if data_warning:
                        if dpc_expected is not None:
                            bar.update(dpc_expected)
                        if (more_sample_than_expected_WARNING):
                            print_warning("Sync rx is receiving more data than expected...")
                            more_sample_than_expected_WARNING = False
                        data_warning = False

        except Empty:
            time.sleep(sleep_time)
            if timeout:
                accumulated_timeout += sleep_time
                if accumulated_timeout > timeout:
                    if not legit_off: print_warning("Sync data receiver timeout condition reached. Closing file...")
                    acquisition_end_flag = True
                    break
        if CLIENT_STATUS["keyboard_disconnect"] == True:
            Disconnect()
            acquisition_end_flag = True
            CLIENT_STATUS["keyboard_disconnect"] = False

        try:
            bar.update(spc_acc[meta_data['front_end_code']])
        except NameError:
            pass
        except:
            if (more_sample_than_expected_WARNING): print_debug("Sync RX received more data than expected.")

        EOM_cond.acquire()
        if END_OF_MEASURE:
            timeout = .5
            legit_off = True
        EOM_cond.release()

    bar.finish()

    EOM_cond.acquire()
    END_OF_MEASURE = False
    EOM_cond.release()

    if clean_data_queue() != 0:
        print_warning("Residual elements in the libUSRP data queue are being lost!")

    H5_file_pointer.close()
    print "\033[7;1;32mH5 file closed succesfully.\033[0m"
    CLIENT_STATUS["measure_running_now"] = False
    return filename

def USRP_socket_bind(USRP_socket, server_address, timeout):
    """
    Binds a soket object with a server address. Trys untill timeout seconds elaplsed.

    Args:
        - USRP_socket: socket object to bind with the address tuple.
        - server_address: a tuple containing a string with the ip address and a int representing the port.
        - timeout: timeout in seconds to wait for connection.

    Known bugs:
        - On some linux distribution once on two attempts the connection is denied by software. On third attempt however it connects.

    Returns:
        - True: if connection was succesfull.
        - False if no connection was established.

    Examples:
        >>> if(USRP_socket_bind(USRP_data_socket, USRP_server_address_data, 5)):
        #   do stuff with function in this library
        >>> else:
        >>>     print "No connection, check hardware and configs."

    Notes:
        - This method will only connect one soket to the USRP/GPU server, not data and async messages. This function is intended to be used in higher level functions contained in this library. The correct methot for connecting to USRP/GPU server is the use of USERP_Connect(timeout) function.
    """
    if timeout < 0:
        print_warning("No GPU server connection established after timeout.")
        return False
    else:
        try:
            USRP_socket.connect(server_address)
            return True
        except socket.error as msg:
            print(("Socket binding " + str(msg) + ", " + "Retrying..."))
            return False
            USRP_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            USRP_socket.settimeout(1)
            USRP_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            time.sleep(1)
            timeout = timeout - 1
            return USRP_socket_bind(USRP_socket, server_address, timeout)


def Decode_Sync_Header(raw_header, CLIENT_STATUS=CLIENT_STATUS):
    '''
    Decode an async header containing the metadata of the packet.

    Return:
        - The metadata in dictionary form.

    Arguments:
        - The raww header as a string (as returned by the recv() method of socket).
    '''

    def decode_frontend(code):
        return {
            'A': "A_TXRX",
            'B': "A_RX2",
            'C': "B_TXRX",
            'D': "B_RX2"
        }[code]

    try:
        header = np.fromstring(raw_header, dtype=header_type, count=1)
        metadata = {}
        metadata['usrp_number'] = header[0]['usrp_number']
        metadata['front_end_code'] = decode_frontend(header[0]['front_end_code'])
        metadata['packet_number'] = header[0]['packet_number']
        metadata['length'] = header[0]['length']
        metadata['errors'] = header[0]['errors']
        metadata['channels'] = header[0]['channels']
        return metadata
    except ValueError:
        if CLIENT_STATUS["keyboard_disconnect"] == False:
            print_error("Received corrupted header. No recover method has been implemented.")
        return None


def Print_Sync_Header(header):
    print "usrp_number" + str(header['usrp_number'])
    print "front_end_code" + str(header['front_end_code'])
    print "packet_number" + str(header['packet_number'])
    print "length" + str(header['length'])
    print "errors" + str(header['errors'])
    print "channels" + str(header['channels'])


def Decode_Async_header(header):
    ''' Extract the length of an async message from the header of an async package incoming from the GPU server'''
    header = np.fromstring(header, dtype=np.int32, count=2)
    if header[0] == 0:
        return header[1]
    else:
        return 0


def Decode_Async_payload(message):
    '''
    Decode asynchronous payloads coming from the GPU server
    '''
    global ERROR_STATUS, END_OF_MEASURE, REMOTE_FILENAME, EOM_cond

    try:
        res = json.loads(message)
    except ValueError:
        print_warning("Cannot decode response from server.")
        return

    try:
        atype = res['type']
    except KeyError:
        print_warning("Unexpected json string from the server: type")

    #    print "FROM SERVER: "+str(res['payload'])

    if atype == 'ack':
        if res['payload'].find("EOM") != -1:
            print_debug("Async message from server: Measure finished")
            EOM_cond.acquire()
            END_OF_MEASURE = True
            EOM_cond.release()
        elif res['payload'].find("filename") != -1:
            REMOTE_FILENAME = res['payload'].split("\"")[1]
        else:
            print_debug("Ack message received from the server: " + str(res['payload']))

    if atype == 'nack':
        print_warning("Server detected an error.")
        ERROR_STATUS = True
        EOM_cond.acquire()
        END_OF_MEASURE = True
        EOM_cond.release()


def Encode_async_message(payload):
    '''
    Format a JSON string so that the GPU server can read it.

    Arguments:
        - payload: A JSON string.

    Returns:
        - A formatted string ready to be sent via socket method

    Note:
        This function performs no check on the validity of the JSON string.
    '''

    return struct.pack('I', 0) + struct.pack('I', len(payload)) + payload


def Async_send(payload):
    '''
    Send a JSON string to the GPU server. Typically the JSON string represent a command or a status request.

    Arguments:
        -payload: JSON formatted string.

    Returns:
        -Boolean value representing the success of the operation.

    Note:
        In order to use this function the Async_thread has to be up and running. See Start_Async_RX().
    '''
    global Async_condition
    global Async_status
    global USRP_socket

    if (Async_status):

        # send the data
        try:
            USRP_socket.send(Encode_async_message(payload))
        except socket.error as err:
            print_warning("An async message could not be sent due to an error: " + str(err))
            if err.errno == 32:
                print_error("Async server disconnected")
                Async_condition.acquire()
                Async_status = False
                Async_condition.release()
            return False
        return True

    else:
        print_warning("The Async RX thread is not running, cannot send Async message.")
        return False


def Async_thread():
    '''Receiver thread for async messages from the GPU server. This function is ment to be run as a thread'''

    global Async_condition
    global Async_status
    global USRP_socket
    global USRP_server_address

    internal_status = True
    Async_status = False

    # the header iscomposed by two ints: one is always 0 and the other represent the length of the payload
    header_size = 2 * 4

    # just initialization of variables
    old_header_len = 0
    old_data_len = 0

    # try to connect, if it fails set internal status to False (close the thread)
    Async_condition.acquire()
    # if(not USRP_socket_bind(USRP_socket, USRP_server_address, 5)):
    time_elapsed = 0
    timeout = 10  # sys.maxint
    data_timeout_wait = 0.01
    connected = False
    while time_elapsed < timeout and (not connected):
        try:
            print_debug("Async command thread:")
            connected = USRP_socket_bind(USRP_socket, USRP_server_address, 7)
            time.sleep(1)
            time_elapsed += 1
        except KeyboardInterrupt:
            print_warning("Keyboard interrupt aborting connection...")
            break

    if not connected:
        internal_status = False
        Async_status = False

        print_warning("Async data connection failed")
        Async_condition.release()
    else:
        Async_status = True
        print_debug("Async data connected")
        Async_condition.release()
    # acquisition loop
    while (internal_status):

        # counter used to prevent the API to get stuck on sevrer shutdown
        data_timeout_counter = 0
        data_timeout_limit = 5

        header_timeout_limit = 5
        header_timeout_counter = 0
        header_timeout_wait = 0.1

        # lock the "mutex" for checking the state of the main API instance
        Async_condition.acquire()
        if Async_status == False:
            internal_status = False
        Async_condition.release()

        size = 0

        if (internal_status):
            header_data = ""
            try:
                while (len(header_data) < header_size) and internal_status:
                    header_timeout_counter += 1
                    header_data += USRP_socket.recv(min(header_size, header_size - len(header_data)))
                    if old_header_len != len(header_data):
                        header_timeout_counter = 0
                    if (header_timeout_counter > header_timeout_limit):
                        time.sleep(header_timeout_wait)
                    Async_condition.acquire()
                    if Async_status == False:
                        internal_status = False
                    # print internal_status
                    Async_condition.release()
                    old_header_len = len(header_data)
                    # general timer
                    time.sleep(.1)

                if (internal_status): size = Decode_Async_header(header_data)

            except socket.error as msg:
                if msg.errno != None:
                    print_error("Async header: " + str(msg))
                    Async_condition.acquire()
                    internal_status = False
                    Async_status = False
                    Async_condition.release()

        if (internal_status and size > 0):
            data = ""
            try:
                while (len(data) < size) and internal_status:
                    data_timeout_counter += 1
                    data += USRP_socket.recv(min(size, size - len(data)))
                    if old_data_len != len(data):
                        data_timeout_counter = 0
                    if (data_timeout_counter > data_timeout_limit):
                        time.sleep(data_timeout_wait)
                    Async_condition.acquire()
                    if Async_status == False:
                        internal_status = False
                    Async_condition.release()
                    old_data_len = len(data)

                if (internal_status): Decode_Async_payload(data)

            except socket.error as msg:
                if msg.errno == 4:
                    pass  # the ctrl-c exception is handled elsewhere
                elif msg.errno != None:
                    print_error("Async thread: " + str(msg))
                    Async_condition.acquire()
                    internal_status = False
                    Async_status = False
                    Async_condition.release()
                    print_warning("Async connection is down: " + msg)

    USRP_socket.shutdown(1)
    USRP_socket.close()
    del USRP_socket
    gc.collect()


Async_RX_loop = Thread(target=Async_thread, name="Async_RX", args=(), kwargs={})
Async_RX_loop.daemon = True


def Wait_for_async_connection(timeout=None):
    '''
    Block until async thead has established a connection with the server or the thread is expired. In case a timeout value is given, returns after timeout if no connection is established before.

    Arguments:
        - timeout: Second to wait for connection. Default is infinite timeout

    Return:
        - boolean representing the sucess of the operation.
    '''

    global Async_condition
    global Async_status
    time_elapsed = 0

    if timeout is None:
        timeout = sys.maxint
    try:
        while time_elapsed < timeout:
            Async_condition.acquire()

            x = Async_status

            Async_condition.release()

            time.sleep(1)

            if x:
                break
            else:
                time_elapsed += 1
    except KeyboardInterrupt:
        print_warning("keyboard interrupt received. Closing connections.")
        return False
    return x


def Wait_for_sync_connection(timeout=None):
    '''
    Block until async thead has established a connection with the server or the thread is expired. In case a timeout value is given, returns after timeout if no connection is established before.

    Arguments:
        - timeout: Second to wait for connection. Default is infinite timeout

    Return:
        - boolean representing the sucess of the operation.
    '''

    global Sync_RX_condition
    global CLIENT_STATUS
    time_elapsed = 0
    x = False
    if timeout is None:
        timeout = sys.maxint
    try:
        while time_elapsed < timeout:

            Sync_RX_condition.acquire()

            x = CLIENT_STATUS['Sync_RX_status']

            Sync_RX_condition.release()

            time.sleep(1)

            if x:
                break
            else:
                time_elapsed += 1
    except KeyboardInterrupt:
        print_warning("keyboard interrupt received. Closing connections.")
        return False
    return x


def Start_Async_RX():
    '''Start the Aswync thread. See Async_thread() function for a more detailed explanation.'''

    global Async_RX_loop
    reinit_async_socket()
    try:
        Async_RX_loop.start()
    except RuntimeError:
        Async_RX_loop = Thread(target=Async_thread, name="Async_RX", args=(), kwargs={})
        Async_RX_loop.daemon = True
        Async_RX_loop.start()
    # print "Async RX thread launched"


def Stop_Async_RX():
    '''Stop the Async thread. See Async_thread() function for a more detailed explanation.'''

    global Async_RX_loop, Async_condition, Async_status
    Async_condition.acquire()
    print_line("Closing Async RX thread...")
    Async_status = False
    Async_condition.release()
    Async_RX_loop.join()
    print_line("Async RX stopped")


def Connect(timeout=None):
    '''
    Connect both, the Syncronous and Asynchronous communication service.

    Returns:
        - True if both services are connected, False otherwise.

    Arguments:
        - the timeout in seconds. Default is retry forever.
    '''
    ret = True
    try:
        Start_Sync_RX()
        # ret &= Wait_for_sync_connection(timeout = 10)

        Start_Async_RX()
        ret &= Wait_for_async_connection(timeout=10)
    except KeyboardInterrupt:
        print_warning("keyboard interrupt received. Closing connections.")
        exit()

    return ret


def Disconnect(blocking=True):
    '''
    Disconnect both, the Syncronous and Asynchronous communication service.

    Returns:
        - True if both services are connected, False otherwise.

    Arguments:
        - define if the call is blocking or not. Default is blocking.
    '''
    Stop_Async_RX()
    Stop_Sync_RX()


def force_ternimate():
    global Sync_RX_loop, Async_RX_loop
    Sync_RX_loop.terminate()


def Sync_RX(CLIENT_STATUS, Sync_RX_condition, USRP_data_queue):
    '''
    Thread that recive data from the TCP data streamer of the GPU server and loads each packet in the data queue USRP_data_queue. The format of the data is specified in a subfunction fill_queue() and consist in a tuple containing (metadata,data).

    Note:
        This funtion is ment to be a standalone thread handled via the functions Start_Sync_RX() and Stop_Sync_RX().
    '''

    # global Sync_RX_condition
    # global Sync_RX_status
    global USRP_data_socket
    global USRP_server_address_data
    # global USRP_data_queue

    header_size = 5 * 4 + 1

    acc_recv_time = []
    cycle_time = []

    # use to pass stuff in the queue without reference
    def fill_queue(meta_data, dat, USRP_data_queue=USRP_data_queue):
        meta_data_tmp = meta_data
        dat_tmp = dat
        USRP_data_queue.put((meta_data_tmp, dat_tmp))

    # try to connect, if it fails set internal status to False (close the thread)
    # Sync_RX_condition.acquire()

    # if(not USRP_socket_bind(USRP_data_socket, USRP_server_address_data, 7)):
    time_elapsed = 0
    timeout = 10  # sys.maxint
    connected = False

    # try:
    while time_elapsed < timeout and (not connected):
        print_debug("RX sync data thread:")
        connected = USRP_socket_bind(USRP_data_socket, USRP_server_address_data, 7)
        time.sleep(1)
        time_elapsed += 1

    if not connected:
        internal_status = False
        CLIENT_STATUS['Sync_RX_status'] = False
        print_warning("RX data sync connection failed.")
    else:
        print_debug("RX data sync connected.")
        internal_status = True
        CLIENT_STATUS['Sync_RX_status'] = True

    # Sync_RX_condition.release()
    # acquisition loop
    start_total = time.time()

    while (internal_status):

        start_cycle = time.time()

        # counter used to prevent the API to get stuck on sevrer shutdown
        data_timeout_counter = 0
        data_timeout_limit = 5  # (seconds)

        header_timeout_limit = 5
        header_timeout_counter = 0
        header_timeout_wait = 0.01

        # lock the "mutex" for checking the state of the main API instance
        # Sync_RX_condition.acquire()
        if CLIENT_STATUS['Sync_RX_status'] == False:
            CLIENT_STATUS['Sync_RX_status'] = False
        # print internal_status
        # Sync_RX_condition.release()
        if (internal_status):
            header_data = ""
            try:
                old_header_len = 0
                header_timeout_counter = 0
                while (len(header_data) < header_size) and internal_status:
                    header_timeout_counter += 1
                    header_data += USRP_data_socket.recv(min(header_size, header_size - len(header_data)))
                    if old_header_len != len(header_data):
                        header_timeout_counter = 0
                    if (header_timeout_counter > header_timeout_limit):
                        time.sleep(header_timeout_wait)
                    # Sync_RX_condition.acquire()
                    if CLIENT_STATUS['Sync_RX_status'] == False:
                        internal_status = False
                    # print internal_status
                    # Sync_RX_condition.release()
                    old_header_len = len(header_data)
                    if (len(header_data) == 0): time.sleep(0.001)

            except socket.error as msg:
                if msg.errno == 4:
                    pass  # message is handled elsewhere
                elif msg.errno == 107:
                    print_debug("Interface connected too soon. This bug has not been covere yet.")
                else:
                    print_error("Sync thread: " + str(msg) + " error number is " + str(msg.errno))
                    # Sync_RX_condition.acquire()
                    internal_status = False
                    # Sync_RX_condition.release()

        if (internal_status):
            metadata = Decode_Sync_Header(header_data)
            if (not metadata):
                # Sync_RX_condition.acquire()
                internal_status = False
                # Sync_RX_condition.release()
            # Print_Sync_Header(metadata)

        if (internal_status):
            data = ""
            try:
                old_len = 0

                while ((old_len < 8 * metadata['length']) and internal_status):
                    data += USRP_data_socket.recv(min(8 * metadata['length'], 8 * metadata['length'] - old_len))

                    if (len(data) == old_len):
                        data_timeout_counter += 1
                    old_len = len(data)

                    if data_timeout_counter > data_timeout_limit:
                        print_error("Tiemout condition reached for buffer acquisition")
                        internal_status = False



            except socket.error as msg:
                print_error(msg)
                internal_status = False
        if (internal_status):
            try:
                formatted_data = np.fromstring(data[:], dtype=data_type, count=metadata['length'])

            except ValueError:
                print_error("Packet number " + str(metadata['packet_number']) + " has a length of " + str(
                    len(data) / float(8)) + "/" + str(metadata['length']))
                internal_status = False
            else:
                # USRP_data_queue.put((metadata,formatted_data))
                fill_queue(metadata, formatted_data)
    '''
    except KeyboardInterrupt:
            print_warning("Keyboard interrupt aborting connection...")
            internal_status = False
            CLIENT_STATUS['Sync_RX_status'] = False
    '''
    try:
        USRP_data_socket.shutdown(1)
        USRP_data_socket.close()
        del USRP_data_socket
        gc.collect()
    except socket.error:
        print_warning("Sounds like the server was down when the API tried to close the connection")

    # print "Sync client thread id down"

Sync_RX_loop = multiprocessing.Process(target=Sync_RX, name="Sync_RX",
                                       args=(CLIENT_STATUS, Sync_RX_condition, USRP_data_queue), kwargs={})
Sync_RX_loop.daemon = True

def signal_handler(sig, frame):
    if CLIENT_STATUS["measure_running_now"]:
        if CLIENT_STATUS["keyboard_disconnect"] == False:
            print_warning('Got Ctrl+C, Disconnecting and saving last chunk of data.')
            CLIENT_STATUS["keyboard_disconnect"] = True
            CLIENT_STATUS["keyboard_disconnect_attemp"] = 0
        else:
            print_debug("Already disconnecting...")
            CLIENT_STATUS["keyboard_disconnect_attemp"] += 1
            if CLIENT_STATUS["keyboard_disconnect_attemp"] > 2:
                print_warning("Forcing quit")
                force_ternimate();
                exit();
    else:
        force_ternimate();
        exit();

Signal.signal(Signal.SIGINT, signal_handler)

def Start_Sync_RX():
    global Sync_RX_loop, USRP_data_socket, USRP_data_queue
    try:
        try:
            del USRP_data_socket
            reinit_data_socket()
            USRP_data_queue = multiprocessing.Queue()
        except socket.error as msg:
            print msg
            pass
        Sync_RX_loop = multiprocessing.Process(target=Sync_RX, name="Sync_RX",
                                               args=(CLIENT_STATUS, Sync_RX_condition, USRP_data_queue), kwargs={})
        Sync_RX_loop.daemon = True
        Sync_RX_loop.start()
    except RuntimeError:
        print_warning("Falling back to threading interface for Sync RX thread. Network could be slow")
        Sync_RX_loop = Thread(target=Sync_RX, name="Sync_RX", args=(), kwargs={})
        Sync_RX_loop.daemon = True
        Sync_RX_loop.start()

def Stop_Sync_RX(CLIENT_STATUS=CLIENT_STATUS):
    global Sync_RX_loop, Sync_RX_condition
    # Sync_RX_condition.acquire()
    print_line("Closing Sync RX thread...")
    # print_line(" reading "+str(CLIENT_STATUS['Sync_RX_status'])+" from thread.. ")
    CLIENT_STATUS['Sync_RX_status'] = False
    time.sleep(.1)
    # Sync_RX_condition.release()
    # print "Process is alive? "+str(Sync_RX_loop.is_alive())
    if Sync_RX_loop.is_alive():
        Sync_RX_loop.terminate()  # I do not know why it's alive even if it exited all the loops
        # Sync_RX_loop.join(timeout = 5)
    print "Sync RX stopped"
