import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import colorlover as cl
import scipy.signal as signal
import h5py
import sys
import struct
import json
from plotly.graph_objs import Scatter, Layout
from plotly import tools

import socket
import Queue
from Queue import Empty
from threading import Thread,Condition
import time
import gc
import datetime

#needed to print the data acquisition process
import progressbar

def print_warning(message):
    print "\033[40;1;33mWARNING\033[0m: "+str(message)+"."
    
def print_error(message):
    print "\033[1;31mERROR\033[0m: "+str(message)+"."

def print_line(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    
#version of this library
libUSRP_net_version = "2.0"

#ip address of USRP server
USRP_IP_ADDR = 'localhost'

#soket used for command
USRP_server_address = (USRP_IP_ADDR, 22001)
USRP_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#address used for data
USRP_server_address_data = (USRP_IP_ADDR, 61360)
USRP_data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#queue for passing data and metadata from network
USRP_data_queue = Queue.Queue()

USRP_socket.settimeout(1)
USRP_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
USRP_data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

#reproduce the RX_wrapper struct in server_settings
header_type = np.dtype([
    ('usrp_number', np.int32),   
    ('front_end_code', np.dtype('|S1')), 
    ('packet_number', np.int32), 
    ('length', np.int32),
    ('errors', np.int32),
    ('channels', np.int32)
])

#data type expected for the buffer
data_type = np.complex64

#threading condition variables for controlling Sync RX thread activity
Sync_RX_condition = Condition()

#threading condition variables for controlling Async RX and TX thread activity
Async_condition = Condition()

#variable used to notify the end of a measure, See Decode_async_payload()
END_OF_MEASURE = False
#mutex guarding the variable above
EOM_cond = Condition()

#becomes true when a communication error occured
ERROR_STATUS = False

#in case the server has to communicate the current filename
REMOTE_FILENAME = ""

#variable ment to accumulate data coming from the data stream
DATA_ACCUMULATOR = []

#initially the status is off. It's set to True in the Sync_RX function/thread
Sync_RX_status = False
Async_status = False

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
            USRP_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            USRP_socket.settimeout(1)
            USRP_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print(("Socket binding " + str(msg) + ", " + "Retrying..."))
            time.sleep(1)
            timeout = timeout - 1
            return USRP_socket_bind(USRP_socket, server_address, timeout)

def Decode_Sync_Header(raw_header):
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
        print_error("Received corrupted header. No recover method has been implemented.")
        return None
        
def Print_Sync_Header(header):
    print "usrp_number"+str(header['usrp_number'])
    print "front_end_code"+str(header['front_end_code'])
    print "packet_number"+str(header['packet_number'])
    print "length"+str(header['length'])
    print "errors"+str(header['errors'])
    print "channels"+str(header['channels'])
    
def Decode_Async_header(header):
    ''' Extract the length of an async message from the header of an async package incoming from the GPU server'''
    header = np.fromstring(header, dtype=np.int32, count=2)
    if header[0] == 0:
        return header[1]
    else: return 0
    
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
    
    print "FROM SERVER: "+str(res['payload'])
        
    if atype == 'ack':
        if res['payload'].find("EOM")!=-1:
            print "Measure finished"
            EOM_cond.acquire()
            END_OF_MEASURE = True
            EOM_cond.release()
        elif res['payload'].find("filename")!=-1:
            REMOTE_FILENAME = res['payload'].split("\"")[1]
        else:
            print "General ack message received from the server: "+str(res['payload'])
            
            
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
    
    # control (4bytes = 0) + payload len (4bytes) + payload
    return struct.pack('I', 0) + struct.pack('I',len(payload)) + payload
    

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
    
    if(Async_status):
    
        #send the data
        try:
            USRP_socket.send(Encode_async_message(payload))
        except socket.error as err:
            print_warning("An async message could not be sent due to an error: "+str(err))
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
    '''Receiver thread for async messages from the GPU server'''
    
    global Async_condition
    global Async_status
    global USRP_socket
    global USRP_server_address
    
    internal_status = True
    Async_status = False
    
    #the header iscomposed by two ints: one is always 0 and the other represent the length of the payload
    header_size =  2*4
    
    #just initialization of variables
    old_header_len = 0
    old_data_len = 0
    
    #try to connect, if it fails set internal status to False (close the thread)
    Async_condition.acquire()
    if(not USRP_socket_bind(USRP_socket, USRP_server_address, 5)):
        
        internal_status = False
        Async_status = False
        
        print_warning("Async data connection failed")
        Async_condition.release()
    else:
        Async_status = True
        Async_condition.release()
    #acquisition loop
    while(internal_status):
        
        #counter used to prevent the API to get stuck on sevrer shutdown
        data_timeout_counter = 0
        data_timeout_limit = 5 #(seconds)
        
        header_timeout_limit = 5
        header_timeout_counter = 0
        header_timeout_wait = 0.1
        
        #lock the "mutex" for checking the state of the main API instance
        Async_condition.acquire()
        if Async_status == False:
            internal_status = False
        Async_condition.release()
        
        size = 0
        
        if(internal_status):
            header_data = ""
            try:
                while (len(header_data) < header_size) and internal_status:
                    header_timeout_counter += 1 
                    header_data += USRP_socket.recv(min(header_size, header_size-len(header_data)))
                    if old_header_len != len(header_data):
                            header_timeout_counter = 0
                    if (header_timeout_counter > header_timeout_limit):
                        time.sleep(header_timeout_wait)
                    Async_condition.acquire()
                    if Async_status == False:
                        internal_status = False
                    #print internal_status
                    Async_condition.release()
                    old_header_len = len(header_data)
                    
                if(internal_status): size = Decode_Async_header(header_data)
                
            except socket.error as msg:
                if msg.errno != None:
                    print_error("Async header: "+str(msg))
                    Async_condition.acquire()
                    internal_status = False
                    Async_status = False
                    Async_condition.release()
                
        if(internal_status and size>0):
            data = ""
            try:
                while (len(data) < size) and internal_status:
                    data_timeout_counter += 1 
                    data += USRP_socket.recv(min(size, size-len(data)))
                    if old_data_len != len(data):
                            data_timeout_counter = 0
                    if (data_timeout_counter > data_timeout_limit):
                        time.sleep(data_timeout_wait)
                    Async_condition.acquire()
                    if Async_status == False:
                        internal_status = False
                    Async_condition.release()
                    old_data_len = len(data)
                    
                if(internal_status): Decode_Async_payload(data)
                   
            except socket.error as msg:
                if msg.errno != None:
                    print_error(msg)
                    Async_condition.acquire()
                    internal_status = False
                    Async_status = False
                    Async_condition.release()
                    print_warning("Async connection is down: "+msg)

Async_RX_loop = Thread(target=Async_thread, name="Async_RX", args=(), kwargs={})
Async_RX_loop.daemon = True

def Wait_for_async_connection(timeout = None):
    '''
    Block until async thead has established a connection with the server or the thread is expired. In case a timeout value is given, returns after timeout if no connection is established before.
    
    Arguments:
        - timeout: Second to wait for connection. Default is infinite timeout
        
    Return:
        - boolean representing the sucess of the operation.
    '''
    
    global Async_condition
    global Async_status
    
    Async_condition.acquire()

    x = Async_status

    Async_condition.release()
    
    return x

def Wait_for_sync_connection(timeout = None):
    '''
    Block until async thead has established a connection with the server or the thread is expired. In case a timeout value is given, returns after timeout if no connection is established before.
    
    Arguments:
        - timeout: Second to wait for connection. Default is infinite timeout
        
    Return:
        - boolean representing the sucess of the operation.
    '''
    
    global Sync_RX_condition
    global Sync_RX_status
    
    Sync_RX_condition.acquire()

    x = Sync_RX_status

    Sync_RX_condition.release()   
    
    return x
         
def Start_Async_RX():

    '''Start the Aswync thread. See Async_thread() function for a more detailed explanation.'''
    
    global Async_RX_loop
    try:
        Async_RX_loop.start()
    except RuntimeError:
        Async_RX_loop = Thread(target=Async_thread, name="Async_RX", args=(), kwargs={})
        Async_RX_loop.daemon = True
        Async_RX_loop.start()
    #print "Async RX thread launched"
    

def Stop_Async_RX():

    '''Stop the Async thread. See Async_thread() function for a more detailed explanation.'''
    
    global Async_RX_loop,Async_condition,Async_status
    Async_condition.acquire()
    print_line("Closing Async RX thread...")
    Async_status = False
    Async_condition.release()
    Async_RX_loop.join()
    print_warning("Async RX stopped")
    
def Connect(timeout = None):
    '''
    Connect both, the Syncronous and Asynchronous communication service.
    
    Returns:
        - True if both services are connected, False otherwise.
        
    Arguments:
        - the timeout in seconds. Default is retry forever.
    '''
    Start_Sync_RX()
    Wait_for_sync_connection(timeout = None)
    Start_Async_RX()
    Wait_for_async_connection(timeout = None)
    
    
def Disconnect(blocking = True):
    '''
    Disconnect both, the Syncronous and Asynchronous communication service.
    
    Returns:
        - True if both services are connected, False otherwise.
        
    Arguments:
        - define if the call is blocking or not. Default is blocking.
    '''
    Stop_Async_RX()
    Stop_Sync_RX()
    
    
def Sync_RX():
    '''
    Thread that recive data from the TCP data streamer of the GPU server and loads each packet in the data queue USRP_data_queue. The format of the data is specified in a subfunction fill_queue() and consist in a tuple containing (metadata,data).
    
    Note:
        This funtion is ment to be a standalone thread handled via the functions Start_Sync_RX() and Stop_Sync_RX().
    '''
    global Sync_RX_condition
    global Sync_RX_status
    global USRP_data_socket
    global USRP_server_address_data
    global USRP_data_queue
    
    header_size = 5*4 + 1
    
    #use to pass stuff in the queue without reference
    def fill_queue(meta_data,dat):
        global USRP_data_queue
        meta_data_tmp = meta_data
        dat_tmp = dat
        USRP_data_queue.put((meta_data_tmp,dat_tmp))
        
    

    #try to connect, if it fails set internal status to False (close the thread)
    Sync_RX_condition.acquire()
    
    if(not USRP_socket_bind(USRP_data_socket, USRP_server_address_data, 5)):
        
        internal_status = False
        Sync_RX_status = False
        print_warning("RX data sync onnection failed.")
    else:
        internal_status = True
        Sync_RX_status = True
        
    Sync_RX_condition.release()
    #acquisition loop
    while(internal_status):
        #counter used to prevent the API to get stuck on sevrer shutdown
        data_timeout_counter = 0
        data_timeout_limit = 5 #(seconds)
        
        header_timeout_limit = 5
        header_timeout_counter = 0
        header_timeout_wait = 0.1
        
        #lock the "mutex" for checking the state of the main API instance
        Sync_RX_condition.acquire()
        if Sync_RX_status == False:
            internal_status = False
        #print internal_status
        Sync_RX_condition.release()
        if(internal_status):
            header_data = ""
            try:
                old_header_len = 0    
                header_timeout_counter = 0
                while (len(header_data) < header_size) and internal_status:
                    header_timeout_counter += 1
                    header_data += USRP_data_socket.recv(min(header_size, header_size-len(header_data)))
                    if old_header_len != len(header_data):
                        header_timeout_counter = 0
                    if (header_timeout_counter > header_timeout_limit):
                        time.sleep(header_timeout_wait)
                    Sync_RX_condition.acquire()
                    if Sync_RX_status == False:
                        internal_status = False
                    #print internal_status
                    Sync_RX_condition.release()
                    old_header_len = len(header_data)
                    if(len(header_data) == 0): time.sleep(0.01)
                    
            except socket.error as msg:
                print_error(msg)
                Sync_RX_condition.acquire()
                internal_status = False
                Sync_RX_condition.release()

        if(internal_status):
            metadata = Decode_Sync_Header(header_data)
            if(not metadata):
                Sync_RX_condition.acquire()
                internal_status = False
                Sync_RX_condition.release()
            #Print_Sync_Header(metadata)
            
            
        if(internal_status):
            data = ""
            try:
                old_len = 0
                while((old_len<8*metadata['length']) and internal_status):
                    data += USRP_data_socket.recv(min(8*metadata['length'], 8*metadata['length']-old_len))
                    if(len(data) == old_len):
                        data_timeout_counter+=1
                    old_len = len(data)
                    
                    if data_timeout_counter > data_timeout_limit:
                        print_error("Tiemout condition reached for buffer acquisition")
                        Sync_RX_condition.acquire()
                        internal_status = False
                        Sync_RX_condition.release()
                    
            except socket.error as msg:
                print_error(msg)
                Sync_RX_condition.acquire()
                internal_status = False
                Sync_RX_condition.release()
        if(internal_status):
            try:
                formatted_data = np.fromstring(data, dtype=data_type, count=metadata['length'])
            
            except ValueError:
                print_error("Packet number "+str(metadata['packet_number'])+" has a length of "+str(len(data)/float(8))+"/"+str(metadata['length']))
                internal_status = False
            else:
                fill_queue(metadata,formatted_data)
        '''        
        if(internal_status):
            R = USRP_data_queue.get()
            print "Received packet No. "+str((R[0])['packet_number'])
            gc.collect()
            del R
        '''
    try:        
        USRP_data_socket.close()
    except socket.error:
        print_warning("Sounds like the server was down when the API tried to close the connection")
    
 
Sync_RX_loop = Thread(target=Sync_RX, name="Sync_RX", args=(), kwargs={})
Sync_RX_loop.daemon = True

        
def Start_Sync_RX():
    global Sync_RX_loop
    try:
        Sync_RX_loop.start()
    except RuntimeError:
        Sync_RX_loop = Thread(target=Sync_RX, name="Sync_RX", args=(), kwargs={})
        Sync_RX_loop.daemon = True
        Sync_RX_loop.start()
    print "Thread launched"
    
def Stop_Sync_RX():
    global Sync_RX_loop,Sync_RX_condition,Sync_RX_status
    Sync_RX_condition.acquire()
    print_line("Closing Sync RX thread...")
    Sync_RX_status = False
    Sync_RX_condition.release()
    Sync_RX_loop.join()
    print_warning("Sync RX stopped")
    
def bound_open(filename):

    try:
        f = h5py.File(filename+".h5",'r')
    except IOError as msg:
        print_error("Cannot open the specified file: "+str(msg))
        f = None
    return f
    
def chk_multi_usrp(h5file):
    return len(h5file.keys())

def get_receivers(h5group):
    receivers = []
    for i in range( len(h5group.keys()) ):
        mode = h5group[h5group.keys()[i]].attrs.get("mode")
        if mode == "RX":
            receivers.append(str(h5group.keys()[i]))
    return receivers    
    
def openH5file(filename, ch_list= None, start_sample = None, last_sample = None, usrp_number = None, front_end = None, verbose = False):
    try:
        filename = filename.split(".")[0]
    except:
        print_error("cannot interpret filename while opening a H5 file")
        return None
        
    if(verbose):
        sys.stdout.write("Opening file \"" +filename+".h5\"... ")
        sys.stdout.flush()
        
    f = bound_open(filename)
    if not f:
        return np.asarray([])
        
    if(verbose):print "Done!"
    
    if(verbose):
        sys.stdout.write("Checking openH5 file function args... ")
        sys.stdout.flush()
    
    
    if chk_multi_usrp(f) == 0:
        print_error("No USRP data found in the hdf5 file")
        return np.asarray([])

    if (not usrp_number) and chk_multi_usrp(f) != 1:
        this_warning = "Multiple usrp found in the file but no preference given to open file function. Assuming usrp "+str((f.keys()[0]).split("ata")[1])
        print_warning(this_warning)
        group_name = "raw_data"+str((f.keys()[0]).split("ata")[1])
        
    if (not usrp_number) and chk_multi_usrp(f) == 1:
        group_name = f.keys()[0]
        
    if (usrp_number != None):
        group_name = "raw_data"+str(int(usrp_number))
    
    try:    
        group = f[group_name]
    except KeyError:
        print_error("Cannot recognize group format")
        return np.asarray([])
        
    recv = get_receivers(group)    
    
    if len(recv) == 0:
        print_error("No USRP data found in the hdf5 file for the selected usrp number. Maybe RX mode attribute is missing or the H5 file is only descriptive of TX commands")
        return np.asarray([])
    
    if (not front_end) and len(recv) != 1:
        this_warning = "Multiple acquisition frontend subgroups found but no preference given to open file function. Assuming "+recv[0]
        sub_group_name = recv[0]
        
    if (not front_end) and len(recv) == 1:    
        sub_group_name = recv[0]
        
    if front_end:
        sub_group_name = str(front_end)
        
    try:    
        sub_group = group[sub_group_name]
    except KeyError:
        print_error("Cannot recognize sub group format. For X300 USRP possible frontends are \"A_TXRX\",\"B_TXRX\",\"A_RX2\",\"B_RX2\"")
        return np.asarray([])

    n_chan = sub_group.attrs.get("n_chan")
    
    if n_chan == None:
        print_warning("There is no attribute n_chan in the data group, cannot execute checks. Number of channels will be deducted from dataset")
        n_chan = np.shape(sub_group["dataset_1"])[0]


        
    if ch_list == None:
        ch_list = range(n_chan)
    
    if  n_chan < max(ch_list):
        this_error = "Channel selected: " +str(max(ch_list))+ " in channels list in open file function exceed the total number of channels found: "+str(n_chan)
        print_error(this_error)
        return np.asarray([])
    
        
    if start_sample == None:
        start_sample = 0

    if start_sample < 0:
        print_warning("Start sample selected in open file function < 0: setting it to 0")
        start_sample = 0
        
    if last_sample == None:
        last_sample = sys.maxint    
        
    if last_sample < 0 or last_sample < start_sample:
        print_warning("Last sample selected in open file function < 0 or < Start sample: setting it to maxint")
        last_sample = sys.maxint
    
    if(verbose):print "Done!"
    
    if(verbose):
        sys.stdout.write("Collecting samples...")
        sys.stdout.flush()
    
    z = ()
    sample_index = 0
    sample_count = 0
    for i in range(len(sub_group.keys())):
        dataset_name = "dataset_"+str(int(1+i))
        current_len = np.shape(sub_group[dataset_name])[1]
        sample_index += current_len
        
        truncate_final = min(last_sample,last_sample - sample_index)
        if(last_sample>=sample_index):
            truncate_final = current_len
        elif(last_sample<sample_index):
            truncate_final = current_len - (sample_index - last_sample)
        

        if (sample_index > start_sample) and (truncate_final > 0):
            
            truncate_initial = max(0,current_len - (sample_index - start_sample))
            z += ((sub_group[dataset_name])[ch_list, truncate_initial:truncate_final],)
            sample_count += len((sub_group[dataset_name])[ch_list, truncate_initial:truncate_final])


            
    if(verbose):print "Done!"  
    return np.concatenate(z,1)
        
class global_parameter(object):
    '''
    Global paramenter object representing a measure.
    '''
    def __init__ (self):
        self.initialized = False
    
    def initialize(self):
        '''
        Initialize the parameter object to a zero configuration.
        '''
        if self.initialized == True:
            print_warning("Reinitializing global parameters to blank!")
        self.initialized = True
        empty_spec = {}
        empty_spec['mode'] = "OFF"
        empty_spec['rate'] = 0
        empty_spec['rf'] = 0
        empty_spec['gain'] = 0
        empty_spec['bw'] = 0
        empty_spec['samples'] = 0
        empty_spec['delay'] = 1
        empty_spec['burst_on'] = 0
        empty_spec['burst_off'] = 0
        empty_spec['buffer_len'] = 0
        empty_spec['freq'] = [0]
        empty_spec['wave_type'] = [0]
        empty_spec['ampl'] = [0]
        empty_spec['decim'] = 0
        empty_spec['chirp_f'] = [0]
        empty_spec['swipe_s'] = [0]
        empty_spec['chirp_t'] = [0]
        empty_spec['fft_tones'] = 0
        empty_spec['pf_average'] = 0
        prop = {}
        prop['A_TXRX'] = empty_spec.copy()
        prop['B_TXRX'] = empty_spec.copy()
        prop['A_RX2'] = empty_spec.copy()
        prop['B_RX2'] = empty_spec.copy()
        prop['device'] = 0
        self.parameters = prop.copy()
        
        
    def set(self,ant,param_name, val):
        '''
        Initialize the global parameters object and set a parameter value.
        
        Arguments:
            - ant: a string containing one of the following 'A_TXRX', 'B_TXRX', 'A_RX2', 'B_RX2'. Where the first letter ferers to the front end and the rest to the connector.
            - param_name: a string containing the paramenter name one wants to change. For a complete list of accepeted parameters see section ? in the documentation.
            - val: value to assign.
            
        Returns:
            Boolean value representing the success of the operation.
            
        Note:
            if the parameter object is already initialized it does not overwrite the other paramenters.
            This function DOES NOT perform any check on the input parameter; for that check the self_check() method.
        '''
        if not self.initialized:
            self.initialize()
        try:
            test = self.parameters[ant]
        except KeyError:
            print_error("The antenna \'"+ant+"\' is not an accepted frontend name.")
            return False        
        try:   
            test = self.parameters[ant][param_name] 
        except KeyError:
            print_error("The parameter \'"+param_name+"\' is not an accepted parameter.")
            return False
        self.parameters[ant][param_name] = val
        return True
        
        
        
    def is_legit(self):
        if not self.initialized:
            print_warning("Cannot return correct parameters because the parameter object has not been initialized")
            return None
        return \
        self.parameters['A_TXRX']['mode'] != "OFF" or\
        self.parameters['B_TXRX']['mode']  != "OFF" or\
        self.parameters['A_RX2']['mode']  != "OFF" or \
        self.parameters['B_RX2']['mode']  != "OFF"
    
    def self_check(self):
        '''
        Check if the parameters are coherent.

        Returns:
            booleare representing the result of the check
            
        Note:
            To know what's wrong, check the warnings.
        '''
        if self.initialized:
            if not self.is_legit():
                return False
            for ant_key in self.parameters:
                if ant_key == 'device':
                    continue
                if self.parameters[ant_key]['mode']!="OFF":
                    try:
                        len(self.parameters[ant_key]['chirp_f'])
                    except TypeError:
                        print_warning("\'chirp_f\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['chirp_f'] = [self.parameters[ant_key]['chirp_f']]
                    try:
                        int(self.parameters[ant_key]['chirp_f'][0])
                    except IndexError:
                        None
                    except TypeError:
                        None
                    except ValueError:
                        print_error("parameter \'chirp_f\' should be a list of numerical values, not a string")
                        return False
                    try:
                        len(self.parameters[ant_key]['swipe_s'])
                    except TypeError:
                        print_warning("\'swipe_s\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['swipe_s'] = [self.parameters[ant_key]['swipe_s']]       
                    try:
                        int(self.parameters[ant_key]['swipe_s'][0])
                    except IndexError:
                        None
                    except TypeError:
                        None
                    except ValueError:
                        print_error("parameter \'swipe_s\' should be a list of numerical values, not a string")
                        return False
                    try:
                        len(self.parameters[ant_key]['chirp_t'])
                    except TypeError:
                        print_warning("\'chirp_t\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['chirp_t'] = [self.parameters[ant_key]['chirp_t']]    
                    try:
                        int(self.parameters[ant_key]['chirp_t'][0])
                    except IndexError:
                        None
                    except TypeError:
                        None
                    except ValueError:
                        print_error("parameter \'chirp_t\' should be a list of numerical values, not a string")
                        return False
                    try:
                        len(self.parameters[ant_key]['freq'])
                    except TypeError:
                        print_warning("\'freq\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['freq'] = [self.parameters[ant_key]['freq']]   
                    try:
                        int(self.parameters[ant_key]['freq'][0])
                    except IndexError:
                        None
                    except TypeError:
                        None
                    except ValueError:
                        print_error("parameter \'freq\' should be a list of numerical values, not a string")
                        return False   
                    try:
                        len(self.parameters[ant_key]['wave_type'])
                    except TypeError:
                        print_warning("\'wave_type\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['wave_type'] = [self.parameters[ant_key]['wave_type']] 
                    
                    try:
                        len(self.parameters[ant_key]['ampl'])
                    except TypeError:
                        print_warning("\'ampl\" attribute in parameters has to be a list, changing value to list...")
                        self.parameters[ant_key]['ampl'] = [self.parameters[ant_key]['ampl']] 
                    try:
                        int(self.parameters[ant_key]['ampl'][0])
                    except IndexError:
                        None
                    except TypeError:
                        None
                    except ValueError:
                        print_error("parameter \'ampl\' should be a list of numerical values, not a string")
                        return False
                        
                        
                    #matching the integers conversion
                    for j in range(len(self.parameters[ant_key]['freq'])):
                        self.parameters[ant_key]['freq'][j] = int(self.parameters[ant_key]['freq'][j])
                    for j in range(len(self.parameters[ant_key]['swipe_s'])):
                        self.parameters[ant_key]['swipe_s'][j] = int(self.parameters[ant_key]['swipe_s'][j])
                    for j in range(len(self.parameters[ant_key]['chirp_f'])):
                        self.parameters[ant_key]['chirp_f'][j] = int(self.parameters[ant_key]['chirp_f'][j])
        else:
            return False
        
        print_warning("check function partially implemented yet")
        return True
        
    def from_dict(self,ant,dictionary):
        '''
        Initialize the global parameter object from a dictionary.
        
        Arguments:
            - ant: a string containing one of the following 'A_TXRX', 'B_TXRX', 'A_RX2', 'B_RX2'. Where the first letter ferers to the front end and the rest to the connector. 
            - dictionary: a dictionary containing the parameters.
            
        Note:
            if the object is already initialize it overwrites only the parameters in the dictionary otherwise it initializes the object to flat zero before applying the dictionary.
            
        Returns:
            - boolean representing the success of the operation
        '''    
        print_warning("function not implemented yet")
        return True
    def pprint(self):
        '''
        Output on terminal a diagnosti string representing the parameters.
        '''
        x = self.to_json()
        parsed = json.loads(x)
        print json.dumps(parsed, indent=4, sort_keys=True)
        
    def to_json(self):
        '''
        Convert the global parameter object to JSON string.
        
        Returns:
            - the string to be filled with the JSON
        '''
        return json.dumps(self.parameters)
    
    
    def get_active_rx_param(self):
        '''
        Discover which is(are) the active receiver designed by the global parameter object.
        
        Returns:
            list of active rx antenna names : this names correspond to the parameter group in the h5 file and to the dictionary name in the (this) parameter object. 
        '''
        if not self.initialized:
            print_warning("Cannot return correct parameters because the parameter object has not been initialized")
            return None
        
        if not self.is_legit():
            print_warning("There is no active RX channel in property tree")
            return None
            
        active_rx = []
        
        if self.parameters['A_TXRX']['mode'] == "RX":
            active_rx.append('A_TXRX')
        if self.parameters['B_TXRX']['mode'] == "RX":
            active_rx.append('B_TXRX')
        if self.parameters['A_RX2']['mode'] == "RX":
            active_rx.append('A_RX2')
        if self.parameters['B_RX2']['mode'] == "RX":
            active_rx.append('B_RX2')
            
        return active_rx
        
    def get_active_tx_param(self):
        '''
        Discover which is(are) the active emitters designed by the global parameter object.
        
        Returns:
            list of active tx antenna names : this names correspond to the parameter group in the h5 file and to the dictionary name in the (this) parameter object. 
        '''
        if not self.initialized:
            print_warning("Cannot return correct parameters because the parameter object has not been initialized")
            return None
        
        if not self.is_legit():
            print_warning("There is no active TX channel in property tree")
            return None
            
        active_tx = []
        
        if self.parameters['A_TXRX']['mode'] == "TX":
            active_tx.append('A_TXRX')
        if self.parameters['B_TXRX']['mode'] == "TX":
            active_tx.append('B_TXRX')
        if self.parameters['A_RX2']['mode'] == "TX":
            active_tx.append('A_RX2')
        if self.parameters['B_RX2']['mode'] == "TX":
            active_tx.append('B_RX2')
            
        return active_tx    

    def retrive_prop_from_file(self, filename, usrp_number = None):     
        def read_prop(group, sub_group_name):
            def missing_attr_warning(att_name, att):
                if att.all() == None:
                    print_warning("Parameter \""+str(att_name)+"\" is not defined")

            sub_prop = {}
            try:    
                sub_group = group[sub_group_name]
            except KeyError:
                sub_prop['mode'] = "OFF"
                return sub_prop
            
            sub_prop['mode'] = sub_group.attrs.get('mode')
            missing_attr_warning('mode', sub_prop['mode'])
                
            sub_prop['rate'] = sub_group.attrs.get('rate')
            missing_attr_warning('rate', sub_prop['rate'])

            sub_prop['rf'] = sub_group.attrs.get('rf')
            missing_attr_warning('rf', sub_prop['rf'])

            sub_prop['gain'] = sub_group.attrs.get('gain')
            missing_attr_warning('gain', sub_prop['gain'])

            sub_prop['bw'] = sub_group.attrs.get('bw')
            missing_attr_warning('bw', sub_prop['bw'])

            sub_prop['samples'] = sub_group.attrs.get('samples')
            missing_attr_warning('samples', sub_prop['samples'])

            sub_prop['delay'] = sub_group.attrs.get('delay')
            missing_attr_warning('delay', sub_prop['delay'])

            sub_prop['burst_on'] = sub_group.attrs.get('burst_on')
            missing_attr_warning('burst_on', sub_prop['burst_on'])

            sub_prop['burst_off'] = sub_group.attrs.get('burst_off')
            missing_attr_warning('burst_off', sub_prop['burst_off'])

            sub_prop['buffer_len'] = sub_group.attrs.get('buffer_len')
            missing_attr_warning('buffer_len', sub_prop['buffer_len'])

            sub_prop['freq'] = sub_group.attrs.get('freq')
            missing_attr_warning('freq', sub_prop['freq'])

            sub_prop['wave_type'] = sub_group.attrs.get('wave_type')
            missing_attr_warning('wave_type', sub_prop['wave_type'])

            sub_prop['ampl'] = sub_group.attrs.get('ampl')
            missing_attr_warning('ampl', sub_prop['ampl'])

            sub_prop['decim'] = sub_group.attrs.get('decim')
            missing_attr_warning('decim', sub_prop['decim'])

            sub_prop['chirp_t'] = sub_group.attrs.get('chirp_t')
            missing_attr_warning('chirp_t', sub_prop['chirp_t'])

            sub_prop['swipe_s'] = sub_group.attrs.get('swipe_s')
            missing_attr_warning('swipe_s', sub_prop['swipe_s'])

            sub_prop['fft_tones'] = sub_group.attrs.get('fft_tones')
            missing_attr_warning('fft_tones', sub_prop['fft_tones'])

            sub_prop['pf_average'] = sub_group.attrs.get('pf_average')
            missing_attr_warning('pf_average', sub_prop['pf_average'])

            return sub_prop
      
        f = bound_open(filename)
        if not f:
            return None
            
        if (not usrp_number) and chk_multi_usrp(f) != 1:
            this_warning = "Multiple usrp found in the file but no preference given to get prop function. Assuming usrp "+str((f.keys()[0]).split("ata")[1])
            print_warning(this_warning)
            group_name = "raw_data"+str((f.keys()[0]).split("ata")[1])
            
        if (not usrp_number) and chk_multi_usrp(f) == 1:
            group_name = f.keys()[0]
            
        if (usrp_number != None):
            group_name = "raw_data"+str(int(usrp_number))
        
        try:    
            group = f[group_name]
        except KeyError:
            print_error("Cannot recognize group format")
            return None
        
        prop = {}
        prop['A_TXRX'] = read_prop(group,'A_TXRX')
        prop['B_TXRX'] = read_prop(group,'B_TXRX')
        prop['A_RX2'] = read_prop(group,'A_RX2')
        prop['B_RX2'] = read_prop(group,'B_RX2')
        self.initialized = True
        self.parameters = prop
    

    
def spec_from_samples(samples, param = None, welch = None): 
  
    '''
    Calculate real and immaginary part of the dBc spectra of a complex array using the Welch method.
    
    Arguments:
        - Samples: complex array representing samples.
        - param: dictionary containing the ANTENNA parameter relative to the samples
        - welch: in how many segment to divide the samples given for applying the Welch method
        
    Returns:
        - Frequency array,
        - Immaginary spectrum array
        - Real spectrum array
    '''

    try:
        L = len(samples)
    except TypeError:
        print_error("Expecting complex array for dBc spectra calculation, got something esle.")
        return None, None ,None
        
    if np.median(samples) == 0:
        print_error("Median of the data resulted 0 in dBc spectra calculation.")
        return None, None ,None
        
    if welch == None:
        welch = L
    else:
        welch = L/welch
        
    if param == None:
        sampling_rate = 1
    else:
        try:
            sampling_rate = param['rate']/param['fft_tones']
            
        except TypeError:
            print_warning("Parameters passed to dbc spectrum evaluation are not valid. Sampling rate = 1")
            sampling_rate = 1
        
        except ZeroDivisionError:
            print_warning("Parameters passed to dbc spectrum evaluation are not valid. Sampling rate = 1")
            sampling_rate = 1
    
    samples = samples  / np.mean(samples)
    samples = samples - np.mean(samples)
    #samples = samples * (np.abs(np.mean(samples))/np.mean(samples))
    
    Frequencies , RealPart      = signal.welch( samples.real ,nperseg=welch, fs=sampling_rate ,detrend='linear',scaling='density')
    Frequencies , ImaginaryPart = signal.welch( samples.imag ,nperseg=welch, fs=sampling_rate ,detrend='linear',scaling='density')

    return Frequencies, RealPart, ImaginaryPart


def Device_chk(device):
    '''
    Check if the device is recognised by the server or assign to 0 by default.
    
    Arguments:
        - device number or None.
        
    Returns:
        - boolean representing the result of the check or true is assigned by default.
    '''
    if device == None:
        device = 0
        return True
    
    print_warning("Async HW information has not been implemented yet")
    return True

def Front_end_chk(Front_end):
    '''
    Check if the front end code is recognised by the server or assign to A by default.
    
    Arguments:
        - front end code (A or B) or None.
        
    Returns:
        - boolean representing the result of the check or true is assigned by default.
    '''
    if Front_end == None:
        Front_end = "A"
        return True
        
    if (Front_end!="A") or (Front_end!="B"):
        print_error("Front end \""+str(Front_end) +"\" not recognised in VNA scan setting.")
        return False
    
    return True
        
def Param_to_H5(H5fp, parameters_class,tag):
    '''
    Generate the internal structure of a H5 file correstonding to the parameters given.
    
    Arguments:
        - H5fp: already opened H5 file with write permissions
        - parameters: an initialized global_parameter object containing the informations used to drive the GPU server.
        
    Returns:
        - A list of names of H5 groups where to write incoming data.
        
    Note:
        This function is ment to be used inside the Packets_to_file() function for data collection.
    '''
    if parameters_class.self_check():
        rx_names = parameters_class.get_active_rx_param()
        tx_names = parameters_class.get_active_tx_param()
        usrp_group = H5fp.create_group("raw_data"+str(int(parameters_class.parameters['device'])))
        if tag != None:
            usrp_group.attrs.create(name = "tag", data=str(tag))
        for ant_name in tx_names:
            tx_group = usrp_group.create_group(ant_name)
            for param_name in parameters_class.parameters[ant_name]:
                tx_group.attrs.create(name = param_name, data=parameters_class.parameters[ant_name][param_name])
   
        for ant_name in rx_names: 
            rx_group = usrp_group.create_group(ant_name)
            for param_name in parameters_class.parameters[ant_name]:
                rx_group.attrs.create(name = param_name, data=parameters_class.parameters[ant_name][param_name])
            
        return rx_names
        
    else:
        print_error("Cannot initialize H5 file without checked parameters.self_check() failed.")
        return []
        
        
def Packets_to_variable(max_sample, average = None):
    '''
    Write all packets in the DATA_ACCUMULATOR variable
    '''
    global DATA_ACCUMULATOR, USRP_data_queue, END_OF_MEASURE


def to_list_of_str(user_input):
    '''
    Determines if the input is a string or a list of string. In case is not a list of string, returns a single element list; returns the list otherwise.
    
    Arguments:
        - string or list of strings.
        
    Returns:
        - list of strings.
        
    Note:
        - I'm assuming the strings contain filenames that can't be long 1.
    '''
    try:
        l = len(user_input[0])
    except:
        print_error("Something went wrong in the string to list of string conversion. Check function input: "+str(user_input))
        
    if l == 1:
        return [user_input]
    elif l == 0:
        print_warning("an input name has length 0.")
        return user_input
    else:
        return user_input
        
        
def get_timestamp():
    '''
    Returns the timestamp formatted in a stirng.
    
    Returns:
        string containing the timestamp.
    '''    
    return str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    
def Packets_to_file(parameters, timeout = None, filename = None, meas_tag = None, dpc_expected = None):
    '''
    Consume the USRP_data_queue and writes an H5 file on disk.
    
    Arguments:
        - parameters: global_parameter object containing the informations used to drive the GPU server.
        - timeout: time after which the function stops and tries to stop the server.
        - filename: eventual filename. Default is datetime.
        - meas_tag: an eventual string attribute to the raw_data# group to tag the measure.
        - dpc_expected: number of sample per channel expected. if given display a percentage progressbar.
        
    Returns:
        - filename or empty string if something went wrong
        
    Note:
        - if the \"End of measurement\" async signal is received from the GPU server the timeout mode becomes active.
    '''
    
    def write_single_H5_packet(metadata,data,h5fp):
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
            
        dev_name = "raw_data"+str(int(metadata['usrp_number']))
        group_name = metadata['front_end_code']
        dataset_name = "dataset_"+str(int(metadata['packet_number']))
        
        shaped_data = np.reshape(data,(metadata['channels'],metadata['length']/metadata['channels']))
        ds = h5fp[dev_name][group_name].create_dataset(dataset_name, data = shaped_data )
        ds.attrs.create(name = "errors", data=metadata['errors'])
        if metadata['errors'] != 0:
            print_warning("The server encounterd a transmission error: "+str(metadata['errors']))
        
    
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
            h5file = h5py.File(filename+".h5", 'r')
            h5file.close()
        except IOError:
            try:
                h5file = h5py.File(filename+".h5", 'w')
                return h5file
            except IOError as msg:
                print_error("Cannot create the file "+filename+".h5:")
                print msg
                return ""
        else:
            print_warning("Filename "+filename+".h5 is already present in the folder, adding old(#)_ to the filename")
            count = 0
            while True:
                new_filename = "old("+str(int(count))+")_"+filename+".h5"
                try:
                    test = h5py.File(new_filename ,'r')
                    tets.close()
                except IOError:
                    os.rename(filename+".h5",new_filename)
                    return open_h5_file(filename)
                else:
                    count += 1
        
    global USRP_data_queue, END_OF_MEASURE, EOM_cond
    
    accumulated_timeout = 0
    sleep_time = 0.1
    
    acquisition_end_flag = False
    
    spc_acc = 0
    
    if filename == None:
        filename = "USRP_DATA_"+get_timestamp()
        print "Writing data on disk with filename: \""+filename+".h5\""
    
    H5_file_pointer = create_h5_file(str(filename))
    Param_to_H5(H5_file_pointer, parameters, tag = meas_tag)
    if dpc_expected!=None:
        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=max_samp).start()
    else:
        widgets = [progressbar.FormatLabel(
            'Processed: %(value)d samples per channel (in: %(elapsed)s)')]
        bar = progressbar.ProgressBar(widgets=widgets)
    data_warning = True

    while(not acquisition_end_flag):
        try:
            try:
                meta_data, data = USRP_data_queue.get(timeout = 0.1)
                USRP_data_queue.task_done()
                accumulated_timeout = 0
                if meta_data == None:
                    acquisition_end_flag = True
                else:
                    write_single_H5_packet(meta_data, data, H5_file_pointer)
                    spc_acc += meta_data['length']/meta_data['channels']
                    try:
                        bar.update(spc_acc)
                    except:
                        if data_warning:
                            bar.update(dpc_expected)
                            print_warning("Sync rx is receiving more data than expected.")
                            data_warning = False
                
            except Empty:
                time.sleep(sleep_time)
                if timeout:
                    accumulated_timeout+=sleep_time
                    if accumulated_timeout>timeout:
                        print_warning("Sync data receiver timeout condition reached. Closing file...")
                        acquisition_end_flag = True
                        break
        except KeyboardInterrupt:
            print_warning("keyboard interrupt received. closing file...")
            acquisition_end_flag = True
            
        EOM_cond.acquire()             
        if END_OF_MEASURE:
            timeout = 1
        EOM_cond.release()
        
    print "Setting EOM mutex"
    EOM_cond.acquire()
    END_OF_MEASURE = False
    EOM_cond.release()   
             
    H5_file_pointer.close()
    print "H5 file closed succesfully."
    return filename

def Single_VNA(start_f, last_f, measure_t, n_points, tx_gain, Rate = None, decimation = True, RF = None, Front_end = None, Device = None, filename = None, Multitone_compensation = None, Iterations = 1, verbose = False):

    '''
    Perform a VNA scan.
    
    Arguments:
        - start_f: frequency in Hz where to start scanning (absolute if RF is not given, relative to RF otherwise).
        - last_f: frequency in Hz where to stop scanning (absolute if RF is not given, relative to RF otherwise).
        - measure_t: duration of the measure in seconds.
        - n_points: number of points to use in the VNA scan.
        - tx_gain: transmission amplifier gain.
        - Rate: Optional parameter to control the scan rate. Default is calculate from start_f an last_f args.
        - decimation: if True the decimation of the signal will occour on-server. Default is True.
        - RF: central up/down mixing frequency. Default is deducted by other arguments.
        - filename: eventual filename. default is datetime.
        - Front_end: the front end to be used on the USRP. default is A.
        - Device: the on-server device number to use. default is 0.
        - Multitone_compensation: integer representing the number of tones: compensate the amplitude of the signal to match a future multitones accuisition.
        - Iterations: by default a single VNA scan pass is performed.
        - verbose: if True outputs on terminal some diagnostic info. deafult is False.
        
    Returns:
        - filename where the measure is or empty string if something went wrong.
    '''        
    global USRP_data_queue, REMOTE_FILENAME, END_OF_MEASURE
    
    if measure_t <= 0:
        print_error("Cannot execute a VNA measure with "+str(measure_t)+"s duration.")
        return ""
    if n_points <= 0:
        print_error("Cannot execute a VNA measure with "+str(n_points)+" points.")
        return ""
    if RF == None:
        delta_f = np.abs(start_f - last_f)
        RF = delta_f/2.
        print "Setting RF central frequency to %.2f MHz"%(RF/1.e6)
    else:
        delta_f = max(start_f,last_f) - min(start_f,last_f)
        
    if delta_f > 1.6e8:
        print_error("Frequency range for the VNA scan is too large compared to maximum system bandwidth")
        return ""
    elif delta_f > 1e8:
        print_error("Frequency range for the VNA scan is too large compared to actual system bandwidth")
        return ""
    
    if not Device_chk(Device):
        return ""
        
    if not Front_end_chk(Front_end):
        return ""
        
    if Multitone_compensation == None:
        Amplitude = 1.
    else:
        Amplitude = 1./Multitone_compensation
       
    if decimation:
        decimation = 2
    else:
        decimation = 0
        
    number_of_samples = Rate* measure_t
       
        
    vna_command = global_parameter()
    
    vna_command.set("A_TXRX","mode", "TX")
    vna_command.set("A_TXRX","buffer_len", 1e6)
    vna_command.set("A_TXRX","gain", tx_gain)
    vna_command.set("A_TXRX","delay", 1)
    vna_command.set("A_TXRX","samples", number_of_samples)
    vna_command.set("A_TXRX","rate", Rate)
    vna_command.set("A_TXRX","bw", 2*Rate)
    
    vna_command.set("A_TXRX","wave_type", ["CHIRP"])
    vna_command.set("A_TXRX","ampl", [Amplitude])
    vna_command.set("A_TXRX","freq", [start_f])
    vna_command.set("A_TXRX","chirp_f", [last_f])
    vna_command.set("A_TXRX","swipe_s", [n_points])
    vna_command.set("A_TXRX","chirp_t", [measure_t])
    vna_command.set("A_TXRX","rf", RF)
    
    vna_command.set("A_RX2","mode", "RX")
    vna_command.set("A_RX2","buffer_len", 1e6)
    vna_command.set("A_RX2","gain", 0)
    vna_command.set("A_RX2","delay", 1)
    vna_command.set("A_RX2","samples", number_of_samples)
    vna_command.set("A_RX2","rate", Rate)
    vna_command.set("A_RX2","bw", 2*Rate)
    
    vna_command.set("A_RX2","wave_type", ["CHIRP"])
    vna_command.set("A_RX2","ampl", [Amplitude])
    vna_command.set("A_RX2","freq", [start_f])
    vna_command.set("A_RX2","chirp_f", [last_f])
    vna_command.set("A_RX2","swipe_s", [n_points])
    vna_command.set("A_RX2","chirp_t", [measure_t])
    vna_command.set("A_RX2","rf", RF)
    vna_command.set("A_RX2","decim", decimation)#THIS only activate the decimation.
    
    if vna_command.self_check():
        if(verbose): print "VNA command succesfully checked"
        Async_send(vna_command.to_json())
        
    else:
        print_warning("Something went wrong with the setting of VNA command.")
        return ""
    
        
    return ""
    
def Single_VNA_analysis(filename):
    '''
    Open a H5 file containing data collected with the function single_VNA() and analyze them as a VNA scan. Write the results in a corresponding VNA# group in the root of the H% file.
    
    Arguments:
        - filename: string containing the name of the H5 file
    
    Return:
        - boolean representing the success of the operation
    '''    
    return False
    
def Get_noise(tones, measure_t, rate, decimation = None, powers = None, RF = None, filename = None, Front_end = None, Device = None):
    '''
    Perform a noise acquisition using single tone technique.
    
    Arguments:
        - tones: list of tones frequencies in Hz (absolute if RF is not given, relative to RF otherwise).
        - measure_t: duration of the measure in seconds.
        - decimation: the decimation factor to use for the acquisition. Default is maximum.
        - powers: a list of linear power to use with each tone. Must be the same length of tones arg; will be normalized. Default is equally splitted power.
        - RF: central up/down mixing frequency. Default is deducted by other arguments.
        - filename: eventual filename. default is datetime.
        - Front_end: the front end to be used on the USRP. default is A.
        - Device: the on-server device number to use. default is 0.
        
    Returns:
        - filename where the measure is or empty string if something went wrong.
    '''    
    
    
    
    
    
    
    
