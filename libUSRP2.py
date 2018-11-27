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
from threading import Thread,Condition
import multiprocessing
from joblib import Parallel, delayed
from subprocess import call
import time
import gc
import datetime

#plotly stuff
from plotly.graph_objs import Scatter, Layout
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import colorlover as cl

#matplotlib stuff
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches

#needed to print the data acquisition process
import progressbar

def print_warning(message):
    print "\033[40;33mWARNING\033[0m: "+str(message)+"."
    
def print_error(message):
    print "\033[1;31mERROR\033[0m: "+str(message)+"."

def print_debug(message):
    print "\033[3;2;37m"+str(message)+"\033[0m"

def print_line(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    
#version of this library
libUSRP_net_version = "2.0"

#ip address of USRP server
USRP_IP_ADDR = '127.0.0.1'

#soket used for command
USRP_server_address = (USRP_IP_ADDR, 22001)
USRP_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#address used for data
USRP_server_address_data = (USRP_IP_ADDR, 61360)
USRP_data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

USRP_socket.settimeout(1)
USRP_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
USRP_data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

def reinit_data_socket():
    global USRP_data_socket
    USRP_data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    USRP_data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
def reinit_async_socket():
    global USRP_socket
    USRP_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    USRP_socket.settimeout(1)
    USRP_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
#queue for passing data and metadata from network
#USRP_data_queue = Queue.Queue()
USRP_data_queue = multiprocessing.Queue()

#compression used in H5py datasets
H5PY_compression = "gzip"
HDF5_compatible_compression = 'gzip'

#Cores to use in analysis
N_CORES = 10
parallel_backend = 'multiprocessing'

#usrp output power at 0 tx gain
USRP_power = -6.15

#Colors used for plotting
COLORS = ['black','red','green','blue','violet','brown','purple']

#this enable disable warning generated when more data that expected is received in the h5 file.
dynamic_alloc_warning = True


def get_color(N):
    '''
    Get a color for the list above without exceeding bounds. Usefull to change the overall color scheme.
    
    Arguments:
        N: index identifying stuff that has to have the same color.
        
    Return:
        string containing the color name.
    '''
    N = int(N)
    return COLORS[N%len(COLORS)]
    
def clean_data_queue(USRP_data_queue = USRP_data_queue):
    '''
    Clean the USRP_data_queue from residual elements. returns the number of element found in the queue.
    
    Returns:
        - Integer number of packets removed from the queue.
    '''
    #global USRP_data_queue
    print_debug("Cleaning data queue... ")
    residual_packets = 0
    while(True):
        try:
            meta_data, data = USRP_data_queue.get(timeout = 0.1)
            #USRP_data_queue.task_done()
            residual_packets += 1
        except Empty:
            break
    print_debug("Queue cleaned of "+str(residual_packets)+" packets.")
    return residual_packets
    


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
Async_status = False

#manager = multiprocessing.Manager() # this was too high level
from multiprocessing.managers import SyncManager
manager = SyncManager()
# explicitly starting the manager, and telling it to ignore the interrupt signal
# initializer for SyncManager
def mgr_init():
    Signal.signal(Signal.SIGINT, Signal.SIG_IGN)
    print_debug('initializing global variable manager...')
manager.start(mgr_init)

CLIENT_STATUS = manager.dict()
CLIENT_STATUS["Sync_RX_status"] = False
CLIENT_STATUS["keyboard_disconnect"] = False
CLIENT_STATUS["keyboard_disconnect_attemp"] = 0
CLIENT_STATUS["measure_running_now"] = False
#threading condition variables for controlling Sync RX thread activity
Sync_RX_condition = manager.Condition()




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

def Decode_Sync_Header(raw_header, CLIENT_STATUS = CLIENT_STATUS):
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
    
#    print "FROM SERVER: "+str(res['payload'])
        
    if atype == 'ack':
        if res['payload'].find("EOM")!=-1:
            print_debug("Async message from server: Measure finished")
            EOM_cond.acquire()
            END_OF_MEASURE = True
            EOM_cond.release()
        elif res['payload'].find("filename")!=-1:
            REMOTE_FILENAME = res['payload'].split("\"")[1]
        else:
            print_debug( "Ack message received from the server: "+str(res['payload']))
            
            
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
    #if(not USRP_socket_bind(USRP_socket, USRP_server_address, 5)):
    time_elapsed = 0
    timeout = 10#sys.maxint
    connected = False
    while time_elapsed < timeout and (not connected):
        try:
            print_debug("Async command thread:")
            connected = USRP_socket_bind(USRP_socket, USRP_server_address, 7)
            time.sleep(1)
            time_elapsed+=1
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
                    #general timer
                    time.sleep(.1)
                    
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
                if msg.errno == 4:
                    pass #the ctrl-c exception is handled elsewhere
                elif msg.errno != None:
                    print_error("Async thread: "+str(msg))
                    Async_condition.acquire()
                    internal_status = False
                    Async_status = False
                    Async_condition.release()
                    print_warning("Async connection is down: "+msg)

    USRP_socket.shutdown(1)
    USRP_socket.close()
    del USRP_socket
    gc.collect()

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

def Wait_for_sync_connection(timeout = None):
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
    #print "Async RX thread launched"
    

def Stop_Async_RX():

    '''Stop the Async thread. See Async_thread() function for a more detailed explanation.'''
    
    global Async_RX_loop,Async_condition,Async_status
    Async_condition.acquire()
    print_line("Closing Async RX thread...")
    Async_status = False
    Async_condition.release()
    Async_RX_loop.join()
    print_line("Async RX stopped")
    
def Connect(timeout = None):
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
        #ret &= Wait_for_sync_connection(timeout = 10)
        
        Start_Async_RX()
        ret &= Wait_for_async_connection(timeout = 10)
    except KeyboardInterrupt:
        print_warning("keyboard interrupt received. Closing connections.")
        exit()

    
    return ret
    
    
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

def force_ternimate():
    global Sync_RX_loop,Async_RX_loop
    Sync_RX_loop.terminate()
    
def Sync_RX(CLIENT_STATUS,Sync_RX_condition,USRP_data_queue):
    '''
    Thread that recive data from the TCP data streamer of the GPU server and loads each packet in the data queue USRP_data_queue. The format of the data is specified in a subfunction fill_queue() and consist in a tuple containing (metadata,data).
    
    Note:
        This funtion is ment to be a standalone thread handled via the functions Start_Sync_RX() and Stop_Sync_RX().
    '''

    #global Sync_RX_condition
    #global Sync_RX_status
    global USRP_data_socket
    global USRP_server_address_data
    #global USRP_data_queue

    header_size = 5*4 + 1
    
    acc_recv_time = []
    cycle_time = []
    
    #use to pass stuff in the queue without reference
    def fill_queue(meta_data,dat,USRP_data_queue=USRP_data_queue):
        meta_data_tmp = meta_data
        dat_tmp = dat
        USRP_data_queue.put((meta_data_tmp,dat_tmp))
        
    

    #try to connect, if it fails set internal status to False (close the thread)
    #Sync_RX_condition.acquire()
    
    #if(not USRP_socket_bind(USRP_data_socket, USRP_server_address_data, 7)):
    time_elapsed = 0
    timeout = 10#sys.maxint
    connected = False
    
    #try:
    while time_elapsed < timeout and (not connected):
        print_debug("RX sync data thread:")
        connected = USRP_socket_bind(USRP_data_socket, USRP_server_address_data, 7)
        time.sleep(1)
        time_elapsed+=1
        
        
    if not connected:    
        internal_status = False
        CLIENT_STATUS['Sync_RX_status'] = False
        print_warning("RX data sync connection failed.")
    else:
        print_debug("RX data sync connected.")
        internal_status = True
        CLIENT_STATUS['Sync_RX_status'] = True
        
    #Sync_RX_condition.release()
    #acquisition loop
    start_total = time.time()
    
    while(internal_status):
    
        start_cycle = time.time()
    
        #counter used to prevent the API to get stuck on sevrer shutdown
        data_timeout_counter = 0
        data_timeout_limit = 5 #(seconds)
        
        header_timeout_limit = 5
        header_timeout_counter = 0
        header_timeout_wait = 0.01
        
        #lock the "mutex" for checking the state of the main API instance
        #Sync_RX_condition.acquire()
        if CLIENT_STATUS['Sync_RX_status'] == False:
            CLIENT_STATUS['Sync_RX_status'] = False
        #print internal_status
        #Sync_RX_condition.release()
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
                    #Sync_RX_condition.acquire()
                    if CLIENT_STATUS['Sync_RX_status'] == False:
                        internal_status = False
                    #print internal_status
                    #Sync_RX_condition.release()
                    old_header_len = len(header_data)
                    if(len(header_data) == 0): time.sleep(0.001)
                    
            except socket.error as msg:
                if msg.errno == 4:
                    pass #message is handled elsewhere
                elif msg.errno == 107:
                    print_debug("Interface connected too soon. This bug has not been covere yet.")
                else:
                    print_error("Sync thread: "+str(msg)+" error number is "+str(msg.errno))
                    #Sync_RX_condition.acquire()
                    internal_status = False
                    #Sync_RX_condition.release()

        if(internal_status):
            metadata = Decode_Sync_Header(header_data)
            if(not metadata):
                #Sync_RX_condition.acquire()
                internal_status = False
                #Sync_RX_condition.release()
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
                        internal_status = False
                        
                
                    
            except socket.error as msg:
                print_error(msg)
                internal_status = False
        if(internal_status):
            try:
                formatted_data = np.fromstring(data[:], dtype=data_type, count=metadata['length'])

            except ValueError:
                print_error("Packet number "+str(metadata['packet_number'])+" has a length of "+str(len(data)/float(8))+"/"+str(metadata['length']))
                internal_status = False
            else:
                #USRP_data_queue.put((metadata,formatted_data))
                fill_queue(metadata,formatted_data)
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
    
    #print "Sync client thread id down"
    

Sync_RX_loop = multiprocessing.Process(target=Sync_RX, name="Sync_RX", args=(CLIENT_STATUS,Sync_RX_condition,USRP_data_queue), kwargs={})
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
    global Sync_RX_loop,USRP_data_socket,USRP_data_queue
    try:
        try:
            del USRP_data_socket
            reinit_data_socket()
            USRP_data_queue = multiprocessing.Queue()
        except socket.error as msg:
            print msg
            pass
        Sync_RX_loop = multiprocessing.Process(target=Sync_RX, name="Sync_RX", args=(CLIENT_STATUS,Sync_RX_condition,USRP_data_queue), kwargs={})
        Sync_RX_loop.daemon = True
        Sync_RX_loop.start()
    except RuntimeError:
        print_warning("Falling back to threading interface for Sync RX thread. Network could be slow")
        Sync_RX_loop = Thread(target=Sync_RX, name="Sync_RX", args=(), kwargs={})
        Sync_RX_loop.daemon = True
        Sync_RX_loop.start()
    
def Stop_Sync_RX(CLIENT_STATUS=CLIENT_STATUS):
    global Sync_RX_loop,Sync_RX_condition
    #Sync_RX_condition.acquire()
    print_line("Closing Sync RX thread...")
    #print_line(" reading "+str(CLIENT_STATUS['Sync_RX_status'])+" from thread.. ")
    CLIENT_STATUS['Sync_RX_status'] = False
    time.sleep(.1)
    #Sync_RX_condition.release()
    #print "Process is alive? "+str(Sync_RX_loop.is_alive())
    if Sync_RX_loop.is_alive():
        Sync_RX_loop.terminate() #I do not know why it's alive even if it exited all the loops
        #Sync_RX_loop.join(timeout = 5)
    print "Sync RX stopped"
    
def bound_open(filename):

    try:
        filename = format_filename(filename)
        f = h5py.File(filename,'r')
    except IOError as msg:
        print_error("Cannot open the specified file: "+str(msg))
        f = None
    return f
    
def chk_multi_usrp(h5file):
    n = 0
    for i in range(len(h5file.keys())):
        if h5file.keys()[i][:8] == 'raw_data':
            n+=1
    return n

def get_receivers(h5group):
    receivers = []
    subs = h5group.keys()
    for i in range( len(subs) ):
        mode = (h5group[subs[i]]).attrs.get("mode")
        if mode == "RX":
            receivers.append(str(h5group.keys()[i]))
    return receivers    

def get_rx_info(filename, ant = None):
    '''
    Retrive RX information from file. 
    
    Arguments:
        - optional ant string to specify receiver. Default is the first found.
    
    Return:
        Parameter dictionary
    '''
    filename = format_filename(filename)
    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)
    if ant is None:
        ant = parameters.get_active_rx_param()[0]
    else:
        ant = str(ant)
        
    return parameters.parameters[ant]
    

def openH5file(filename, ch_list= None, start_sample = None, last_sample = None, usrp_number = None, front_end = None, verbose = False, error_coord = False):
    '''
    Arguments:
        error_coord: if True returns (samples, err_coord) where err_coord is a list of tuples containing start and end sample of each faulty packet.
        
    '''
    
    try:
        filename = filename.split(".")[0]
    except:
        print_error("cannot interpret filename while opening a H5 file")
        return None
        
    if(verbose):
        print_debug("Opening file \"" +filename+".h5\"... ")

        
    f = bound_open(filename)
    if not f:
        return np.asarray([])
        
    if(verbose):
        print_debug("Checking openH5 file function args... ")
    
    
    if chk_multi_usrp(f) == 0:
        print_error("No USRP data found in the hdf5 file")
        return np.asarray([])

    if (not usrp_number) and chk_multi_usrp(f) != 1:
        this_warning = "Multiple usrp found in the file but no preference given to open file function. Assuming usrp "+str((f.keys()[0]).split("ata")[1])
        print_warning(this_warning)
        group_name = "raw_data"+str((f.keys()[0]).split("ata")[1])
        
    if (not usrp_number) and chk_multi_usrp(f) == 1:
        group_name = "raw_data0"#f.keys()[0]
        
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
        #print_warning("There is no attribute n_chan in the data group, cannot execute checks. Number of channels will be deducted from the freq attribute")
        n_chan = len(sub_group.attrs.get("wave_type"))
        print_debug("Getting number of cannels from wave_type attribute shape: %d channel(s) found"%n_chan)


        
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
    else:
        start_sample = int(start_sample)
        
    if last_sample == None:
        last_sample = sys.maxint  
    else:
        last_sample = int(last_sample)   
        
    if last_sample < 0 or last_sample < start_sample:
        print_warning("Last sample selected in open file function < 0 or < Start sample: setting it to maxint")
        last_sample = sys.maxint
    
    if(verbose):
        print_debug("Collecting samples...")


    z = []
    err_index = []
    sample_index = 0
    errors = 0
    
    #check if the opening mode is from the old server or the new one
    try:
        test = sub_group["dataset_1"]
        old_mode = True
    except KeyError:
        old_mode = False
    
    if old_mode:
        skip_warning = True
        print_debug("Using old dataset mode to open file \'%s\'"%filename)
        #data are contained in multiple dataset
            
        if verbose:
            widgets = [progressbar.Percentage(), progressbar.Bar()]
            bar = progressbar.ProgressBar(widgets=widgets, max_value=len(sub_group.keys())).start()
            read = 0
        
        current_len = np.shape(sub_group["dataset_1"])[1]
        N_dataset = len(sub_group.keys())

        print_warning("Raw amples inside "+filename+" have not been rearranged: the read from file can be slow for big files due to dataset reading overhead")
         
        for i in range(N_dataset):
            try:
                dataset_name = "dataset_"+str(int(1+i))
                
                sample_index += current_len
                
                truncate_final = min(last_sample,last_sample - sample_index)
                if(last_sample>=sample_index):
                    truncate_final = current_len
                elif(last_sample<sample_index):
                    truncate_final = current_len - (sample_index - last_sample)
                
                
                if (sample_index > start_sample) and (truncate_final > 0):
                    present_error = sub_group[dataset_name].attrs.get('errors')
                    errors += int(present_error)
                    if present_error !=0:
                        err_index.append( ( sample_index-current_len, sample_index+current_len))
                    truncate_initial = max(0,current_len - (sample_index - start_sample))
                    
                    z.append(sub_group[dataset_name][ch_list, truncate_initial:truncate_final])
            except KeyError:
                if skip_warning:
                    print_warning("Cannot find one or more datasets in the h5 file")
                    skip_warning = False
                
                
            if verbose:
                try:
                    bar.update(read)
                except:
                    print_debug("decrease samples in progeressbar")
                read+=1
        if errors > 0: print_warning("The measure opened contains %d erorrs!"%errors)
        if(verbose):print "Done!" 
        f.close()
    
        if error_coord:
            return np.concatenate(tuple(z),1),err_index
        return np.concatenate(tuple(z),1)
        
    else:
        samples = sub_group["data"].attrs.get("samples")
        if samples is None:
            print_warning("Non samples attrinut found: data extracted from file could include zero padding")
            samples = last_sample
        if len(sub_group["errors"])>0:
            print_warning("The measure opened contains %d erorrs!"%len(sub_group["errors"]))
        if error_coord:
            data = sub_group["data"][ch_list,start_sample:last_sample]  
            errors = sub_group["errors"][:]
            if errors is None:
                errors = []
            f.close()
            return data,errors
            
        data = sub_group["data"][ch_list,start_sample:last_sample]  
        print np.shape(data)
        f.close()
        return data
            
        
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
        empty_spec['pf_average'] = 4
        empty_spec['tuning_mode'] = 0
        prop = {}
        prop['A_TXRX'] = empty_spec.copy()
        prop['B_TXRX'] = empty_spec.copy()
        prop['A_RX2'] = empty_spec.copy()
        prop['B_RX2'] = empty_spec.copy()
        prop['device'] = 0
        self.parameters = prop.copy()
        
    def get(self,ant,param_name):
        if not self.initialized:
            print_error("Retriving parameters %s from an uninitialized global_parameter object"%param_name)
            return None
        try:
            test = self.parameters[ant]
        except KeyError:     
            print_error("The antenna \'"+ant+"\' is not an accepted frontend name or is not present.")
            return None
        try:   
            return test[param_name] 
        except KeyError:
            print_error("The parameter \'"+param_name+"\' is not an accepted parameter or is not present.")
            return None
        
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
                    try:
                        if self.parameters[ant_key]['tuning_mode'] is None: 
                            self.parameters[ant_key]['tuning_mode'] = 0
                        else:
                            try:
                                int(self.parameters[ant_key]['tuning_mode'])
                            except ValueError:
                                try:
                                    if "int" in str(self.parameters[ant_key]['tuning_mode']):
                                        self.parameters[ant_key]['tuning_mode'] = 0
                                    elif "frac" in str(self.parameters[ant_key]['tuning_mode']):
                                        self.parameters[ant_key]['tuning_mode'] = 1
                                except:
                                    print_warning("Cannot recognize tuning mode\'%s\' setting to integer mode."%str(self.parameters[ant_key]['tuning_mode']))
                                    self.parameters[ant_key]['tuning_mode'] = 0
                    except KeyError:
                        self.parameters[ant_key]['tuning_mode'] = 0    
                        
                    #matching the integers conversion
                    for j in range(len(self.parameters[ant_key]['freq'])):
                        self.parameters[ant_key]['freq'][j] = int(self.parameters[ant_key]['freq'][j])
                    for j in range(len(self.parameters[ant_key]['swipe_s'])):
                        self.parameters[ant_key]['swipe_s'][j] = int(self.parameters[ant_key]['swipe_s'][j])
                    for j in range(len(self.parameters[ant_key]['chirp_f'])):
                        self.parameters[ant_key]['chirp_f'][j] = int(self.parameters[ant_key]['chirp_f'][j])
                        
                    self.parameters[ant_key]['samples'] = int(self.parameters[ant_key]['samples'])
                #case in which it is OFF:
                else:
                    self.parameters[ant_key]['mode'] = "OFF"
                    self.parameters[ant_key]['rate'] = 0
                    self.parameters[ant_key]['rf'] = 0
                    self.parameters[ant_key]['gain'] = 0
                    self.parameters[ant_key]['bw'] = 0
                    self.parameters[ant_key]['samples'] = 0
                    self.parameters[ant_key]['delay'] = 1
                    self.parameters[ant_key]['burst_on'] = 0
                    self.parameters[ant_key]['burst_off'] = 0
                    self.parameters[ant_key]['buffer_len'] = 0
                    self.parameters[ant_key]['freq'] = [0]
                    self.parameters[ant_key]['wave_type'] = [0]
                    self.parameters[ant_key]['ampl'] = [0]
                    self.parameters[ant_key]['decim'] = 0
                    self.parameters[ant_key]['chirp_f'] = [0]
                    self.parameters[ant_key]['swipe_s'] = [0]
                    self.parameters[ant_key]['chirp_t'] = [0]
                    self.parameters[ant_key]['fft_tones'] = 0
                    self.parameters[ant_key]['pf_average'] = 4
                    self.parameters[ant_key]['tuning_mode'] = 1 # fractional
                    
        else:
            return False
        
        print_debug("check function is not complete yet. In case something goes unexpected, double check parameters.")
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
                if att == None:
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

            sub_prop['freq'] = sub_group.attrs.get('freq').tolist()
            missing_attr_warning('freq', sub_prop['freq'])

            sub_prop['wave_type'] = sub_group.attrs.get('wave_type').tolist()
            missing_attr_warning('wave_type', sub_prop['wave_type'])

            sub_prop['ampl'] = sub_group.attrs.get('ampl').tolist()
            missing_attr_warning('ampl', sub_prop['ampl'])

            sub_prop['decim'] = sub_group.attrs.get('decim')
            missing_attr_warning('decim', sub_prop['decim'])

            sub_prop['chirp_t'] = sub_group.attrs.get('chirp_t').tolist()
            missing_attr_warning('chirp_t', sub_prop['chirp_t'])

            sub_prop['swipe_s'] = sub_group.attrs.get('swipe_s').tolist()
            missing_attr_warning('swipe_s', sub_prop['swipe_s'])

            sub_prop['fft_tones'] = sub_group.attrs.get('fft_tones')
            missing_attr_warning('fft_tones', sub_prop['fft_tones'])

            sub_prop['pf_average'] = sub_group.attrs.get('pf_average')
            missing_attr_warning('pf_average', sub_prop['pf_average'])
            
            sub_prop['tuning_mode'] = sub_group.attrs.get('tuning_mode')
            missing_attr_warning('tuning_mode', sub_prop['tuning_mode'])

            return sub_prop
      
        f = bound_open(filename)
        if f is None:
            return None
            
        if (not usrp_number) and chk_multi_usrp(f) != 1:
            this_warning = "Multiple usrp found in the file but no preference given to get prop function. Assuming usrp "+str((f.keys()[0]).split("ata")[1])
            print_warning(this_warning)
            group_name = "raw_data0"#+str((f.keys()[0]).split("ata")[1])
            
        if (not usrp_number) and chk_multi_usrp(f) == 1:
            group_name = "raw_data0"#f.keys()[0]
            
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
            #Avoid dynamical disk space allocation by forecasting the size of the measure
            try:
                n_chan = len(parameters_class.parameters[ant_name]['wave_type'])
            except KeyError:
                print_warning("Cannot extract number of channel from signal processing descriptor")
                n_chan = 0
            
            if parameters_class.parameters[ant_name]['wave_type'][0] == "TONES":
                data_len = int(np.ceil(parameters_class.parameters[ant_name]['samples'] / (parameters_class.parameters[ant_name]['fft_tones']*  max(parameters_class.parameters[ant_name]['decim'],1))))
            elif parameters_class.parameters[ant_name]['wave_type'][0] == "CHIRP":
                if parameters_class.parameters[ant_name]['decim'] < 1:
                    data_len = parameters_class.parameters[ant_name]['samples']
                else:
                    data_len = 0
            elif parameters_class.parameters[ant_name]['wave_type'][0] == "NOISE":  
                data_len = int(np.ceil(parameters_class.parameters[ant_name]['samples'] / max(parameters_class.parameters[ant_name]['decim'],1)))
            else:
                print_warning("No file size could be determined from DSP descriptor: \'%s\'"%str(parameters_class.parameters[ant_name]['wave_type'][0]))
                data_len = 0
                
            data_shape_max = (n_chan,data_len)
            data_shape = (0,0)
            print "dataset initial length is %d"%data_len
            rx_group.create_dataset("data",data_shape_max , dtype = np.complex64, maxshape = (None,None), chunks=True)#, compression = H5PY_compression
            rx_group.create_dataset("errors",(0,0) ,dtype = np.dtype(np.int64), maxshape = (None,None))#, compression = H5PY_compression
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

def format_filename(filename):
    return os.path.splitext(filename)[0]+".h5"

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
    
    global dynamic_alloc_warning
    
    def write_ext_H5_packet(metadata,data,h5fp,index):
        '''
        Write a single packet inside an already opened and formatted H5 file as an ordered dataset.
        
        Arguments:
            - metadata: the metadata describing the packet directly coming from the GPU sevrer.
            - data: the data to be written inside the dataset.
            - dataset: file pointer to the h5 file. extensible dataset has to be already created.
            - index: dictionary containg the accumulated length of the dataset.
        
        Returns:
            - The updated index dictionaty.    
            
        Notes:
            - The way this function write the packets inside the h5 file is strictly related to the metadata type in decribed in USRP_server_setting.hpp as RX_wrapper struct.
        '''
        
        global dynamic_alloc_warning
        dev_name = "raw_data"+str(int(metadata['usrp_number']))
        group_name = metadata['front_end_code']
        samples_per_channel = metadata['length']/metadata['channels']
        dataset = h5fp[dev_name][group_name]["data"]
        errors = h5fp[dev_name][group_name]["errors"]
        data_shape = np.shape(dataset)
        data_start = index
        data_end = data_start + samples_per_channel
        
        try:
            if data_shape[0] < metadata['channels']:
                print_warning("Main dataset in H5 file not initialized.")
                dataset.resize(metadata['channels'],0)
            
            if data_end > data_shape[1]:
                if dynamic_alloc_warning:
                    print_warning("Main dataset in H5 file not correctly sized. Dynamically extending dataset...")
                    #print_debug("File writing thread is dynamically extending datasets.")
                    dynamic_alloc_warning = False
                dataset.resize(data_end,1)
            
            dataset[:,data_start:data_end] = np.reshape(data,(metadata['channels'],samples_per_channel))
            dataset.attrs.__setitem__("samples", data_end)
            
            if metadata['errors'] != 0:
                print_warning("The server encounterd an error")
                err_shape = np.shape(errors)
                err_len = err_shape[1]
                if err_shape[0] == 0:
                    errors.resize(2,0)
                errors.resize(err_len+1,1)
                errors[:,err_len] = [data_start,data_end]
        except RuntimeError as err:
            print_error("A packet has not been written because of a problem: "+str(err))

            
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
        try:
            ds = h5fp[dev_name][group_name].create_dataset(
                dataset_name,
                data = np.reshape(data,(metadata['channels'],metadata['length']/metadata['channels']))
                #compression = H5PY_compression
            )
            ds.attrs.create(name = "errors", data=metadata['errors'])
            if metadata['errors'] != 0:
                print_warning("The server encounterd a transmission error: "+str(metadata['errors']))
        except RuntimeError as err:
            print_error("A packet has not been written because of a problem: "+str(err))
            
    
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
        
    global USRP_data_queue, END_OF_MEASURE, EOM_cond, CLIENT_STATUS
    more_sample_than_expected_WARNING = True
    accumulated_timeout = 0
    sleep_time = 0.1
    
    acquisition_end_flag = False
    
    spc_acc = 0
    
    #this variable disciminate between a timeout condition generated on purpose to wait the queue and one reached because of an error
    legit_off = False
    
    if filename == None:
        filename = "USRP_DATA_"+get_timestamp()
        print "Writing data on disk with filename: \""+filename+".h5\""
    
    H5_file_pointer = create_h5_file(str(filename))
    Param_to_H5(H5_file_pointer, parameters, tag = meas_tag)
    CLIENT_STATUS["measure_running_now"] = True
    if dpc_expected!=None:
        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=dpc_expected).start()
    else:
        widgets = [progressbar.FormatLabel(
            '\033[7;1;32mReceived: %(value)d samples per channel in: %(elapsed)s\033[0m')]
        bar = progressbar.ProgressBar(widgets=widgets)
    data_warning = True

    while(not acquisition_end_flag):
        try:
            meta_data, data = USRP_data_queue.get(timeout = 0.1)
            #USRP_data_queue.task_done()
            accumulated_timeout = 0
            if meta_data == None:
                acquisition_end_flag = True
            else:
                #write_single_H5_packet(meta_data, data, H5_file_pointer)
                write_ext_H5_packet(meta_data, data, H5_file_pointer,spc_acc)
                spc_acc += meta_data['length']/meta_data['channels']
                try:
                    bar.update(spc_acc)
                except:
                    if data_warning:
                        bar.update(dpc_expected)
                        if(more_sample_than_expected_WARNING):
                            print_warning("Sync rx is receiving more data than expected...")
                            more_sample_than_expected_WARNING = False
                        data_warning = False
            
        except Empty:
            time.sleep(sleep_time)
            if timeout:
                accumulated_timeout+=sleep_time
                if accumulated_timeout>timeout:
                    if not legit_off: print_warning("Sync data receiver timeout condition reached. Closing file...")
                    acquisition_end_flag = True
                    break
        if CLIENT_STATUS["keyboard_disconnect"] == True:
            Disconnect()
            acquisition_end_flag = True
            CLIENT_STATUS["keyboard_disconnect"]  = False

        try:
            bar.update(spc_acc)
        except:
            if(more_sample_than_expected_WARNING): print_debug("Sync RX received more data than expected.")
        bar.finish    
        EOM_cond.acquire()             
        if END_OF_MEASURE:
            timeout = .5
            legit_off = True
        EOM_cond.release()
        
    EOM_cond.acquire()
    END_OF_MEASURE = False
    EOM_cond.release()   
    
    if clean_data_queue() != 0:
        print_warning("Residual elements in the libUSRP data queue are being lost!")
             
    H5_file_pointer.close()
    print "\033[7;1;32mH5 file closed succesfully.\033[0m"
    CLIENT_STATUS["measure_running_now"] = False
    return filename

def linear_phase(phase):
    '''
    Unwrap the pgase and subtract linear and constrant offset.
    '''
    phase = np.unwrap(phase)
    x =np.arange(len(phase))
    m,q = np.polyfit(x, phase, 1)
    
    linear_phase = m*x +q
    phase -= linear_phase

    return phase
    
def spec_from_samples(samples, sampling_rate = 1, welch = None, dbc = False, rotate = False, verbose = True): 
  
    '''
    Calculate real and immaginary part of the spectra of a complex array using the Welch method.
    
    Arguments:
        - Samples: complex array representing samples.
        - sampling_rate: sampling_rate
        - welch: in how many segment to divide the samples given for applying the Welch method
        - dbc: scales samples to calculate dBc spectra.
        - rotate: if True rotate the IQ plane
    Returns:
        - Frequency array,
        - Immaginary spectrum array
        - Real spectrum array
    '''
    if verbose: print_debug("[Welch worker]")
    try:
        L = len(samples)
    except TypeError:
        print_error("Expecting complex array for spectra calculation, got something esle.")
        return None, None ,None
        
    if welch == None:
        welch = L
    else:
        welch = int(L/welch)
    
    if rotate:    
        samples = samples * (np.abs(np.mean(samples))/np.mean(samples))
        
    if dbc:
        samples = samples  / np.mean(samples)
        samples = samples - np.mean(samples)
    
    
    Frequencies , RealPart      = signal.welch( samples.real ,nperseg=welch, fs=sampling_rate ,detrend='linear',scaling='density')
    Frequencies , ImaginaryPart = signal.welch( samples.imag ,nperseg=welch, fs=sampling_rate ,detrend='linear',scaling='density')
    
    return Frequencies, RealPart, ImaginaryPart
    
def calculate_noise(filename, welch = None, dbc = False, rotate = False, usrp_number = 0, ant = None, verbose = True, clip = 0.5):
    '''
    Generates the FFT of each channel stored in the .h5 file and stores the results in the same file.
    '''
    
    if verbose: print_debug("Calculating noise spectra for "+filename)
    
    if verbose: print_debug("Reading attributes...")
    
    filename = format_filename(filename)
    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)
    
    if ant is None:
        ant = parameters.get_active_rx_param()
    else:
        ant = to_list_of_str(ant)
    
    if len(ant) > 1:
        print_error("multiple RX devices not yet supported")
        return
    
    active_RX_param = parameters.parameters[ant[0]]

    try:
        sampling_rate = active_RX_param['rate']/active_RX_param['fft_tones']
        
    except TypeError:
        print_warning("Parameters passed to spectrum evaluation are not valid. Sampling rate = 1")
        sampling_rate = 1
    
    except ZeroDivisionError:
        print_warning("Parameters passed to spectrum evaluation are not valid. Sampling rate = 1")
        sampling_rate = 1
    
    if sampling_rate != 1:
        clip_samples = clip * sampling_rate
    else:
        clip_samples = None
    
    if verbose: print_debug("Opening file...")
    
    samples, errors = openH5file(
        filename,
        ch_list= None,
        start_sample = clip_samples ,
        last_sample = None,
        usrp_number = usrp_number,
        front_end = None, # this will change for double RX
        verbose = True,
        error_coord = True
    )
    if len(errors)>0:
        print_error("Cannot evaluate spectra of samples containing transmission error")
        return
    
    if verbose: print_debug("Calculating spectra...")
    
    Results = Parallel(n_jobs=N_CORES,verbose=1 ,backend  = parallel_backend)(
        delayed(spec_from_samples)(
            np.asarray(i), sampling_rate = sampling_rate, welch = welch, dbc = dbc, rotate = rotate, verbose = verbose
        ) for i in samples
    )
    '''
    Results = spec_from_samples(
            samples, sampling_rate = sampling_rate, welch = welch, dbc = dbc, rotate = rotate, verbose = verbose
        )
    '''
    if verbose: print_debug("Saving result on file "+filename+" ...")
    
    fv = h5py.File(filename,'r+')
    
    noise_group_name = "Noise"+str(int(usrp_number))
    
    try:
        noise_group = fv.create_group(noise_group_name)
    except ValueError:
        noise_group = fv[noise_group_name]
        
    try:  
        noise_subgroup = noise_group.create_group(ant[0]) 
    except ValueError:     
        if verbose: print_debug("Overwriting Noise subgroup %s in h5 file"%ant[0])
        del noise_group[ant[0]]
        noise_subgroup = noise_group.create_group(ant[0])
        
    noise_subgroup.attrs.create(name = "welch", data=welch)
    noise_subgroup.attrs.create(name = "dbc", data=dbc)
    noise_subgroup.attrs.create(name = "rotate", data=rotate)
    noise_subgroup.attrs.create(name = "rate", data=sampling_rate)
    noise_subgroup.attrs.create(name = "n_chan", data=len(Results))
    
    noise_subgroup.create_dataset("freq", data = Results[0][0], compression = H5PY_compression)
    
    for i in range(len(Results)):
        tone_freq = active_RX_param['rf']+active_RX_param['freq'][i]
        ds = noise_subgroup.create_dataset("real_"+str(i), data = Results[i][1], compression=H5PY_compression ,dtype=np.dtype('Float32'))
        ds.attrs.create(name = "tone", data=tone_freq)
        ds = noise_subgroup.create_dataset("imag_"+str(i), data = Results[i][2], compression=H5PY_compression ,dtype=np.dtype('Float32'))
        ds.attrs.create(name = "tone", data=tone_freq)
        
    if verbose: print_debug("calculate_noise_spec() has done.")
    fv.close()
    
def get_noise(filename, usrp_number = 0, front_end = None, channel_list = None):
    '''
    Get the noise samples froma a pre-analized H5 file.
    
    Argumers:
        - filename: [string] the name of the file.
        - usrp_number: the server number of the usrp device. default is 0.
        - front_end: [string] name of the front end. default is extracted from data.
        - channel_list: [listo of int] specifies the channels from which to get samples
    Returns:
        - Noise info, Frequency axis, real axis, imaginary axis
    
    Note:
        Noise info is a dictionary containing the following parameters [whelch, dbc, rotate, rate, tone].
        The first four give inforamtion about the fft done to extract the noise; the last one is a list coherent with channel list containing the acquisition frequency of each tone in Hz.
    '''
    if usrp_number is None:
        usrp_number = 0
    
    filename = format_filename(filename)
    fv = h5py.File(filename,'r')
    noise_group = fv["Noise"+str(int(usrp_number))]
    if front_end is not None:
        ant = front_end
    else:
        if len(noise_group.keys())>0:
            ant = noise_group.keys()[0]
        else:
            print_error("get_noise() cannot find valid front end names in noise group!")
            raise IndexError
                    
    noise_subgroup = noise_group[ant]
    
    info = {}
    info['welch'] = noise_subgroup.attrs.get("welch")
    info['dbc'] = noise_subgroup.attrs.get("dbc")
    info['rotate'] = noise_subgroup.attrs.get("rotate")
    info['rate'] = noise_subgroup.attrs.get("rate")
    info['n_chan'] = noise_subgroup.attrs.get("n_chan")
    
    if channel_list is None:
        channel_list = range(info['n_chan'])
    
    info['tones'] = []
    
    frequency_axis = np.asarray(noise_subgroup['freq'])
    real = []
    imag = []
    for i in channel_list:
       real.append(np.asarray(noise_subgroup['real_'+str(int(i))]))
       imag.append(np.asarray(noise_subgroup['imag_'+str(int(i))]))
       info['tones'].append(noise_subgroup['imag_'+str(int(i))].attrs.get("tone"))
       
    fv.close()
    
    return info, frequency_axis, real, imag
    
def get_readout_power(filename, channel, front_end = None, usrp_number = 0):
    '''
    Get the readout power for a given single tone channel.
    '''
    global USRP_power
    if usrp_number is None:
        usrp_number = 0
    filename = format_filename(filename)
    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)
    if front_end is None:
        ant = parameters.get_active_tx_param()
    else:
        ant = [front_end]
    try:
        ampl = parameters.get(ant[0],'ampl')[channel]
    except IndexError:
        print_error("Channel %d is not present in file %s front end %s"%(channel,filename,front_end))
        raise IndexError
        
    gain = parameters.get(ant[0],'gain')
    
    return gain + USRP_power + 20*np.log10(ampl)
    
def plot_noise_spec(filenames, channel_list = None, max_frequency = None, title_info = None, backend = 'matplotlib', cryostat_attenuation = 0, auto_open = True, output_filename = None, **kwargs):
    '''
    Plot the noise spectra of given, pre-analized, H5 files.
    
    Arguments:
        - filenames: list of strings containing the filenames.
        - channel_list:
        - max_frequency: maximum frequency to plot.
        - title_info: add a custom line to the plot title
        - backend: see plotting backend section for informations.
        - auto_open: open the plot in default system browser if plotly backend is selected (non-blocking) or open the matplotlib figure (blocking). Default is True.
        - output_filename: string: if given the function saves the plot (in png for matplotlib backend and html for plotly backend) with the given name.
        - **kwargs: usrp_number and front_end can be passed to the openH5file() function. tx_front_end can be passed to manually determine the tx frontend to calculate the readout power. add_info could be a list of the same length og filenames containing additional leggend informations.
    '''
    filenames = to_list_of_str(filenames)
    
    add_info_labels = None
    try:
        add_info_labels = kwargs['add_info']
    except KeyError:
        pass
    
    plot_title = 'USRP Noise spectra from '
    if len(filenames)<2:
        plot_title+="file: "+filenames[0]+"."
    else:
        plot_title+= "multiple files."
        
        
    if backend == 'matplotlib':
        fig, ax = pl.subplots(nrows=1, ncols=1)
        try:
            fig.set_size_inches(kwargs['size'][0],kwargs['size'][1])
        except KeyError:
            pass
        ax.set_xlabel("Frequency [Hz]")
        
    elif backend == 'plotly':
        fig = tools.make_subplots(rows=1, cols=1)
        fig['layout']['xaxis1'].update(title="Frequency [Hz]",type='log')
        
        
    y_name_set = True
    rate_tag_set = True
    
    try:
        usrp_number = kwargs['usrp_number']
    except KeyError:
        usrp_number = None
    try:
        front_end = kwargs['front_end']
    except KeyError:
        front_end = None
    try:
        tx_front_end = kwargs['tx_front_end']
    except KeyError:
        tx_front_end = None
    f_count = 0    
    for filename in filenames:
        info, freq, real, imag = get_noise(
            filename,
            usrp_number = usrp_number,
            front_end = front_end,
            channel_list = channel_list
        )
        if y_name_set:
            y_name_set = False
            
            if backend == 'matplotlib':
                if info['dbc']:
                    ax.set_ylabel("PSD [dBc]")
                else:
                    ax.set_ylabel("PSD [dBm/sqrt(Hz)]")
                    
            elif backend == 'plotly':
                if info['dbc']:
                    fig['layout']['yaxis1'].update(title="PSD [dBc]")
                else:
                    fig['layout']['yaxis1'].update(title="PSD [dBm/sqrt(Hz)]")
                    
        if rate_tag_set:
            rate_tag_set = False
            if info['rate']/1e6 > 1.:
                plot_title+= "Effective rate: %.2f Msps"%(info['rate']/1e6)
            else: 
                plot_title+= "Effective rate: %.2f ksps"%(info['rate']/1e3)
            
        for i in range(len(info['tones'])):
            readout_power = get_readout_power(filename, i, tx_front_end, usrp_number)-cryostat_attenuation
            label = "%.2f MHz"%(info['tones'][i]/1e6)
            R = 10*np.log10(real[i])
            I = 10*np.log10(imag[i])
            if backend == 'matplotlib':
                label+= "\nReadout pwr %.1f dBm"%(readout_power)
                if add_info_labels is not None:
                    label+="\n"+add_info_labels[f_count]
                ax.semilogx(freq, R, '--', color = get_color(f_count+i), label = "Real "+label)
                ax.semilogx(freq, I, color = get_color(f_count+i), label = "Imag "+label)
            elif backend == 'plotly':    
                label+= "<br>Readout pwr %.1f dBm"%(readout_power)
                if add_info_labels is not None:
                    label+="<br>"+add_info_labels[f_count]
                fig.append_trace(go.Scatter(
                    x = freq,
                    y = R,
                    name = "Real "+label,
                    legendgroup = "group" +str(f_count+i),
                    line = dict(color = get_color(f_count+i)),
                    mode = 'lines'
                ), 1, 1)
                fig.append_trace(go.Scatter(
                    x = freq,
                    y = I,
                    name = "Imag " + label,
                    legendgroup = "group" +str(f_count+i),
                    line = dict(color = get_color(f_count+i),dash = 'dot'),
                    mode = 'lines'
                ), 1, 1)
        #increase file counter
        f_count+=1
        
    if backend == 'matplotlib':
        if title_info is not None:
            plot_title+= "\n"+title_info
        fig.suptitle(plot_title)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc=7)
        ax.grid(True)   
        if output_filename is not None:
            fig.savefig(output_filename+'.png')
        if auto_open:
            pl.show()
        else:
            pl.close()
            
    elif backend == 'plotly':
        if title_info is not None:
            plot_title+= "<br>"+title_info
            
        fig['layout'].update( title=plot_title)
        if output_filename is None:
            output_filename = "raw_data_plot"

        plotly.offline.plot(fig, filename=output_filename+".html",auto_open=auto_open)
    
def plot_raw_data(filenames, decimation = None, low_pass = None, backend = 'matplotlib', output_filename = None, channel_list = None, mode = 'IQ', start_time = None, end_time = None, auto_open = True, **kwargs):
    '''
    Plot raw data group from given H5 files.
    
    Arguments:
        - a list of strings containing the files to plot.
        - decimation: eventually deciamte the signal before plotting.
        - low pass: floating point number controlling the cut-off frequency of a low pass filter that is eventually applied to the data.
        - backend: [string] choose the return type of the plot. Allowed backends for now are:
            * matplotlib: creates a matplotlib figure, plots in non-blocking mode and return the matplotlib figure object. **kwargs in this case accept:
                - size: size of the plot in the form of a tuple (inches,inches). Default is matplotlib default.
            * plotly: plot using plotly and webgl interface, returns the html code descibing the plot. **kwargs in this case accept:
                - size: size of the plot. Default is plotly default.
        - output_filename: string: if given the function saves the plot (in png for matplotlib backend and html for plotly backend) with the given name.
        - channel_list: select only al list of channels to plot.
        - mode: [string] how to print the IQ signals. Allowed modes are:
            * IQ: default. Just plot the IQ signal with no processing.
            * PM: phase and magnitude. The fase will be unwrapped and the offset will be removed.
        - start_time: time where to start plotting. Default is 0.
        - end_time: time where to stop plotting. Default is end of the measure.
        - auto_open: open the plot in default system browser if plotly backend is selected (non-blocking) or open the matplotlib figure (blocking). Default is True.
        - **kwargs:
            * usrp_number and front_end can be passed to the openH5file() function.
            * size: the size of matplotlib figure.
            * add_info: list of strings as long as the file list to add info to the legend.
    Returns:
        - the return is up tho the backend choosen and could be a matplotlib figure or a html string.
        
    Note:
        - Possible errors are signaled on the plot.
    '''
    add_info_labels = None
    try:
        add_info_labels = kwargs['add_info']
    except KeyError:
        pass
    plot_title = 'USRP raw data acquisition. '
    if backend == 'matplotlib':
        fig, ax = pl.subplots(nrows=2, ncols=1, sharex=True)
        try:
            fig.set_size_inches(kwargs['size'][0],kwargs['size'][1])
        except KeyError:
            pass
        ax[1].set_xlabel("Time [s]")
        if mode == 'IQ':
            ax[0].set_ylabel("I [fp ADC]")
            ax[1].set_ylabel("Q [fp ADC]")
        elif mode == 'PM':
            ax[0].set_ylabel("Magnitude [abs(ADC)]")
            ax[1].set_ylabel("Phase [Rad]")
            
    elif backend == 'plotly':   
        if mode == 'IQ':
            fig = tools.make_subplots(rows=2, cols=1,subplot_titles=('I timestream', 'Q timestream'),shared_xaxes=True)
            fig['layout']['yaxis1'].update(title='I [fp ADC]')
            fig['layout']['yaxis2'].update(title='Q [fp ADC]')
        elif mode == 'PM':
            fig = tools.make_subplots(rows=2, cols=1,subplot_titles=('Magnitude', 'Phase'),shared_xaxes=True)
            fig['layout']['yaxis1'].update(title='Magnitude [abs(ADC)]')
            fig['layout']['yaxis2'].update(title='Phase [Rad]')
            
        fig['layout']['xaxis1'].update(title='Time [s]')    
        
    filenames = to_list_of_str(filenames)
    
    print_debug("Plotting from files:")
    for i in range(len(filenames)):
        print_debug("%d) %s"%(i,filenames[i]))
    try:
        usrp_number = kwargs['usrp_number']
    except KeyError:
        usrp_number = None
    try:
        front_end = kwargs['front_end']
    except KeyError:
        front_end = None
    file_count = 0    
    for filename in filenames:
        
        filename = format_filename(filename)
        parameters = global_parameter()
        parameters.retrive_prop_from_file(filename)
        ant = parameters.get_active_rx_param()
        
        if len(ant) > 1:
            print_error("multiple RX devices not yet supported")
            return
        freq = None
        if parameters.get(ant[0],'wave_type')[0] == "TONES":
            decimation_factor = parameters.get(ant[0],'fft_tones')
            freq =  np.asarray(parameters.get(ant[0],'freq')) + parameters.get(ant[0],'rf')
            if parameters.get(ant[0],'decim') != 0:
                decimation_factor*=parameters.get(ant[0],'decim')
                
            effective_rate = parameters.get(ant[0],'rate') / float(decimation_factor)
            
        elif parameters.get(ant[0],'wave_type')[0] == "CHIRP":
            decimation_factor = 1
            if parameters.get(ant[0],'decim') !=0:
                decimation_factor *= parameters.get(ant[0],'decim') * parameters.get(ant[0],'chirp_t')[0]/parameters.get(ant[0],'swipe_s')[0]
            effective_rate = parameters.get(ant[0],'rate') / float(decimation_factor)
            
        else:
            decimation_factor = max(1,parameters.get(ant[0],'fft_tones'))
                   
            if parameters.get(ant[0],'decim') != 0:
                decimation_factor*=parameters.get(ant[0],'decim')
                
            effective_rate = parameters.get(ant[0],'rate') / float(decimation_factor)   
                
        if start_time is not None:
            file_start_time = start_time* effective_rate
        else:
            file_start_time = 0
        if end_time is not None:
            file_end_time = end_time * effective_rate
        else:
            file_end_time = None
        
        samples, errors = openH5file(
            filename,
            ch_list= channel_list,
            start_sample = file_start_time,
            last_sample = file_end_time,
            usrp_number = usrp_number,
            front_end = front_end,
            verbose = False,
            error_coord = True
        )
        print_debug("plot_raw_data() found %d channels each long %d samples"%(len(samples),len(samples[0])))
        if channel_list == None:
            ch_list  = range(len(samples))
        else:
            if max(channel_list) > len(samples):
                print_warning("Channel list selected in plot_raw_data() is bigger than avaliable channels. plotting all available channels")
                ch_list  = range(len(samples))
            else:
                ch_list = channel_list
                
        #prepare samples TODO
        for i in ch_list:
        
            if mode == 'IQ':
                Y1 = samples[i].real
                Y2 = samples[i].imag
            elif mode == 'PM':
                Y1 = np.abs(samples[i])
                Y2 = np.angle(samples[i]) 
            
            if decimation is not None and decimation > 1:
                decimation = int(np.abs(decimation))
                Y1 = signal.decimate(Y1,decimation,ftype = 'fir')
                Y2 = signal.decimate(Y2,decimation,ftype = 'fir')
            else:
                decimation = 1.
            
            X = np.arange(len(Y1))/float(effective_rate/decimation) + file_start_time
        
            if effective_rate/1e6 > 1:
                rate_tag = 'DAQ rate: %.2f Msps'%(effective_rate/1e6)
            else:
                rate_tag = 'DAQ rate: %.2f ksps'%(effective_rate/1e3)
            
            
            if freq is None:
                label = "Channel %d"%i
            else:
                label = "Channel %.2f MHz"%(freq[i]/1.e6)
                
            if backend == 'matplotlib':
                if add_info_labels is not None:
                    label+="\n"+add_info_labels[f_count]
                ax[0].plot(X,Y1, color = get_color(i+file_count), label = label)
                ax[1].plot(X,Y2, color = get_color(i+file_count))
            elif backend == 'plotly':
                if add_info_labels is not None:
                    label+="<br>"+add_info_labels[f_count]
                fig.append_trace(go.Scatter(
                    x = X,
                    y = Y1,
                    name = label,
                    legendgroup = "group" +str(i+file_count),
                    line = dict(color = get_color(i+file_count)),
                    mode = 'lines'
                ), 1, 1)
                fig.append_trace(go.Scatter(
                    x = X,
                    y = Y2,
                    #name = "channel %d"%i,
                    showlegend = False,
                    legendgroup = "group" +str(i+file_count),
                    line = dict(color = get_color(i+file_count)),
                    mode = 'lines'
                ), 2, 1)
        file_count+=1
    if backend == 'matplotlib':
        for error in errors:
            err_start_coord = (error[0]-decimation/2)/float(effective_rate) + file_start_time
            err_end_coord = (error[1] + decimation/2 )/float(effective_rate) + file_start_time
            ax[0].axvspan(err_start_coord, err_end_coord, facecolor='yellow', alpha=0.4)
            ax[1].axvspan(err_start_coord, err_end_coord, facecolor='yellow', alpha=0.4)
        fig.suptitle(plot_title+"\n"+rate_tag)
        handles, labels = ax[0].get_legend_handles_labels()
        if len(errors)>0:
            yellow_patch = mpatches.Patch(color='yellow', label='ERRORS')
            handles.append(yellow_patch)
            labels.append('ERRORS')
        fig.legend(handles, labels, loc=7)
        ax[0].grid(True)   
        ax[1].grid(True)   
        if output_filename is not None:
            fig.savefig(output_filename+'.png')
        if auto_open:
            pl.show()
        else:
            pl.close()
            
    if backend == 'plotly':

        fig['layout'].update( title=plot_title+"<br>"+rate_tag)

        if output_filename is None:
            output_filename = "PFB_waterfall"
        plotly.offline.plot(fig, filename=output_filename+".html",auto_open=auto_open)
        
def plot_all_pfb(filename, decimation = None, low_pass = None, backend = 'matplotlib', output_filename = None, start_time = None, end_time = None, auto_open = True, **kwargs):
    '''
    Plot the output of a PFB acquisition as an heatmap.
    '''      
    filename = format_filename(filename)
    parameters = global_parameter()
    parameters.retrive_prop_from_file(filename)
    ant = parameters.get_active_rx_param()
    try:
        usrp_number = kwargs['usrp_number']
    except KeyError:
        usrp_number = None
    if len(ant) > 1:
        print_error("multiple RX devices not yet supported")
        return
    
    if parameters.get(ant[0],'wave_type')[0] != "NOISE":
        print_warning("The file selected does not have the PFB acquisition tag. Errors may occour")
        
    fft_tones = parameters.get(ant[0],'fft_tones')
    rate = parameters.get(ant[0],'rate')
    channel_width = rate/fft_tones
    decimation = parameters.get(ant[0],'decim')
    integ_time = fft_tones*max(decimation,1)/rate
    rf = parameters.get(ant[0],'rf')
    if start_time is not None:
        start_time *= effective_rate
    else:
        start_time = 0
    if end_time is not None:
        end_time *= effective_rate 
        
    try:
        front_end = kwargs['front_end']
    except KeyError:
        front_end = None
    samples, errors = openH5file(
            filename,
            ch_list= None,
            start_sample = start_time,
            last_sample = end_time,
            usrp_number = usrp_number,
            front_end = front_end,
            verbose = False,
            error_coord = True
        )
        
    y_label = np.arange(len(samples[0])/fft_tones)/(rate/(fft_tones*max(1,decimation)))
    x_label = (rf+(np.arange(fft_tones)-fft_tones/2)*(rate/fft_tones))/1e6
    title = "PFB acquisition form file %s"%filename
    subtitle = "Channel width %.2f kHz; Frame integration time: %.2e s"%(channel_width/1.e3, integ_time)
    z = 20*np.log10(np.abs(samples[0]))
    try:
        z_shaped = np.roll(np.reshape(z,(len(z)/fft_tones,fft_tones)),fft_tones/2,axis = 1)
    except ValueError as msg:
        print_warning("Error while plotting pfb spectra: "+str(msg))
        cut = len(z) - len(z)/fft_tones*fft_tones
        z = z[:-cut]
        print_debug("Cutting last data (%d samples) to fit"%cut)
        #z_shaped = np.roll(np.reshape(z,(len(z)/fft_tones,fft_tones)),fft_tones/2,axis = 1)
        z_shaped = np.roll(np.reshape(z,(len(z)/fft_tones,fft_tones)),fft_tones/2,axis = 1)
    
    #pl.plot(z_shaped.T, alpha = 0.1, color = "k")
    #pl.show()
    
    if backend == 'matplotlib':
        fig, ax = pl.subplots(nrows=2, ncols=1, sharex=True)
        try:
            fig.set_size_inches(kwargs['size'][0],kwargs['size'][1])
        except KeyError:
            pass
        ax[0].set_xlabel("Channel [MHz]")
        ax[0].set_ylabel("Time [s]")
        ax[0].set_title(title+"\n"+subtitle)
        imag = ax[0].imshow(z_shaped,aspect = 'auto',interpolation = 'nearest',extent=[min(x_label),max(x_label),min(y_label),max(y_label)])
        #fig.colorbar(imag)#,ax=ax[0]
        for zz in z_shaped[::100]:
            ax[1].plot(x_label,zz, color = 'k', alpha = 0.1)
        ax[1].set_xlabel("Channel [MHz]")
        ax[1].set_ylabel("Power [dBm]")
        #ax[1].set_title("Trace stack")
        
        
        if output_filename is not None:
            fig.savefig(output_filename+'.png')
        if auto_open:
            pl.show()
        else:
            pl.close()
            
    if backend == 'plotly':
        data = [
            go.Heatmap(
                z=z_shaped,
                x=x_label,
                y=y_label,
                colorscale='Viridis',
            )
        ]

        layout = go.Layout(
            title=title+"<br>"+subtitle,
            xaxis = dict(title = "Channel [MHz]"),
            yaxis = dict(title = "Time [s]")
        )

        fig = go.Figure(data=data, layout=layout)
        if output_filename is None:
            output_filename = "PFB_waterfall"
        plotly.offline.plot(fig, filename=output_filename+".html",auto_open=auto_open)
        
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
    
    
    
    
    
    
    
