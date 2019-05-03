#########################################################
# DEVELOPED TO MAKE SINGLE FILE CONTAINING A BEAM MAP   #
# OF OPTICALLY COUPLED KIDS                             #
#########################################################

from threading import Thread
import csv,sys
import socket
import time

try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print "Cannot find the pyUSRP package"

POSITION_SERVER_ADDR = '137.79.43.78'
POSITION_SERVER_PORT = 30245
POSITION_ACTIVE = True #Controls activity of the receiver thread
def ETCPClient(tcpIP, tcpPort):
    BUFFER_SIZE = 14
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((tcpIP, tcpPort))
    data = s.recv(BUFFER_SIZE)
    s.close()

    ux = float(data[0:6])
    uy = float(data[7:])

    return ux, uy

def position_acquisition(filename):
    '''
    Connect to the server dispensing the position and write to file.
    This function is ment to be run as a thread.

    :param filename The name of the file filenamecontaining the data.
    '''
    global POSITION_SERVER_ADDR
    global POSITION_SERVER_PORT
    global POSITION_ACTIVE
    with open(filename+'.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)

        while(POSITION_ACTIVE):
            vx, vy = ETCPClient(POSITION_SERVER_ADDR, POSITION_SERVER_PORT)
            writer.writerow([vx, vy])

        time.sleep(1e-3)

def start_position(filename):
    loop = Thread(target=position_acquisition, name="Position_RX", args=(), kwargs={})
    loop.daemon = True
    loop.start()
