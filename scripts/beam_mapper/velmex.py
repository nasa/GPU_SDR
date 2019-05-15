############################################################
##            STEP MOTOR CONTROLLER LIBRARY               ##
## ORIGINALLY TAKEN FROM GITHUB MODIFIED FOR THE VELMAX-2 ##
############################################################

'''
STEP MOTOR CONTROLLER LIBRARY
=============================

Assuming that you are controlling the device with a usb to serial port adapter that is the only usb connected to the machine. The library has been modified to do beam mapping.

WARNING: this library assumes that you have already checked the functionality and the position of the limit switch. This library USES the limit switches for calibration purpose only. There is no limit switch safety feature.


In order to make it work there is a manual procedure for fixing the 'home' of the hardware.

Make the motor hit the limits and than use the set home function.


Usage
=====

Scanning
--------

v.start_thread() is a nonblocking call that sets the control loop. also initialize the position of the motors to home position.

v.start_scan() will start the scan. still non blocking.

v.stop() will kill the thread. it will block untill the velmax stops moving.

Just moving
-----------

v.init() will initialize the library. Functions like moveFor() will control the motors. note that the motors available are only 1 == x direction and 2 == y direction.


WARNING: if for some reason the yellow led on the velmax blinks fast it ha to be power cycled before it can be controlled again. This happens when the library is initialized while the motors are already moving.


'''

import serial,time,csv
import threading

ser = serial.Serial()
motors = ['X','Y','Z','T' ]
LIMIT_HIGH_X = 118347
LIMIT_HIGH_Y = 109928
LIMIT_LOW_X = 0
LIMIT_LOW_Y = 0
CONV= 0.00025 # inch/step
thread_scan = None
start_scanning = False
positioning = False

def init():
    global ser
    #catch a SerialException and check for return code
    ser = serial.Serial("/dev/ttyUSB0", 9600, bytesize=serial.EIGHTBITS , parity= serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=.1)
    ser.write("F")  # enable online mode
    ser.write("C")  # clear current program


def moveFor (motor, distance):
    global ser
    ser.write("C")  # clear current program
    ser.write("I" + str(motor) + "M" + str(distance) + ",") # send movement command
    ser.write("R")  # run current program
    return

def moveTo (motor, destination):
    global ser
    ser.write("C")  # clear current program
    ser.write("IA" + str(motor) + "M" + str(destination) + ",") # send movement command
    ser.write("R")  # run current program
    return;

# send all motors to their zero positions
def homeAll():
#    ser.write("C")  # clear current program
#    for i in range (1,5):
#        ser.write("IA" + str(i) + "M" + str(0) + ",") # send movement commands
#    ser.write("R")  # run current program
    moveHome(2)
    wait()
    moveHome(1)
    return

# sets registers to zero at current position
def setHome():
    global ser
    ser.write("N")
    return

# returns the position of motor m
def getPos (m):
    global ser
    wait()                   # wait for motors to finish moving
    u = ser.readline();          # clear current buffer
    ser.write(motors[m-1])   # query for position
    pos = ser.readline()
    sign=1                   # extract sign
    if pos[0]=='-':
        sign=-1
    pos = int(''.join([x for x in pos if x.isdigit()])) # parseint
    return sign*pos


# send motor to positive limit switch
def moveHome(motor):
    ser.write("C")
    ser.write("I"+str(motor)+"M-0,")
    ser.write("R")

def moveMax(motor):
    ser.write("C")
    ser.write("I"+str(motor)+"M0,")
    ser.write("R")

# run all the instruction given or repeat
def go():
    ser.write("R")

def stop():
    ser.write("D")-10200

# delay until current program is done
def wait():
    global ser
    ser.readline()                  # clear current buffer
    ser.write('V')                  # query for velmex's status
    busy_sig = ser.readline()
    while (busy_sig=="B"):    # if busy,
        ser.write("V")
        busy_sig = ser.readline()
        time.sleep(0.1)

def kill_loop():
    global ser
    ser.write('K')      # kill the current loop

def end():
    global ser
    ser.write('Q')      # quit online mode
    ser.close()         # close serial port


stop_thread = False
def scan_thread(nx, ny, timer, pre_timer, name):

    '''
    Scanning thread

    it's controlled via the functions start() and stop()
    '''

    global LIMIT_HIGH_Y, LIMIT_HIGH_X, LIMIT_LOW_Y, LIMIT_LOW_X, ser, stop_thread
    global start_scanning, positioning
    init()
    step_size_x = (LIMIT_HIGH_X - LIMIT_LOW_X)/(nx-1)
    step_size_y = (LIMIT_HIGH_Y - LIMIT_LOW_Y)/(ny-1)
    positioning = True
    homeAll()
    wait()
    positioning = False
    versor = +1
    with open(name+'.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["start_time(epoch[s])","end_time(epoch[s])","x_pos(inch)","y_pos(inch)"])
        print "waining scan signal"
        while not(start_scanning):
            time.sleep(0.1)

            if stop_thread:
                break
        print "waiting pretimer"
        time.sleep(pre_timer)
        for i in range(0, nx):
            if stop_thread:
                break

            if versor>0:
                y_range = range(ny)
            else:
                y_range = reversed(range(ny))

            #scan line
            do_move = False
            for j in y_range:
                if stop_thread:
                    break
                else:
                    print "Current position X: %d Y: %d"%(i,j)
                if do_move:
                    init()
                    moveFor (2, versor*step_size_y)
                do_move = True
                wait()
                t_start = time.time()
                time.sleep(timer)
                writer.writerow([t_start,time.time(),i*CONV*step_size_x ,j*CONV*step_size_y])

            if stop_thread:
                break

            #move row
            init()
            moveFor (1, step_size_x)
#            wait()


            #change y direction
            versor *= -1


def start_thread(nx, ny, timer, pre_timer, name):
    global stop_thread,thread_scan,start_scanning,positioning
    positioning = True
    thread_scan = threading.Thread(target=scan_thread, args=(nx, ny, timer, pre_timer, name,))
    thread_scan.daemon = True
    start_scanning = False
    thread_scan.start()

def wait_for_positioning():
    global positioning
    print "waiting for motor position 0,0 ..."
    while(positioning):
        time.sleep(0.1)

def start_scan():
    global positioning, start_scanning
    while positioning:
        time.sleep(1)
    start_scanning = True

def stop():

    global stop_thread,thread_scan
    stop_thread = True
    thread_scan.join()
    stop_thread = False
    print "Thread stoped"

def calibrate():
    '''
    Overwite the original calibration.
    '''
    global LIMIT_HIGH_Y, LIMIT_HIGH_X, LIMIT_LOW_Y, LIMIT_LOW_X

    # calibrate origin position

    print "Getting minimum X"
    moveHome(1)
    time.sleep(1)
    wait()
    LIMIT_LOW_X = getPos(1)

    print "Getting minimum Y"
    moveHome(2)
    time.sleep(1)
    wait()
    LIMIT_LOW_Y = getPos(2)


    print "Getting maximum X"
    moveMax(1)
    time.sleep(1)
    wait()
    LIMIT_HIGH_X = getPos(1)

    print "Getting maximum Y"
    moveMax(2)
    time.sleep(1)
    wait()
    LIMIT_HIGH_Y = getPos(2)


    print "LIMIT_HIGH_Y = %d steps"%LIMIT_HIGH_Y
    print "LIMIT_HIGH_X = %d steps"%LIMIT_HIGH_X
    print "LIMIT_LOW_Y = %d steps"%LIMIT_LOW_Y
    print "LIMIT_LOW_X = %d steps"%LIMIT_LOW_X

def center():
    '''
    Go to the center of the XY plane.
    '''
    #global LIMIT_HIGH_Y, LIMIT_HIGH_X, LIMIT_LOW_Y, LIMIT_LOW_X

    target_y = (LIMIT_HIGH_Y - LIMIT_LOW_Y)/2


    target_x = (LIMIT_HIGH_X - LIMIT_LOW_X)/2

    #move x
    print "Centering X..."
    current_x = getPos(1)

    print "current %d"%current_x
    print "target %d"%target_x

    movement = target_x - current_x
    print "Moving %d step"%movement
    if movement != 0:
        moveFor (1, movement)
        time.sleep(0.5)
        wait()
    else:
        print "already centered"

    #move y
    print "Centering Y..."
    current_y = getPos(2)
    print "current %d"%current_y
    print "target %d"%target_y
    movement = target_y - current_y
    if movement != 0:
        print "Moving %d step"%movement
        moveFor (2, movement)
        wait()
    else:
        print "already centered"
