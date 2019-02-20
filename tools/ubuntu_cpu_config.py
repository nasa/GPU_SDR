import subprocess as sp
import sys
import os
import numpy as np
import time

def cpu_avg():
    N = 500
    x = []
    for i in range(N):
        x.append(get_cpu_freq())
        time.sleep(2./N)
    return np.mean(np.asarray(x),0)
def get_cpu_freq():
    rc = os.popen("cat /proc/cpuinfo | grep MHz").read().split("\t")[2::2]
    return np.asarray( [float(x.split("\n")[0][2:]) for x in rc if len(x)>1])

def scan_governor():
    rc = os.popen("cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor").read().split("\n")[:-1]
    rx_v = cpu_avg()
    for i in range(len(rc)):
        outstring = "Core %d is in %s mode"%(i,rc[i])
        if rc[i] == "powersave":
            outstring = "* "+outstring
        print outstring+"\t Freq: %.2f"%rx_v[i]
    avg = np.mean(rx_v)
    print "average frequency is %.2f"%avg
    return avg

def set_governor(gov):
    print "attmpting governor switch to "+str(gov)+" governor..."
    rc = os.popen("cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor").read().split("\n")[:-1]
    for i in range(len(rc)):
        change = os.popen("sudo sh -c \"echo -n "+str(gov)+" > /sys/devices/system/cpu/cpu"+str(int(i))+"/cpufreq/scaling_governor\"").read()
        if len(change)!=0:
            print "something went wrong:"
            print change
            return
    
    
if __name__ == "__main__":
    avg_i = scan_governor()
    set_governor("performance")# "performance" or "powersave"
    avg_f = scan_governor()
    
    print "frequency improved by %.2f percent"%(100*(avg_f-avg_i)/avg_i)

