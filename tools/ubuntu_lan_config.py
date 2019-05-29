import subprocess as sp
import sys
import os,time
import argparse
from PyInquirer import prompt, print_json
from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt, Separator
help_message_pci = "Help section.\nIf you are reading this message the automatic PCIe tuning for the NIC failed. The automatic process has only been tested on Ubuntu 18.04 with a SFP+ intel NIC.\nThe process is described at: http://dak1n1.com/blog/7-performance-tuning-intel-10gbe/\n"


help_message_pci = "Here\'s a partial copy:\nThe first thing you\'"
help_message_pci += "ll need to do is find the PCI address, as shown by lspci:\n\t[test ~]$ "
help_message_pci += "lspci\n\t07:00.0 Ethernet controller: Intel Corporation 82599EB 10-Gigabit SFI/SFP+ Network Connection (rev 01)\n"
help_message_pci += "Here 07.00.0 is the pci bus address. Now we can grep for that in /proc/bus/pci/devices to gather even more information:\n\t[test ~]$ grep 0700 /proc/bus/pci/devices\n\t0700\t808610fb\t28\td590000c (...)\nVarious information about the PCI device will display, as you can see above."
help_message_pci += " But the number we\'re interested in is the second field, 808610fb. This is the Vendor ID and Device ID together. Vendor: 8086 Device: 10fb. You can use these values to tune the PCI bus MMRBC, or Maximum Memory Read Byte Count. This will increase the MMRBC to 4k reads, increasing the transmit burst lengths on the bus.\n\t[test ~] setpci -v -d 8086:10fb e6.b=2e"

search_key = "SFP+" #this is a search keyword to find the ethernet adapter
prefix="192.168."
final = ".1"
MTU = 9000
def config(adapter_name, domains):
    global final
    global prefix
    if adapter_name == None:
        print "First argument should be the name of the network adapter."
        return adapter_name
    print "Running config on "+adapter_name
    for d in domains:
        print "Setting port speed... (for multi speed addapters only)"
        out_string = "ethtool -s "+adapter_name+" speed 10000 autoneg off"
        os.system(out_string)
        time.sleep(1)
        current_address = prefix + str(d) + final
        usrp_addr = prefix + str(d) + "."+str(int(2))
        out_string = "ifconfig "+adapter_name+" "+current_address
        print "Changing address to " + current_address
        os.system(out_string)
        child = sp.Popen("/usr/local/bin/uhd_find_devices --args=\"address="+usrp_addr+"\"", stdout=sp.PIPE,shell=True)
        streamdata = child.communicate()[0]
        rc = child.returncode
        if rc == 0:
            print "USRP found at " + current_address
            print "Setting network MTU..."
            out_string = "ifconfig "+adapter_name+" mtu "+str(MTU)
            os.system(out_string)
            print "Setting number of descriptors in the NIC..."
            out_string = "sudo ethtool -G "+ adapter_name + " rx 4096 tx 4096"
            os.system(out_string)

            print "Setting network buffers..."
            out_string = "sysctl -w net.core.rmem_max=1621498630"
            os.system(out_string)
            out_string = "sysctl -w net.core.wmem_max=1621498630"
            os.system(out_string)
            print "Configurating the pcie bus..."
            bus_address = os.popen('lspci | grep '+search_key).read().split(' ')[0].split('.')[0].replace(":","")
            if len(bus_address)<4:
                print "Cannot automatically determine the PCIe address of the NIC."
                print help_message_pci
                exit()
            else:
                print "NIC PCIe bus address: "+bus_address
            f = os.popen("grep "+bus_address+" /proc/bus/pci/devices")
            ID_device = f.read().split("\t")[1]
            if ID_device<8:
                print "cannot get the NIC vendor and ID."
                print help_message_pci
            os.system("sudo setpci -v -d "+ID_device[0:4]+":"+ID_device[4:]+" e6.b=2e")
            return "you are all set."

    return "No USRP found. Maybe it is connected to an other network adapter?"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--adapter','-a', help='Name of the adapter. Optional argument, if not given a menu will pop-up.',type = str)
    parser.add_argument('--subnet','-s',help='subnet numbers to use in the address scan in the form 192.168.xx.1', nargs='+', type=int)

    args = parser.parse_args()
    if os.getuid()!= 0:
        print "Script must have admin privilege. Try running with sudo."
        exit(-1)
    if args.adapter is None:
        selected_adapter_names = []
        adp = os.popen("ls /sys/class/net")
        ad_names = adp.read().split("\n")
        for nn in ad_names:
            if len(nn)>2:
                selected_adapter_names.append(nn)

        questions = [
            {
                'type': 'list',
                'name': 'adapters',
                'message': 'No adapter specified via system arguments. Pick the adapter to probe in the menu:',
                'choices': selected_adapter_names+[
                    Separator(),
                    'Exit'
                ]
            }
        ]

        adapter_name = prompt(questions)['adapters']
        if adapter_name == "Exit":
            exit()

    else:
        adapter_name = args.adapter


    print "Looking for usrp on adapter: "+adapter_name

    if args.subnet is None:
        print 'No subnet number specified in system arguments. Will changing the ip address to 192.168.xx.1\nDefault is [30,40]'
        questions = [
            {
                'type': 'checkbox',
                'name': 'subnet',
                'message': 'Select subnet range(s) (space to select, enter to confirm)',
                'choices': [
                    {
                        'name': '10'
                    },
                    {
                        'name': '20'
                    },
                    {
                        'name': '30'
                    },
                    {
                        'name': '40'
                    },
                    {
                        'name': '50'
                    },
                    {
                        'name': 'Custom'
                    },
                    Separator(),
                    {
                        'name': 'Exit'
                    },
                    {
                        'name': 'All'
                    },
                ]
            }
        ]
        dms = prompt(questions)['subnet']
        domains = []
        for nn in dms:
            if nn == 'Exit':
                exit()
            if nn == 'Custom':
                valid_entry = False
                while not valid_entry:
                    try:
                        domains.append(int(raw_input("Custom subnet: ")))
                        valid_entry = True
                    except ValueError:
                        valid_entry = False
            try:
                domains.append(int(nn))
            except ValueError:
                pass
            if nn == 'All':
                domains = [10,20,30,40,50]
                break
        if len(domains)==0:
            print "Selecting default domains."
            domains = [30,40]

    else:
        domains = args.subnet
    print  "Scanning on domains: "+str(domains)

    print config(adapter_name, domains)
