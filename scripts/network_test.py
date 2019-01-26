import libUSRP2 as lu
import numpy as np
import time
rate =1e8
channels = 1000
RF = 300e6
tuning_mode = 'frac'
lu.print_debug( "Async thread launched, waiting for connection..")
if not lu.Connect():
    print "cannot find server!"
    exit()
lu.print_debug( "Connected, sending message...")


#N = int(raw_input("Ntones: "))
loop_freqs = [ 5e6 ]
#loop_freqs = [-50e6, -25e6, -10e6, -5e6, -1e6,50e6, 25e6, 10e6, 5e6, 1e6]#[-3e6+i*1e6 for i in range(N)]
N = len(loop_freqs)
samples = 160000e6#int(raw_input("eff Nsamples: "))*1e6
TX_ANT_NAME = 'B_RX2'
x = lu.global_parameter()
x.initialize()
x.set(TX_ANT_NAME,'mode','TX')
x.set(TX_ANT_NAME,'rate',25e6)
x.set(TX_ANT_NAME,'bw',2e8)
x.set(TX_ANT_NAME,'rf',RF)
x.set(TX_ANT_NAME,'buffer_len',1e6)
x.set(TX_ANT_NAME,'samples',samples)
x.set(TX_ANT_NAME,'tuning_mode',tuning_mode)

'''
x.set(TX_ANT_NAME,'chirp_t',[1])
x.set(TX_ANT_NAME,'swipe_s',[1230])
x.set(TX_ANT_NAME,'wave_type',["CHIRP"])
x.set(TX_ANT_NAME,'chirp_f',[7e6])
x.set(TX_ANT_NAME,'freq',[1e6])
x.set(TX_ANT_NAME,'ampl',[1.])
'''

x.set(TX_ANT_NAME,'fft_tones',1e8)
x.set(TX_ANT_NAME,'wave_type',["TONES" for i in range(N)])
x.set(TX_ANT_NAME,'ampl',[1./N for i in range(N)])
x.set(TX_ANT_NAME,'ampl',[1./2])
x.set(TX_ANT_NAME,'freq',loop_freqs)


x.set('A_RX2','mode','RX')
#x.set('A_RX2','buffer_len','1937000')

x.set('A_RX2','rate',rate)
x.set('A_RX2','bw',2e8)
x.set('A_RX2','rf',RF)
x.set('A_RX2','tuning_mode',tuning_mode)
x.set('A_RX2','samples',samples)
x.set('A_RX2','fft_tones',channels)
x.set('A_RX2','wave_type',["TONES" for i in range(N)])
x.set('A_RX2','pf_average',1)
#x.set('A_RX2','wave_type',["NOISE"])
#x.set('A_RX2','wave_type',["NODSP"])
x.set('A_RX2','ampl',[1./N for i in range(N)])
x.set('A_RX2','freq',loop_freqs)
x.set('A_RX2','decim',0)
x.set('A_RX2','buffer_len',1e6)
lu.print_debug( "Checking parameters... "+str(x.self_check()))
x.pprint()
if not lu.Async_send(x.to_json()):
    lu.Connect()
    lu.Async_send(x.to_json())
    
lu.Packets_to_file(parameters = x, timeout = None, filename = None, meas_tag = None, dpc_expected = samples/channels)
    #exit(0)
lu.Disconnect()
