import matplotlib.pyplot as pl
import numpy as np
from scipy import signal
import os

if __name__ == "__main__":
    Z = np.fromfile("USRP_polyphase_filter_window.dat", dtype=np.complex64)
    w, h = signal.freqz(Z, fs = 1e6, worN = int(1e5))
    pl.plot(w,np.abs(h))
    pl.figure()
    pl.plot(Z.real, label = "real part")
    pl.plot(Z.imag, label = "imag part")
    pl.legend()
    pl.show()
