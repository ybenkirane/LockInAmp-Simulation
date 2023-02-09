# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:07:22 2023

@author: Yacine Benkirane
"""

import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import numpy as np

def signal(t, amp, freq, phi):
    return amp*np.sin(2.0*np.pi * freq*t + phi)



phaseOffset = 0         #Leads to a DC Offset
inputFreq = 100
refFreq = 100
inputV = 1
refV = 1

N = 1400
T = 1.0/800.0
t = np.linspace(0.0, N*T, N, endpoint=False)
 
# =============================================================================
# Signal Multiplication 
# =============================================================================

Input = signal(t, inputV, inputFreq, phaseOffset)
Reference = signal(t, refV, refFreq, 0)
MixedSignal = Input*Reference

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
plt.title("Time Domain Mixing")
plt.plot(t, Input, 'r--', label = f"Input {inputFreq} Hz")
plt.plot(t, Reference, 'g--', label = f"Reference {refFreq} Hz")
plt.plot(t, MixedSignal, 'b', label = "Product")
plt.xlabel("Time (sec)")
plt.ylabel("Voltage")
plt.grid()
plt.legend()
plt.show()


# =============================================================================
# FFT / Power Spectrum of Signals
# =============================================================================

yf_mix = fft(MixedSignal)
yf_ref = fft(Reference)
yf_input = fft(Input)
xf = fftfreq(N, T)[: N//2]


fig, ax = plt.subplots(1, 1, figsize=(16, 8))
plt.title("Frequency Domain Mixing")

ax.plot(xf, 2.0/N * np.abs(yf_mix[0: N//2]), 'b', label = "Mixed")
ax.plot(xf, 2.0/N * np.abs(yf_ref[0: N//2]), 'g', label = "Reference")
ax.plot(xf, 2.0/N * np.abs(yf_input[0: N//2]), 'r',  label = "Input")
#ax.set_xlim(0, 1000)
plt.grid()
plt.legend()
plt.show()

# =============================================================================
# 
# =============================================================================
