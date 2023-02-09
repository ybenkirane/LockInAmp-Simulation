# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:07:22 2023

@author: Yacine Benkirane
"""

import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy import signal
import numpy as np

def sinSignal(t, amp, freq, phi):
    return amp*np.sin(2.0*np.pi * freq*t + phi)

def squareSignal(t, amp, freq, phi):
    return signal.square(2*np.pi * freq * t + phi)


phaseOffset = 0       #Phase offset in radians : should lead to a DC Offset
inputFreq = 157
refFreq = 143
inputV = 1
refV = 0.8

N = 1000000
T = 1.0/1600.0
t = np.linspace(0.0, N*T, N, endpoint=False)

# =============================================================================
# Note: Add a device under test (DUT) prior to the input signal. This should 
# simulate the output of the photodiode as a function of the drive signal. 
# =============================================================================
 
# =============================================================================
# Signal Multiplication 
# =============================================================================

Input = sinSignal(t, inputV, inputFreq, phaseOffset)
# Reference = squareSignal(t, refV, refFreq, 0)
Reference = sinSignal(t, refV, refFreq, 0)
MixedSignal = Input*Reference

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
plt.title("Time Domain Mixing")
plt.plot(t, Input, 'r--', label = f"Input {inputFreq} Hz")
plt.plot(t, Reference, 'g--', label = f"Reference {refFreq} Hz")
plt.plot(t, MixedSignal, 'b', label = "Product")
plt.xlabel("Time (sec)")
plt.ylabel("Voltage")
plt.xlim(0, 20/inputFreq)
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
ax.set_xlim(0, 1.5*(refFreq + inputFreq))
plt.xlabel("Freq (Hz)")
plt.ylabel("Voltage (V)")
plt.grid()
plt.legend()
plt.show()

print("\nInput: ", inputV, " V at ", inputFreq, 'Hz with a phase offset of ', phaseOffset, ' radians')
print("\nReference: ", refV, " V at ",  refFreq, " Hz")

print("\nMixed Freqencies: ", np.max(2.0/N * np.abs(yf_mix[0: N//2])), " V at ", abs(inputFreq + refFreq), 'Hz and ', abs(inputFreq - refFreq), " Hz")

# =============================================================================
# 
# =============================================================================

