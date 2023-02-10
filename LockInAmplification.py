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


phaseOffset = 0     #Phase offset in radians : should lead to a DC Offset
inputFreq = 32
refFreq = 32
inputV = 1
refV = 1

N = 100000
T = 1.0/1000.0
t = np.linspace(-N*T, N*T, N, endpoint=False)

# =============================================================================
# Note: Add a device under test (DUT) prior to the input signal. This should 
# simulate the output of the photodiode as a function of the drive signal. 
# =============================================================================
 
# =============================================================================
# Signal Multiplication 
# =============================================================================

Input = sinSignal(t, inputV, inputFreq, phaseOffset)    #Assuming input has no DC Component
# Reference = squareSignal(t, refV, refFreq, 0)
Reference = sinSignal(t, refV, refFreq, 0)              #Assuming input has no DC Component
MixedSignal = Input*Reference

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
plt.title("Time Domain Mixing")
plt.plot(t, Input, 'r--', label = f"Input {inputFreq} Hz")
plt.plot(t, Reference, 'g--', label = f"Reference {refFreq} Hz")
plt.plot(t, MixedSignal, 'b', label = "Product")
plt.xlabel("Time (sec)")
plt.ylabel("Voltage")
plt.xlim(-6/inputFreq, 6/inputFreq)
plt.grid()
plt.legend(loc = "upper right")
plt.show()

print("Mixed Signal Amplitude: ", round(np.abs(np.max(MixedSignal) - np.min(MixedSignal))/2, 6))
print("Mixed Signal DC Offset: ", round(np.average(MixedSignal), 6), " V")

# print(np.abs(np.max(MixedSignal) - np.min(MixedSignal))/2 + np.average(MixedSignal))


# =============================================================================
# FFT / Power Spectrum of Signals
# -----------------------------------------------------------------------------
# Note: When f1 = f2, we have a freq at 2f and one at 0 Hz. The latter is the 
# DC Offset. 
# =============================================================================

yf_mix = fft(MixedSignal)
yf_ref = fft(Reference)
yf_input = fft(Input)
xf = fftfreq(N, 2*T)[0: N//2]


fig, ax = plt.subplots(1, 1, figsize=(16, 8))
plt.title("Frequency Domain Mixing")

# xftest = fftfreq(50, 0.2)[0: 50//2]
# print("\nxf: ", xftest)

ax.plot(xf, 2.0/N * np.abs(yf_mix[: N//2]), 'b', label = "Mixed")
ax.plot(xf, 2.0/N * np.abs(yf_input[: N//2]), 'r',  label = "Input")
ax.plot(xf, 2.0/N * np.abs(yf_ref[: N//2]), 'g', label = "Reference")
ax.set_xlim(-1, 1.5*(refFreq + inputFreq))
plt.xlabel("Freq (Hz)")
plt.ylabel("|Voltage| (V)")
plt.grid()
plt.legend(loc = "upper right")
plt.show()

# print(yf_mix)

print("\nInput: ", inputV, " V at ", inputFreq, 'Hz with a phase offset of ', round(phaseOffset, 6), ' radians')
print("Reference: ", refV, " V at ",  refFreq, " Hz")

print("\nMixed Freqencies: ", round(np.max(2.0/N * np.abs(yf_mix[0: int(N*T*max(inputFreq, refFreq))])), 6), " V at ", round(abs(inputFreq - refFreq), 6), 'Hz and ', round(np.max(2.0/N * np.abs(yf_mix[int(N*T*max(inputFreq, refFreq)): N//2])), 6)," V at ", abs(inputFreq + refFreq), " Hz")
print(int(N*T*max(inputFreq, refFreq)))
print(len(yf_mix))

print(np.max(2.0/N * np.abs(yf_mix[0: int(N*T*max(inputFreq, refFreq))])))
# =============================================================================
# Low-Pass Filter : Filter out the 2*inputFreq signal and only leave the 0 Hz
# DC Output... 
# =============================================================================

