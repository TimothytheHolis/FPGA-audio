import numpy as np
import sine_wave_generator as swg
import matplotlib.pyplot as plt

def fft_recursive(x):
    x = np.asarray(x, dtype=complex) #creates an array of complex values
    N = x.shape[0] #number of elements in array

    if N == 1:
        return x #FFT of a single element is the element itself

    if N % 2 != 0:
        raise ValueError("Size of x must be a power of 2")

    even = fft_recursive(x[::2])
    odd  = fft_recursive(x[1::2])

    factor = np.exp(-2j * np.pi * np.arange(N) / N)

    return np.concatenate([
        even + factor[:N//2] * odd,
        even - factor[:N//2] * odd
    ])

sine_wave1 = swg.gen_audio(48000, 1000, 1.0, 0.8)  # 1 kHz sine wave
sine_wave2 = swg.gen_audio(48000, 5000, 1.0, 0.8)  # 5 kHz sine wave
mixed_signal = (sine_wave1 + sine_wave2) / 2
mixed_signal_f = mixed_signal.astype(np.float32) / 32768.0 #convert to float

N = 32768
mixed_signal_f = mixed_signal_f[:N]

X = fft_recursive(mixed_signal_f)
magnitude = np.abs(X[:N//2])  # only positive frequencies
fs = 48000
freqs = np.arange(N//2) * fs / N

plt.figure()
plt.plot(freqs, magnitude)
plt.title("FFT of Mixed 1 kHz and 5 kHz Sine Waves")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 8000)  # zoom into region of interest
plt.grid()
plt.show()

#apply a spectral window to reduce spectral leakage
window = np.hamming(N)
X = fft_recursive(mixed_signal_f * window)
magnitude = np.abs(X[:N//2])

plt.figure()
plt.plot(freqs, magnitude)
plt.title("FFT of Mixed 1 kHz and 5 kHz Sine Waves")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 8000)  # zoom into region of interest
plt.grid()
plt.show()