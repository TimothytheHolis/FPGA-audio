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