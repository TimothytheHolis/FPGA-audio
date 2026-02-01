import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Parameters
fs = 48000          # Sample rate (Hz)
f = 1000            # Sine frequency (Hz)
duration = 1.0      # seconds
amplitude = 0.8     # 80% full-scale

# Time vector
t = np.arange(0, duration, 1/fs)

# Generate sine wave
x = amplitude * np.sin(2 * np.pi * f * t)

# Quantize to 16-bit signed PCM
x_int16 = np.int16(np.clip(x, -1.0, 1.0) * 32767)

# Save to WAV
sf.write("sine_1kHz.wav", x_int16, fs)

# Plot first few samples
plt.plot(x_int16[:200])
plt.title("1 kHz Sine Wave (16-bit PCM)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()