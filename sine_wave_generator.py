import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

"""
This function generate sine wave at specified frequency and quantizes it to 16-bit signed PCM format, and returns the output.
The output can be saved as a WAV audio file
Parameters:
- fs: Sample rate in Hz (e.g., 48000)
- f: Frequency of the sine wave in Hz (e.g., 1000)
- duration: Duration of the audio in seconds (e.g., 1.0)
- amplitude: Amplitude of the sine wave, typically between 0.0 and 1.0 (e.g., 0.8 for 80% full-scale)
"""

def gen_audio(fs, f, duration, amplitude):
    # Time vector
    t = np.arange(0, duration, 1/fs)

    # Generate sine wave
    x = amplitude * np.sin(2 * np.pi * f * t)

    # Quantize to 16-bit signed PCM
    x_int16 = np.int16(np.clip(x, -1.0, 1.0) * 32767)
    return x_int16

fs = 48000
f = 1000
duration = 1.0
amplitude = 0.8
audio = gen_audio(fs, f, duration, amplitude)

# Save to WAV
sf.write("sample_audio.wav", audio, fs)

# Plot first few samples
plt.plot(audio[:500])
plt.title("1 kHz Sine Wave (16-bit PCM)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()