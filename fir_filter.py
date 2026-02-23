import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
from scipy.fft import rfft, rfftfreq
import sine_wave_generator as swg

fs = 48000          # sample rate
fc = 3000           # cutoff frequency (Hz)
num_taps = 101      # filter length (odd = linear phase)
cutoff_norm = fc / (fs / 2)

h = firwin(
    numtaps=num_taps,
    cutoff=cutoff_norm,
    window="hamming"
)

audio1 = swg.gen_audio(fs, 1000, 1.0, 0.8)  # 1 kHz sine wave
audio2 = swg.gen_audio(fs, 5000, 1.0, 0.8)  # 5 kHz sine wave
audio = (audio1 + audio2) / 2

# Save to WAV
sf.write("mixed_audio.wav", audio, fs)

# Plot first few samples
plt.plot(audio[:500])
plt.title("mixed 1 kHz and 5kHz Sine Wave (16-bit PCM)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()

# Convert to float for DSP math
audio_f = audio.astype(np.float32) / 32768.0

# Apply FIR filter
filtered = lfilter(h, 1.0, audio_f)

plt.plot(filtered[:500])
plt.title("Filtered output")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# FFT parameters
N = len(audio_f)

# Frequency axis
freqs = rfftfreq(N, 1/fs)

# FFT of original signal
fft_orig = np.abs(rfft(audio_f))

# FFT of filtered signal
fft_filt = np.abs(rfft(filtered))

plt.figure()
plt.plot(freqs, fft_orig)
plt.title("FFT of Original Mixed Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 8000)
plt.grid()
plt.show()

plt.figure()
plt.plot(freqs, fft_filt)
plt.title("FFT After FIR Low-Pass Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 8000)
plt.grid()
plt.show()