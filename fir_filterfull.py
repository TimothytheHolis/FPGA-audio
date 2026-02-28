import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sine_wave_generator as swg
import fft as fftmod  # your module with fft_recursive

# ===============================
# 1️⃣ Parameters
# ===============================
fs = 48000        # sampling frequency
fc = 3000         # cutoff frequency (Hz)
num_taps = 101    # filter length (odd)
duration = 1.0    # seconds

# ===============================
# 2️⃣ Create FIR filter from scratch (sinc + Hamming)
# ===============================
M = num_taps - 1
fc_norm = fc / (fs / 2)  # normalize to Nyquist

n = np.arange(num_taps)
# Ideal sinc filter (low-pass)
h_ideal = np.sinc(2 * fc_norm * (n - M/2))
# Apply Hamming window
window = np.hamming(num_taps)
h = h_ideal * window
# Normalize for unity gain at DC
h /= np.sum(h)

# ===============================
# 3️⃣ Generate mixed sine wave signals
# ===============================
audio1 = swg.gen_audio(fs, 1000, duration, 0.8)  # 1 kHz
audio2 = swg.gen_audio(fs, 5000, duration, 0.8)  # 5 kHz
mixed_signal = (audio1 + audio2) / 2
# Convert to float [-1,1]
mixed_signal_f = mixed_signal.astype(np.float32) / 32768.0

# Save WAV
sf.write("mixed_audio_from_scratch.wav", mixed_signal, fs)

# ===============================
# 4️⃣ Apply FIR filter (convolution)
# ===============================
filtered_signal = np.convolve(mixed_signal_f, h, mode='same')

# ===============================
# 5️⃣ FFT using your fft_fromfft module
# ===============================
# Pick power-of-2 length
N_fft = 32768
x_fft = mixed_signal_f[:N_fft]
y_fft = filtered_signal[:N_fft]

# FFT computation
X = fftmod.fft_recursive(x_fft)
Y = fftmod.fft_recursive(y_fft)

# Magnitude (first half, positive frequencies)
mag_X = np.abs(X[:N_fft//2])
mag_Y = np.abs(Y[:N_fft//2])

# Frequency axis
freqs = np.arange(N_fft//2) * fs / N_fft

# ===============================
# 6️⃣ Plot time-domain signals
# ===============================
plt.figure(figsize=(10,4))
plt.plot(mixed_signal_f[:500], label="Original Mixed Signal")
plt.plot(filtered_signal[:500], label="Filtered Signal")
plt.title("Time Domain Signals (First 500 Samples)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# ===============================
# 7️⃣ Plot FFTs
# ===============================
plt.figure(figsize=(10,4))
plt.plot(freqs, mag_X, label="Original FFT")
plt.plot(freqs, mag_Y, label="Filtered FFT")
plt.title("FFT Before and After FIR Filtering")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 8000)
plt.grid()
plt.legend()
plt.show()