import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import find_peaks

# Charger le fichier audio avec soundfile
file_path = 'C:\\Users\\Nathan\\Desktop\\CS\\METZ2A\\Traitement audio\\audioprocessing\\audio_files\\croisement4.wav'  # Remplacez par le chemin de votre fichier
data, sample_rate = sf.read(file_path)

# Vérifier si le fichier est mono ou stéréo
if len(data.shape) > 1:
    data = data[:, 0]  # Prendre un seul canal si stéréo

# Normaliser le signal
data = data / np.max(np.abs(data))

# Paramètres STFT
n_fft = 2048  # Taille de la fenêtre FFT
hop_length = n_fft // 2  # Chevauchement
window = np.hanning(n_fft)  # Fenêtre de Hanning

# Calculer la STFT
frames = range(0, len(data) - n_fft, hop_length)
stft = np.array([np.fft.rfft(window * data[i:i + n_fft]) for i in frames])

# Magnitude spectrale
magnitude = np.abs(stft)

# Fréquences et temps associées
frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
times = np.arange(len(frames)) * hop_length / sample_rate

# Détection des fréquences dominantes
dominant_frequencies = []
for frame in magnitude:
    peaks, _ = find_peaks(frame, height=np.max(frame) * 0.1)  # Pics locaux significatifs
    dominant_frequencies.append(frequencies[peaks])

# Visualiser les fréquences dominantes au fil du temps
plt.figure(figsize=(12, 6))
plt.imshow(10 * np.log10(magnitude.T), origin="lower", aspect="auto",
           extent=[times.min(), times.max(), frequencies.min(), frequencies.max()],
           cmap="viridis")
plt.colorbar(label="Amplitude (dB)")
plt.title("Spectrogramme avec fréquences dominantes")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")

# Ajouter les fréquences dominantes
for t, freqs in zip(times, dominant_frequencies):
    plt.scatter([t] * len(freqs), freqs, color="red", s=10, label="Fréquences dominantes" if t == times[0] else "")

plt.legend()
plt.show()
