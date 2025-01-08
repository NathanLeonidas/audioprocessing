import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os

import os

# Obtenir le répertoire parent du script
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construire le chemin vers le fichier
file_path = os.path.join(script_dir, 'audio_files', 'croisement.wav')

sample_rate, data = wav.read(file_path)

# Si le fichier est stéréo, convertir en mono
if len(data.shape) == 2:
    data = np.mean(data, axis=1)

# Paramètres pour l'analyse par fenêtres
frame_size = 320  # Taille d'une fenêtre pour la FFT
hop_size = int(sample_rate * 0.01)  # Décalage de 10 ms en échantillons
n_fft = 2**17  # Taille de la FFT

# Calculer la FFT pour chaque fenêtre
def compute_fft(data, frame_size, hop_size, sample_rate):
    num_frames = int((len(data) - frame_size) / hop_size) + 1
    time_axis = []  # Temps pour chaque fenêtre
    frequency_axis = np.fft.rfftfreq(n_fft, 1 / sample_rate)  # Fréquences positives
    spectrogram = []  # Amplitudes spectrales

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        windowed_data = data[start:end] * np.hanning(frame_size)  # Appliquer une fenêtre de Hanning
        fft_result = np.fft.fft(windowed_data, n_fft)  # FFT rapide
        fft_result = fft_result[:n_fft//2]
        spectrogram.append(np.abs(fft_result))
        time_axis.append(start / sample_rate)

    return np.array(spectrogram), time_axis, frequency_axis

# Appliquer la fonction de calcul
spectrogram, time_axis, frequency_axis = compute_fft(data, frame_size, hop_size, sample_rate)

# Détection des pics fréquentiels pour deux sinusoïdes
def detect_sinusoids(spectrogram, frequency_axis, threshold=0.1):
    peaks = []
    for frame in spectrogram:
        peak_indices = np.where(frame > threshold * np.max(frame))[0]  # Détection des pics
        peaks.append(frequency_axis[peak_indices])
    return peaks

# Détection des pics
peaks = detect_sinusoids(spectrogram, frequency_axis)

# Affichage du spectrogramme
plt.figure(figsize=(10, 6))
plt.imshow(
    10 * np.log10(spectrogram.T), 
    extent=[time_axis[0], time_axis[-1], frequency_axis[0], frequency_axis[-1]], 
    aspect='auto', origin='lower', cmap='viridis'
)
plt.colorbar(label="Amplitude (dB)")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Spectrogramme FFT")
plt.show()

# Affichage des pics détectés (facultatif)
for i, peak_freqs in enumerate(peaks):
    print(f"Frame {i}: Peaks at frequencies {peak_freqs} Hz")
