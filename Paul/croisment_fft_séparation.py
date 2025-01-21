import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os

# Charger le fichier audio
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(script_dir, 'audio_files', 'croisement.wav')

sample_rate, data = wav.read(file_path)

# Si le fichier est stéréo, convertir en mono
if len(data.shape) == 2:
    data = np.mean(data, axis=1)

# Paramètres pour l'analyse par fenêtres
frame_size = 1024  # Taille d'une fenêtre pour la FFT
hop_size = 512     # Décalage entre deux fenêtres
n_fft = frame_size  # Taille de la FFT

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
        fft_result = np.fft.rfft(windowed_data, n_fft)  # FFT rapide
        spectrogram.append(np.abs(fft_result))
        time_axis.append(start / sample_rate)

    return np.array(spectrogram), time_axis, frequency_axis

# Appliquer la fonction de calcul
spectrogram, time_axis, frequency_axis = compute_fft(data, frame_size, hop_size, sample_rate)

# Suivi des sinusoïdes d'une trame à l'autre
def track_sinusoids(spectrogram, frequency_axis):
    freq1 = []  # Liste pour les premières sinusoïdes
    freq2 = []  # Liste pour les deuxièmes sinusoïdes

    # Initialisation avec la première trame
    prev_peaks = np.argsort(spectrogram[0])[-2:]  # Deux pics les plus élevés
    freq1.append(frequency_axis[min(prev_peaks)])
    freq2.append(frequency_axis[max(prev_peaks)])

    for frame in spectrogram[1:]:
        # Identifier les deux fréquences dominantes dans la trame actuelle
        current_peaks = np.argsort(frame)[-2:]
        current_freqs = frequency_axis[current_peaks]

        # Associer les fréquences actuelles aux précédentes
        f1, f2 = freq1[-1], freq2[-1]  # Fréquences précédentes
        closest_freq1 = current_freqs[np.argmin(np.abs(current_freqs - f1))]
        closest_freq2 = current_freqs[np.argmin(np.abs(current_freqs - f2))]

        # Ajouter les fréquences associées
        freq1.append(min(closest_freq1, closest_freq2))
        freq2.append(max(closest_freq1, closest_freq2))

    return freq1, freq2

# Suivi des sinusoïdes
freq1, freq2 = track_sinusoids(spectrogram, frequency_axis)

# Affichage des résultats
plt.figure(figsize=(10, 6))
plt.plot(time_axis, freq1, label="Fréquence 1 (Hz)", color="blue")
plt.plot(time_axis, freq2, label="Fréquence 2 (Hz)", color="red")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Fréquences des deux sinusoïdes en fonction du temps (avec suivi)")
plt.legend()
plt.grid()
plt.show()
