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

# Suivi des sinusoïdes par crêtes spectrales
def track_sinusoids(spectrogram, frequency_axis, threshold=0.1):
    freq1 = []  # Liste pour la première sinusoïde
    freq2 = []  # Liste pour la deuxième sinusoïde
    prev_freqs = None  # Fréquences détectées dans la trame précédente

    for frame in spectrogram:
        # Détecter les crêtes spectrales au-dessus du seuil
        peak_indices = np.where(frame > threshold * np.max(frame))[0]
        detected_freqs = frequency_axis[peak_indices]

        if prev_freqs is None:
            # Initialisation avec les deux premières fréquences détectées
            if len(detected_freqs) >= 2:
                freq1.append(detected_freqs[0])
                freq2.append(detected_freqs[1])
            else:
                freq1.append(0)
                freq2.append(0)
        else:
            # Associer les fréquences actuelles aux précédentes
            if len(detected_freqs) >= 2:
                closest_to_f1 = detected_freqs[np.argmin(np.abs(detected_freqs - freq1[-1]))]
                closest_to_f2 = detected_freqs[np.argmin(np.abs(detected_freqs - freq2[-1]))]

                # Mettre à jour les fréquences suivies
                freq1.append(min(closest_to_f1, closest_to_f2))
                freq2.append(max(closest_to_f1, closest_to_f2))
            else:
                # Si moins de deux fréquences sont détectées, on conserve les précédentes
                freq1.append(freq1[-1])
                freq2.append(freq2[-1])

        prev_freqs = detected_freqs

    return freq1, freq2

# Suivi des sinusoïdes
freq1, freq2 = track_sinusoids(spectrogram, frequency_axis)

# Affichage des résultats
plt.figure(figsize=(10, 6))
plt.plot(time_axis, freq1, label="Fréquence 1 (Hz)", color="blue")
plt.plot(time_axis, freq2, label="Fréquence 2 (Hz)", color="red")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Fréquences des deux sinusoïdes en fonction du temps (avec suivi amélioré)")
plt.legend()
plt.grid()
plt.show()

