import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.ndimage import gaussian_filter1d
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
hop_size = int(sample_rate * 0.01)  # Décalage de 10 ms en échantillons
n_fft = 2**18 # Taille de la FFT

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

# Détection des fréquences dominantes par lissage et maxima locaux
def detect_frequencies(spectrogram, frequency_axis, smoothing_sigma=2, threshold_ratio=0.2):
    freq1 = []  # Liste pour la première sinusoïde
    freq2 = []  # Liste pour la deuxième sinusoïde
    
    for frame in spectrogram:
        # Lisser l'amplitude pour réduire le bruit
        smoothed_frame = gaussian_filter1d(frame, sigma=smoothing_sigma)

        # Trouver les maxima locaux
        derivative = np.diff(smoothed_frame)
        maxima_indices = np.where((derivative[:-1] > 0) & (derivative[1:] < 0))[0] + 1

        # Filtrer les maxima significatifs (au-dessus du seuil)
        max_amplitude = np.max(smoothed_frame)
        significant_indices = maxima_indices[smoothed_frame[maxima_indices] > threshold_ratio * max_amplitude]

        # Associer les fréquences dominantes
        if len(significant_indices) >= 2:
            # Trier les indices par ordre décroissant d'amplitude
            sorted_indices = significant_indices[np.argsort(smoothed_frame[significant_indices])[::-1]]
            dominant_freqs = frequency_axis[sorted_indices]
            
            # Assigner freq1 et freq2
            freq1.append(dominant_freqs[0])  # Plus grande amplitude
            freq2.append(dominant_freqs[1])  # Deuxième plus grande amplitude

        elif len(significant_indices) == 1:
            dominant_freq = frequency_axis[significant_indices[0]]
            freq1.append(dominant_freq)
            freq2.append(dominant_freq)
        else:
            # Si aucune fréquence significative n'est trouvée
            freq1.append(0)
            freq2.append(0)

    return freq1, freq2

# Détection des fréquences dominantes
freq1, freq2 = detect_frequencies(spectrogram, frequency_axis)

# Affichage des résultats
plt.figure(figsize=(10, 6))
plt.plot(time_axis, freq1, label="Fréquence 1 (Hz)", color="blue")
plt.plot(time_axis, freq2, label="Fréquence 2 (Hz)", color="red")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Fréquences des deux sinusoïdes en fonction du temps (avec décalage de 10ms)")
plt.legend()
plt.grid()
plt.show()

indice_min = 161
print('indice_min:', indice_min)
to_plot1 = gaussian_filter1d(spectrogram[indice_min], sigma=2)
to_plot2 = gaussian_filter1d(spectrogram[indice_min-1], sigma=2)
to_plot3 = gaussian_filter1d(spectrogram[indice_min+1], sigma=2)

plt.figure(figsize=(10, 6))
plt.plot(frequency_axis, to_plot1, label="Fréquence 1 (Hz)", color="blue")
plt.plot(frequency_axis, to_plot2, label="Fréquence 2 (Hz)", color="red")
plt.plot(frequency_axis, to_plot3, label="Fréquence 2 (Hz)", color="green")
plt.show()

print(freq2[indice_min], freq2[indice_min-1], freq2[indice_min+1])

