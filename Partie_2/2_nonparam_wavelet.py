import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
from scipy.ndimage import gaussian_filter1d

# Charger le fichier audio
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(script_dir, 'audio_files', 'croisement.wav')
y, sr = librosa.load(file_path, sr=None)

# Définir les échelles des ondelettes
scales = np.exp(np.linspace(np.log(10**0.1), np.log(10**2.5), num=1000))


# Définir explicitement l'ondelette Complex Morlet avec un facteur de qualité
wavelet = 'cmor2.5-3.0'  # Facteurs ajustables : cmorB-C où B=1.5 (temps), C=1.0 (fréquence)

# Appliquer la CWT
coefficients, frequencies = pywt.cwt(y, scales, wavelet, sampling_period=1/sr)

# Correction des fréquences (conversion échelle -> Hz)
true_frequencies = sr / (scales * 4.0)

# Affichage du scalogramme
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(coefficients), aspect='auto', 
           extent=[0, len(y)/sr, true_frequencies[-1], true_frequencies[0]], 
           cmap='jet', interpolation='nearest')
plt.colorbar(label="Amplitude")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Scalogramme (Wavelet Transform)")
plt.ylim([0, sr / 5])  # Limiter l'affichage à la moitié de la fréquence d'échantillonnage
plt.show()


# Détection des fréquences dominantes par lissage et maxima locaux avec continuité
def detect_frequencies(spectrogram, frequency_axis, smoothing_sigma=2, threshold_ratio=0.1):
    freq1, freq2, amp1, amp2 = [], [], [], []
    prev_freq1, prev_freq2 = None, None

    for frame in spectrogram.T:  # Transposer pour parcourir le spectrogramme dans le bon sens
        # Lisser l'amplitude pour réduire le bruit
        smoothed_frame = gaussian_filter1d(frame, sigma=smoothing_sigma)

        # Trouver les maxima locaux
        derivative = np.diff(smoothed_frame)
        maxima_indices = np.where((derivative[:-1] > 0) & (derivative[1:] < 0))[0] + 1

        # Filtrer les maxima significatifs
        max_amplitude = np.max(smoothed_frame)
        significant_indices = maxima_indices[smoothed_frame[maxima_indices] > threshold_ratio * max_amplitude]

        # Sélection des deux fréquences dominantes
        if len(significant_indices) >= 2:
            sorted_indices = significant_indices[np.argsort(smoothed_frame[significant_indices])[::-1]]
            dominant_freqs = frequency_axis[sorted_indices]
            dominant_amps = smoothed_frame[sorted_indices]

            # Gestion de la continuité temporelle
            if prev_freq1 is not None and prev_freq2 is not None:
                dist1 = np.abs(dominant_freqs - prev_freq1)
                dist2 = np.abs(dominant_freqs - prev_freq2)

                if dist1[0] < dist1[1]:
                    freq1.append(dominant_freqs[0])
                    freq2.append(dominant_freqs[1])
                    amp1.append(dominant_amps[0])
                    amp2.append(dominant_amps[1])
                else:
                    freq1.append(dominant_freqs[1])
                    freq2.append(dominant_freqs[0])
                    amp1.append(dominant_amps[1])
                    amp2.append(dominant_amps[0])
            else:
                freq1.append(dominant_freqs[0])
                freq2.append(dominant_freqs[1])
                amp1.append(dominant_amps[0])
                amp2.append(dominant_amps[1])

            prev_freq1, prev_freq2 = freq1[-1], freq2[-1]

        elif len(significant_indices) == 1:
            dominant_freq = frequency_axis[significant_indices[0]]
            dominant_amp = smoothed_frame[significant_indices[0]]
            freq1.append(dominant_freq)
            freq2.append(dominant_freq)
            amp1.append(dominant_amp)
            amp2.append(dominant_amp)
            prev_freq1, prev_freq2 = dominant_freq, dominant_freq

        else:
            freq1.append(0)
            freq2.append(0)
            amp1.append(0)
            amp2.append(0)

    return freq1, freq2, amp1, amp2

# Appliquer la détection des fréquences dominantes
freq1, freq2, amp1, amp2 = detect_frequencies(np.abs(coefficients), true_frequencies)

# Créer l'axe du temps
time_axis = np.linspace(0, len(y)/sr, num=coefficients.shape[1])

# Affichage des fréquences dominantes
plt.figure(figsize=(10, 6))
plt.scatter(time_axis, freq1, label="Fréquence 1 (Hz)", color="blue", marker='x', s=2.5)
plt.scatter(time_axis, freq2, label="Fréquence 2 (Hz)", color="red", marker='x', s=2.5)
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Fréquences dominantes détectées")
plt.legend()
plt.grid()
plt.show()

