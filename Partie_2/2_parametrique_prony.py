from scipy.linalg import toeplitz, solve
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import freqz
import os
from scipy.ndimage import gaussian_filter1d

# Charger le signal audio
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
file_path = os.path.join(script_dir, 'audio_files', 'croisement.wav')
y, sr = librosa.load(file_path, sr=None)

# Paramètres pour le découpage
frame_size = 2**10  # Taille des fenêtres
hop_size = int(sr * 0.01)  # Décalage entre les fenêtres (10 ms ici)
order =  2**9-1 # Ordre du modèle Prony
n_fft = 2**13  # Taille FFT pour PSD

# Découper le signal en fenêtres glissantes
frames = [y[i:i + frame_size] for i in range(0, len(y) - frame_size, hop_size)]

# Stockage des spectres
spectrogram = []

def prony_method(signal, order, epsilon=1e-6):
    """
    Implémente la méthode de Prony avec régularisation.
    """
    N = len(signal)
    if N <= order:
        raise ValueError("L'ordre du modèle doit être inférieur à la taille du signal.")

    # Calculer la corrélation auto
    R = np.correlate(signal, signal, mode='full')[N - 1 : N + order]
    T = toeplitz(R[:-1]) + epsilon * np.eye(order)  # Régularisation
    b = -R[1:]

    # Résolution du système linéaire
    ar_coeffs = solve(T, b)
    ar_coeffs = np.insert(ar_coeffs, 0, 1)  # Ajouter a0 = 1
    return ar_coeffs

i = 0
for frame in frames:
    i += 1
    if i % 10 == 0:
        print(f"Processing frame {i}...")
    # Estimation des coefficients AR avec la méthode de Prony
    try:
        ar_coeffs = prony_method(frame, order)
    except np.linalg.LinAlgError:
        print(f"Erreur dans la résolution du système pour une frame. Ignorée.")
        continue

    # Calcul du spectre de puissance
    _, psd = freqz(1, ar_coeffs, worN=n_fft, fs=sr)
    psd = np.abs(psd)**2  # Puissance spectrale

    # Normalisation (échelle cohérente avec dB)
    psd /= np.max(psd)  # Normaliser par le maximum global pour éviter des inversions

    # Conserver uniquement la moitié supérieure (fréquences positives)
    psd = psd[:n_fft // 2]
    spectrogram.append(psd)

# Convertir en tableau numpy pour l'affichage
spectrogram = np.array(spectrogram)

# Vérification : Affichage des valeurs brutes du spectrogramme
print("Valeurs brutes du spectrogramme (avant log):", spectrogram)

# Convertir les indices en fréquences (en Hz)
frequencies = np.linspace(0, sr / 2, n_fft // 2)  # Fréquences positives uniquement

# Temps pour chaque frame
time_axis = np.arange(len(spectrogram)) * hop_size / sr

# Affichage du spectrogramme paramétrique corrigé
plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(spectrogram.T + 1e-6), aspect='auto', 
           extent=[time_axis[0], time_axis[-1], frequencies[0], frequencies[-1]],
           origin='lower', cmap='jet')
plt.colorbar(label="Amplitude (dB)")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Spectrogramme paramétrique corrigé (Prony, échelle cohérente)")
plt.show()

# Détection des fréquences dominantes par lissage et maxima locaux avec continuité
def detect_frequencies(spectrogram, frequency_axis, smoothing_sigma=2, threshold_ratio=0.01):
    freq1 = []  # Liste pour la première sinusoïde
    freq2 = []  # Liste pour la deuxième sinusoïde
    amp1 = []   # Amplitude associée à freq1
    amp2 = []   # Amplitude associée à freq2

    prev_freq1, prev_freq2 = None, None

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
            dominant_amps = smoothed_frame[sorted_indices]

            # Assurer la continuité temporelle
            if prev_freq1 is not None and prev_freq2 is not None:
                # Calculer les distances aux fréquences précédentes
                dist1 = np.abs(dominant_freqs - prev_freq1)
                dist2 = np.abs(dominant_freqs - prev_freq2)

                # Associer la fréquence la plus proche
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
                # Premier échantillon
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
            # Si aucune fréquence significative n'est trouvée
            freq1.append(0)
            freq2.append(0)
            amp1.append(0)
            amp2.append(0)

    return freq1, freq2, amp1, amp2

# Détection des fréquences dominantes et de leurs amplitudes
freq1, freq2, amp1, amp2 = detect_frequencies(spectrogram, frequencies)

# Affichage des fréquences dominantes
plt.figure(figsize=(10, 6))
plt.plot(time_axis, freq1, label="Fréquence 1 (Hz)", color="blue", marker='x', linestyle='', markersize=2.5)
plt.plot(time_axis, freq2, label="Fréquence 2 (Hz)", color="red", marker='x', linestyle='', markersize=2.5)
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Fréquences des deux sinusoïdes en fonction du temps (avec décalage de 10ms)")
plt.legend()
plt.grid()
plt.show()

# Affichage des amplitudes associées
plt.figure(figsize=(10, 6))
plt.plot(time_axis, amp1, label="Amplitude Fréquence 1", color="blue")
plt.plot(time_axis, amp2, label="Amplitude Fréquence 2", color="red")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Amplitudes des deux sinusoïdes en fonction du temps")
plt.legend()
plt.grid()
plt.show()
