import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Chargement du fichier audio
data, samplerate = sf.read('C:\\Users\\Nathan\\Desktop\\CS\\METZ2A\\Traitement audio\\audioprocessing\\audio_files\\fluteircam.wav')
x = data
T = 1 / samplerate

# Paramètres
window_size = 2048  # Taille de la fenêtre en échantillons
hop_size = 512  # Décalage entre les fenêtres (en échantillons)
Fmin = librosa.note_to_hz('C3')
Fmax = librosa.note_to_hz('C8')

# Fonction pour calculer YIN
def yin(signal, samplerate, window_size, hop_size, fmin, fmax, threshold=0.1):
    def difference_function(x, W):
        """Calcul de la fonction de différence cumulée."""
        diff = np.zeros(W)
        for tau in range(W):
            diff[tau] = np.sum((x[:W - tau] - x[tau:W]) ** 2)
        return diff

    def cumulative_mean_normalized_difference(d):
        """Normalisation cumulative de la fonction de différence."""
        cmnd = np.zeros_like(d)
        cmnd[0] = 1  # Éviter les divisions par zéro
        running_sum = 0
        for tau in range(1, len(d)):
            running_sum += d[tau]
            cmnd[tau] = d[tau] / (running_sum / tau)
        return cmnd

    def find_fundamental_frequency(cmnd, samplerate, fmin, fmax, threshold):
        """Détecte la fréquence fondamentale à partir du minimum local."""
        Tmin = int(samplerate / fmax)
        Tmax = int(samplerate / fmin)
        for tau in range(Tmin, min(len(cmnd), Tmax)):
            if cmnd[tau] < threshold and cmnd[tau] < cmnd[tau - 1] and cmnd[tau] < cmnd[tau + 1]:
                return samplerate / tau  # Fréquence fondamentale
        return 0  # Retourne 0 si aucune fréquence fondamentale trouvée

    f0 = []
    times = []
    for start in range(0, len(signal) - window_size, hop_size):
        end = start + window_size
        window = signal[start:end]
        diff = difference_function(window, window_size)
        cmnd = cumulative_mean_normalized_difference(diff)
        freq = find_fundamental_frequency(cmnd, samplerate, fmin, fmax, threshold)
        f0.append(freq)
        times.append(start / samplerate)

    return np.array(times), np.array(f0)

# Estimer la fréquence fondamentale avec pyin
f0_pyin, voiced_flag, voiced_probs = librosa.pyin(x, fmin=Fmin, fmax=Fmax, sr=samplerate, frame_length=window_size, hop_length=hop_size)

# Paramètres/Calcul YIN
threshold = 0.4
times_yin, f0_yin = yin(x, samplerate, window_size, hop_size, Fmin, Fmax, threshold)

# Ajustement des dimensions pour correspondre
min_length = min(len(times_yin), len(f0_pyin))
times_yin = times_yin[:min_length]
f0_yin = f0_yin[:min_length]
f0_pyin = f0_pyin[:min_length]

# Affichage des résultats YIN
plt.figure(figsize=(10, 6))
plt.plot(times_yin, f0_yin, label="YIN", color="blue")
plt.plot(times_yin, f0_pyin, label='Librosa pyin', color='red', alpha=0.7)
plt.title("Fréquence Fondamentale détectée avec YIN")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence fondamentale (Hz)")
plt.legend()
plt.tight_layout()
plt.show()
