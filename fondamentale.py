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
Fmin = librosa.note_to_hz('C4')
Fmax = librosa.note_to_hz('C6')

# Fonction pour détecter la fréquence fondamentale avec FFT
def naive_fft_fundamental(signal, samplerate, window_size, hop_size):
    f0 = []
    times = []
    for start in range(0, len(signal) - window_size, hop_size):
        end = start + window_size
        window = signal[start:end]
        windowed_signal = window * np.hamming(len(window))  # Appliquer une fenêtre de Hamming
        spectrum = np.fft.fft(windowed_signal)[:len(windowed_signal)//2]  # Spectre
        frequencies = np.fft.fftfreq(len(windowed_signal), d=1/samplerate)[:len(windowed_signal)//2]
        magnitude = np.abs(spectrum)
        
        # Trouver le pic maximal dans les fréquences
        fundamental_freq = frequencies[np.argmax(magnitude)]
        f0.append(fundamental_freq)
        times.append(start / samplerate)
    
    return np.array(times), np.array(f0)

def autocorrelation_fundamental(signal, samplerate, window_size, hop_size, fmin, fmax):
    f0 = []
    times = []
    Tmin = int(samplerate / fmax)
    Tmax = int(samplerate / fmin)

    for start in range(0, len(signal) - window_size, hop_size):
        end = start + window_size
        window = signal[start:end]
        windowed_signal = window * np.hamming(len(window))  # Appliquer une fenêtre de Hamming

        # Calcul de l'autocorrélation
        autocorr = np.correlate(windowed_signal, windowed_signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Garder uniquement la seconde moitié

        # Ignorer les lags en dehors de la plage utile
        autocorr[:Tmin] = 0
        autocorr[Tmax:] = 0

        # Trouver le lag du premier maximum local
        lag = np.argmax(autocorr)

        # Calculer la fréquence fondamentale et vérifier qu'elle est dans la plage
        if lag > 0:
            fundamental_freq = samplerate / lag
            if fmin <= fundamental_freq <= fmax:
                f0.append(fundamental_freq)
            else:
                f0.append(0)  # Fréquence hors de la plage
        else:
            f0.append(0)  # Aucun pic détecté

        times.append(start / samplerate)
    
    return np.array(times), np.array(f0)


# Fonction pour calculer YIN (qui se base sur l'autocorrélation)
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



# Calcul avec chaque méthode:
#fft naive (max des pics de fréquence), autoccrélation, YIN, puis librairie python déjà existante
threshold = 0.6
times_fft, f0_fft = naive_fft_fundamental(x, samplerate, window_size, hop_size)
times_autocorr, f0_autocorr = autocorrelation_fundamental(x, samplerate, window_size, hop_size, Fmin, Fmax)
times_yin, f0_yin = yin(x, samplerate, window_size, hop_size, Fmin, Fmax, threshold)
f0_pyin, voiced_flag, voiced_probs = librosa.pyin(x, fmin=Fmin, fmax=Fmax, sr=samplerate, frame_length=window_size, hop_length=hop_size)


# Ajustement des dimensions pour correspondre
min_length = min(len(times_yin), len(f0_pyin), len(f0_fft), len(f0_autocorr))
times_yin = times_yin[:min_length]
f0_yin = f0_yin[:min_length]
f0_pyin = f0_pyin[:min_length]
f0_fft = f0_fft[:min_length]
times_autocorr = times_autocorr[:min_length]
f0_autocorr = f0_autocorr[:min_length]

# Affichage des résultats YIN
plt.figure(figsize=(10, 6))
plt.plot(times_yin, f0_yin, label="YIN", color="blue")
plt.plot(times_yin, f0_pyin, label='Librosa pyin', color='red', alpha=0.7)
plt.plot(times_fft, f0_fft, label='FFT naive', color='green')
plt.plot(times_autocorr, f0_autocorr, label='Autocorrélation', color='purple')
plt.title("Fréquence Fondamentale détectée avec YIN")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence fondamentale (Hz)")
plt.legend()
plt.tight_layout()
plt.show()
