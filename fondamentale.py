import soundfile as sf
import numpy as np
import cmath
import math
import matplotlib.pyplot as plt


data, samplerate = sf.read('C:\\Users\\Nathan\\Desktop\\CS\\METZ2A\\Traitement audio\\audioprocessing\\audio_files\\fluteircam.wav')
x=data
T = 1/samplerate

# Fonction pour calculer la FFT et le spectre
def compute_fft(signal, window_size):
    windowed_signal = signal * np.hamming(window_size)
    spectre = np.fft.fft(windowed_signal)
    spectre = spectre[:len(spectre)//2]  # Conserver uniquement les fréquences positives
    frequencies = np.fft.fftfreq(len(windowed_signal), T)[:len(spectre)//2]
    return spectre, frequencies

# Trouver naivement la fréquence fondamentale
def naive_fundamental_frequency(spectre, frequencies):
    magnitude = np.abs(spectre)
    index_max = np.argmax(magnitude)
    fundamental_frequency = frequencies[index_max]
    return fundamental_frequency




# Paramètres
window_size = 1024  # Taille de la fenêtre en échantillons
hop_size = 512  # Décalage entre les fenêtres (en échantillons)
fundamental_frequencies = []
x=data

for start in range(0, len(x) - window_size, hop_size):
    end = start + window_size
    window = x[start:end]
    
    # Calcul de la FFT et des fréquences
    spectre, frequencies = compute_fft(window, window_size)
    # Calcul de la fondamentale
    fundamental_frequency = naive_fundamental_frequency(spectre, frequencies)
    fundamental_frequencies.append(fundamental_frequency)

# Affichage de la fondamentale au fil du temps
times = np.arange(0, len(fundamental_frequencies)) * (hop_size / samplerate)

plt.figure(figsize=(10, 6))
plt.plot(times, fundamental_frequencies)
plt.title("Fréquence Fondamentale au Fil du Temps")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence fondamentale (Hz)")
plt.tight_layout()
plt.show()






# Paramètres
window_size = 2048  # Taille de la fenêtre en échantillons
hop_size = window_size//2  # Décalage entre les fenêtres (en échantillons)
H = 2  # Nombre de versions compressées pour le produit spectral
Fmin = 50  # Fréquence minimale de détection (Hz)
Fmax = 900  # Fréquence maximale de détection (Hz)


#produit spectral
def compute_spectral_product(spectre, H):
    P = np.ones_like(spectre)  # Initialisation de P avec la même forme que spectre
    # S'assurer que la longueur de spectre[h::H] est la même que celle de P
    for h in range(1, H + 1):
        compressed_spectre = np.abs(spectre[h::H])  # Compression du spectre
        # Redimensionner (ou tronquer) compressed_spectre pour qu'il ait la même longueur que P
        compressed_spectre_resized = np.resize(compressed_spectre, P.shape)
        P *= compressed_spectre_resized  # Multiplier avec la version redimensionnée
    return P

# Fonction pour détecter la fréquence fondamentale par produit spectral
def detect_f0(signal, samplerate, window_size, Fmin, Fmax, H):
    # Calcul de la FFT du signal
    spectre, frequencies = compute_fft(signal, window_size)
    
    # Calcul du produit spectral
    P = compute_spectral_product(spectre, H)
    
    # Conversion des fréquences min et max en indices
    Nmin = int(Fmin / (samplerate / window_size))
    Nmax = int(Fmax / (samplerate / window_size))
    
    # Trouver l'index correspondant à la fréquence fondamentale dans l'intervalle [Fmin, Fmax]
    f0_index = np.argmax(P[Nmin:Nmax])
    f0 = frequencies[Nmin + f0_index]
    
    return f0

# Estimation de la fréquence fondamentale pour chaque fenêtre
fundamental_frequencies = []
x = data

for start in range(0, len(x) - window_size, hop_size):
    end = start + window_size
    window = x[start:end]
    
    # Calcul de la fréquence fondamentale par produit spectral
    fundamental_frequency = detect_f0(window, samplerate, window_size, Fmin, Fmax, H)
    fundamental_frequencies.append(fundamental_frequency)

# Affichage de la fréquence fondamentale au fil du temps
times = np.arange(0, len(fundamental_frequencies)) * (hop_size / samplerate)

plt.figure(figsize=(10, 6))
plt.plot(times, fundamental_frequencies)
plt.title("Fréquence Fondamentale au Fil du Temps (Méthode du Produit Spectral)")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence fondamentale (Hz)")
plt.tight_layout()
plt.show()