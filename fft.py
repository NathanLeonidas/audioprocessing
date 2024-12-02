import soundfile as sf
import numpy as np
import cmath
import matplotlib.pyplot as plt

data, samplerate = sf.read('fluteircam.wav')




def fft(x):
    # Nombre d'échantillons
    N = len(x)
    
    # Cas de base : si la taille du signal est 1, renvoyer le signal lui-même
    if N <= 1:
        return x
    
    # Découper le signal en deux (partie impaire et paire)
    even = fft(x[0::2])  # Composantes paires
    odd = fft(x[1::2])   # Composantes impaires
    
    # Calcul des racines de l'unité (utilisation de la formule de la FFT)
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    
    # Combinaison des résultats pour obtenir la FFT
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]


def fftfreq(n,t):
    return [i/(n*t) for i in range(n)]

# Exemple d'utilisation :
N = 16  # Nombre d'échantillons
T = 1.0 / 800.0  # Période d'échantillonnage
x = np.sin(2 * np.pi * 50 * np.arange(N) * T)  # Signal sinusoïdal à 50 Hz

# Appliquer la FFT sur le signal
X = fft(x)
frequencies = fftfreq(N,T)

# Affichage des résultats
plt.figure(figsize=(10, 6))

# Affichage du signal d'origine
plt.subplot(2, 1, 1)
plt.plot(np.arange(N) * T, x)
plt.title("Signal Original (sinusoïdal à 50 Hz)")

# Affichage de la magnitude de la FFT
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(X))  # On ne prend que les fréquences positives
plt.title("Magnitude de la FFT")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
