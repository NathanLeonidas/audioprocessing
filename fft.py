import soundfile as sf
import numpy as np
import cmath
import math
import matplotlib.pyplot as plt

data, samplerate = sf.read('C:\\Users\\Nathan\\Desktop\\CS\\METZ2A\\Traitement audio\\audioprocessing\\audio_files\\fluteircam.wav')

def pad2n(x):
    N = len(x)
    L = math.floor(math.log(N,2)+1)
    x = np.pad(x, (0,2**L-N), 'constant', constant_values=0)
    return x



def fft_padded(x):
    # Nombre d'échantillons
    N = len(x)
    
    # Cas de base : si la taille du signal est 1, renvoyer le signal lui-même
    if N <= 1:
        return x
    
    # Découper le signal en deux (partie impaire et paire)
    even = fft_padded(x[0::2])  # Composantes paires
    odd = fft_padded(x[1::2])   # Composantes impaires
    
    # Calcul des racines de l'unité (utilisation de la formule de la FFT)
    T = [cmath.exp(-2j*np.pi*i/N)*odd[i] for i in range(N//2)]

    return [even[i]+T[i] for i in range(N//2)] + [even[i]-T[i] for i in range(N//2)]


def fftfreq(n,t):
    return [i/(n*t) for i in range(n//2)]+[i/(n*t) for i in range(-n//2,0)]

# Exemple d'utilisation :
  # Nombre d'échantillons
T = 1/samplerate  # Période d'échantillonnage
x=data[:4000]
x=pad2n(x)
N = len(x)
print(T,N)


# Normaliser le signal pour faciliter l'affichage
x = np.array(x, dtype=np.float32)
x = x / np.max(np.abs(x))  # Normalisation du signal

# Appliquer la FFT sur le signal
X = fft_padded(x)
frequencies = fftfreq(N, T)

# Affichage des résultats
plt.figure(figsize=(10, 6))

print("si")
# Affichage du signal d'origine
plt.subplot(2, 1, 1)
plt.plot(np.arange(N) * T, x[:N])  # Afficher une partie du signal
plt.title("Signal Original Normalisé")

# Affichage de la magnitude de la FFT
plt.subplot(2, 1, 2)
plt.bar(frequencies, np.abs(X))
plt.title("Magnitude de la FFT")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()