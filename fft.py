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


def fft(x,T):
    #where x isn't padded size 2**n
    x=pad2n(x)
    full_spectre = fft_padded(x)
    spectre = full_spectre[:len(full_spectre)//2]
    N = len(x)
    full_frequencies = fftfreq(N,T)
    frequencies = full_frequencies[:N//2]
    return spectre, frequencies




#
# Exemple d'utilisation :
  # Nombre d'échantillons
T = 1/samplerate  # Période d'échantillonnage
x=data[:40000]





# Normaliser le signal pour faciliter l'affichage
x = np.array(x, dtype=np.float32)
#x = x / np.max(np.abs(x))
spectre, frequencies = fft(x,T)

# Affichage du signal d'origine
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(x)) * T, x)  # Afficher une partie du signal
plt.title("Signal Original Normalisé")

# Affichage de la magnitude de la FFT
plt.subplot(2, 1, 2)
plt.bar(frequencies, np.abs(spectre), color='red')
plt.title("Magnitude de la FFT")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()


# Trouver la fréquence fondamentale
def find_fundamental_frequency(spectre, frequencies):
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
    spectre, frequencies = fft(window, T)
    
    # Calcul de la fondamentale
    fundamental_frequency = find_fundamental_frequency(spectre, frequencies)
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