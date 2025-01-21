import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from scipy.signal import find_peaks

# Construire le chemin relatif vers le fichier audio
script_dir = os.path.dirname(os.path.abspath(__file__))  

# Charger le fichier audio avec soundfile
file_path = os.path.join(script_dir, 'audio_files', 'croisement.wav')
data, sample_rate = sf.read(file_path)
T = 1 / sample_rate

data = np.sin(200*np.pi*2*np.linspace(0,2,int(2*sample_rate)))


# Vérifier si le fichier est mono ou stéréo
if len(data.shape) > 1:
    data = data[:, 0]  # Prendre un seul canal si stéréo

def levinson_durbin(r,order):
    # Initialization
    p=order-1
    a = np.zeros(p + 1)
    a[0] = 1.0
    G = np.zeros(p)
    eps = np.zeros(p + 1)
    eps[0] = r[0]

    for tau in range(p):
        # Compute reflection coefficient
        conv = r[tau + 1]
        for s in range(1, tau + 1):
            conv = conv + a[s] * r[tau - s + 1]
        G[tau] = -conv / eps[tau]

        # Update 'a' vector
        a_cpy = np.copy(a)
        for s in range(1, tau + 1):
            a_cpy[s] = a[s] + G[tau] * np.conj(a[tau - s + 1])
        a = a_cpy
        a[tau + 1] = G[tau]
        eps[tau + 1] = eps[tau] * (1 - np.abs(G[tau])**2)
    return a, G, eps

def compute_dsp(a, order, freq_range, sigma2):
    dsp = []
    for f in freq_range:
        numerator = sigma2
        denominator = sample_rate*abs(1 + np.sum(a * np.exp(-1j * 2 * np.pi * f * np.arange(1, order + 1)) / sample_rate)) ** 2
        dsp.append(numerator / denominator)
    return np.array(dsp)


def find_peaks_simple(x, distance, padding, T, n_fft):
    # Convertir en numpy array pour la manipulation
    x = np.array(x)
    
    peak1 = np.argmax(x)
    peak2 = 0

    # Trouver les maxima locaux
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:  # Comparaison avec voisins
            if x[i] > x[peak2] and np.abs(i - peak1) / (T * padding) > 1 / (T * n_fft):  # Filtre de hauteur
                peak2 = i

    return np.array([peak1, peak2])


def separate_levdur(data):

    # Calcul de l'autocorrélation et des coefficients AR avec Levinson-Durbin
    order = 10  # Choisir l'ordre du modèle AR (p)
    # Plage de fréquences pour calculer la DSP (ici, de 0 Hz à la moitié de la fréquence d'échantillonnage)
    freq_range = np.linspace(0, sample_rate / 2, num=1024)

    r = np.correlate(data, data, mode='full')[len(data)-1:]/len(data)  # Fonction d'autocorrélation
    a, G, error = levinson_durbin(r,order)
    dsp = compute_dsp(a, order, freq_range, error)
    print(a)
    print(error)


    # Visualiser les fréquences dominantes
    plt.figure(figsize=(12, 6))

    # Création de l'image avec les coefficients AR
    plt.plot(freq_range,dsp, color='red')

    plt.show()

# Paramètres
window_size = 0.05  # Taille de la fenêtre en secondes
hop_size = 0.01  # Décalage entre les fenêtres (en secondes)
padding = 2**16
distancemin = 1  # en Hz, espace minimal entre deux crêtes que l'on peut bien séparer

# conversion en nombre d'échantillons
n_fenetre = int(window_size / T)
n_hop = int(hop_size / T)
window = np.ones(n_fenetre)

# Essai pour différentes fenêtres

separate_levdur(data)
