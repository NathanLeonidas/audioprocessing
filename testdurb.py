import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from scipy.signal import find_peaks
from scipy.linalg import solve_toeplitz, toeplitz

# Construire le chemin relatif vers le fichier audio
script_dir = os.path.dirname(os.path.abspath(__file__))  

# Charger le fichier audio avec soundfile
file_path = os.path.join(script_dir, 'audio_files', 'croisement.wav')
_, sample_rate = sf.read(file_path)
T = 1 / sample_rate

data = np.sin(100*np.pi*2*np.linspace(0,5,int(5*sample_rate))) + np.sin(1000*np.pi*2*np.linspace(0,5,int(5*sample_rate)))
#data = np.random.randn(1*sample_rate)


# Vérifier si le fichier est mono ou stéréo
if len(data.shape) > 1:
    data = data[:, 0]  # Prendre un seul canal si stéréo



def compute_dsp(a, order, freq_range, sigma2):
    dsp = []
    for f in freq_range:
        numerator = sigma2
        sum = 0
        for i in range(order):
            sum += a[i] * np.exp(-1j * 2 * np.pi * f * (i+1)/ sample_rate)
        denominator = sample_rate*abs(1 + sum) ** 2
        dsp.append(numerator / denominator)
    return np.array(dsp)





def separate_levdur(data):

    # Calcul de l'autocorrélation et des coefficients AR avec Levinson-Durbin
    order = 200  # Choisir l'ordre du modèle AR (p)
    # Plage de fréquences pour calculer la DSP (ici, de 0 Hz à la moitié de la fréquence d'échantillonnage)
    freq_range = np.linspace(-sample_rate/2, sample_rate/2, num=10000)

    r = np.correlate(data, data, mode='full')[len(data)-1:]/len(data)  # Fonction d'autocorrélation
    row = r[:order]
    column = r[:order]
    B = -r[1: order + 1]
    coefs_autoregressifs = solve_toeplitz((row, column), B)

    sigma2 = r[0]
    for i in range(len(coefs_autoregressifs)):
        sigma2 += r[i+1] * coefs_autoregressifs[i]



    print(sigma2)

    dsp = compute_dsp(coefs_autoregressifs, order, freq_range, sigma2)



    # Visualiser les fréquences dominantes
    plt.figure(figsize=(12, 6))

    # Création de l'image avec les coefficients AR
    plt.semilogy(freq_range, dsp, color='red')


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
