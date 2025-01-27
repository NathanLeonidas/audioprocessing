import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from scipy.signal import find_peaks
from scipy.linalg import solve_toeplitz, toeplitz

# Construire le chemin relatif vers le fichier audio
script_dir = os.path.dirname(os.path.abspath(__file__))  

# Charger le fichier audio avec soundfile
file_path = os.path.join(script_dir, 'audio_files', 'croisement4.wav')
data, sample_rate = sf.read(file_path)
T = 1 / sample_rate



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

def compute_dsp(a, order, freq_range, sigma2):
    exp_term = np.exp(-1j * 2 * np.pi * freq_range[:, None] * np.arange(1, order + 1) / sample_rate)
    sum_term = np.dot(exp_term, a)
    denominator = sample_rate * np.abs(1 + sum_term) ** 2
    return sigma2 / denominator

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



def separate_levdur(n_fenetre, data):
    # Créez toutes les fenêtres en une seule opération
    num_frames = (len(data) - n_fenetre) // n_hop
    indices = np.arange(0, num_frames * n_hop, n_hop)[:, None] + np.arange(n_fenetre)
    windows = data[indices]  # (num_frames, n_fenetre)

    # Calculer l'autocorrélation pour toutes les fenêtres
    autocorr = np.apply_along_axis(
        lambda x: np.correlate(x, x, mode='full')[len(x) - 1:] / len(x),
        axis=1,
        arr=windows
    )  # (num_frames, n_fenetre)

    # Initialisation pour stocker les DSP
    dsp_list = []

    # Calcul de l'autocorrélation et des coefficients AR avec Levinson-Durbin
    order = 300  # Choisir l'ordre du modèle AR (p)
    freq_range = np.linspace(0, sample_rate / 2, num=8000)
    
    for i, r in enumerate(autocorr):
            # Extraire la première colonne et ligne pour Toeplitz
            row = r[:order]
            column = r[:order]
            B = -r[1: order + 1]

            # Résoudre le système de Toeplitz pour les coefficients AR
            coefs_autoregressifs = solve_toeplitz((row, column), B)

            # Calcul de sigma2
            sigma2 = r[0] + np.dot(r[1:order + 1], coefs_autoregressifs)

            # Calcul de la DSP pour cette fenêtre
            dsp = compute_dsp(coefs_autoregressifs, order, freq_range, sigma2)
            dsp_list.append(dsp / np.max(dsp))  # Normalisation

            # Afficher la progression
            if i % 10 == 0:
                print(f"{i} / {num_frames} frames traitées.")

    # Visualiser les fréquences dominantes
    plt.figure(figsize=(12, 6))

    # Création de l'image avec les coefficients AR
    img = plt.imshow(np.array(dsp_list).T, origin="lower", aspect="auto",
                     cmap="viridis", interpolation='none')

    # Ajout de la colorbar en associant l'objet `img`
    plt.colorbar(img, label="Coefficients AR")
    plt.title("Coefficients AR")
    plt.xlabel("Temps (s)")
    plt.ylabel("Coefficient AR")

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
for window_size_ in [ 0.05, 0.07]:
    n_fenetre = int(window_size_ / T)
    separate_levdur(n_fenetre, data)

print('Pour la fenêtre rectangulaire:')
print('La fenêtre à 0.01 est trop petite (on sépare trop de fréquences mal) et celle de 0.7 trop grande (la raie glissante est diffuse) mais avec les meilleurs résultats')
