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
print(data)

# Vérifier si le fichier est mono ou stéréo
if len(data.shape) > 1:
    data = data[:, 0]  # Prendre un seul canal si stéréo


def levinson_durbin(r, order):
    """
    Implémente l'algorithme de Levinson-Durbin pour résoudre les coefficients de prédiction linéaire.
    r: La fonction d'autocorrélation (liste numpy)
    order: L'ordre du modèle AR (p)
    """
    a = np.zeros(order)  # Coefficients AR
    e = np.zeros(order)  # Erreur (variance résiduelle)
    
    a[0] = -r[1] / r[0]
    e[0] = r[0] * (1 - a[0] ** 2)

    for k in range(1, order):
        # Calcul de lambda_k en utilisant la formule correcte
        lambda_k = (r[k + 1] - np.dot(a[:k], r[k::-1][:k])) / e[k - 1]
        a[k] = lambda_k
        e[k] = e[k - 1] * (1 - lambda_k ** 2)
        
        # Mise à jour des coefficients AR
        a[:k] = a[:k] - lambda_k * a[k - 1::-1]

    return a, e[-1]


def compute_dsp(a, order, freq_range, sigma2):
    """
    Calculer la densité spectrale de puissance (DSP) à partir des coefficients AR.
    a: Les coefficients AR obtenus via Levinson-Durbin
    order: L'ordre du modèle AR
    freq_range: Plage de fréquences pour lesquelles calculer la DSP
    sigma2: La variance du bruit
    """
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


def separate_levdur(n_fenetre, data):
    # Calculer la FFT
    frames = range(0, len(data) - n_fenetre, n_hop)
    allwindows = [data[i:i + n_fenetre] for i in frames]
    liste_dsp = []

    # Fréquences et temps associées
    times = np.arange(len(frames)) * hop_size * T

    # Calcul de l'autocorrélation et des coefficients AR avec Levinson-Durbin
    order = 10  # Choisir l'ordre du modèle AR (p)
    # Plage de fréquences pour calculer la DSP (ici, de 0 Hz à la moitié de la fréquence d'échantillonnage)
    freq_range = np.linspace(0, sample_rate / 2, num=1024)

    for window in allwindows:
        r = np.correlate(window, window, mode='full')[len(window) - 1:]  # Fonction d'autocorrélation
        a, error = levinson_durbin(r, order)
        dsp = compute_dsp(a, order, freq_range, error)
        liste_dsp.append(dsp)

    # Visualiser les fréquences dominantes
    plt.figure(figsize=(12, 6))

    # Création de l'image avec les coefficients AR
    img = plt.imshow(np.array(liste_dsp).T, origin="lower", aspect="auto",
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
for window_size_ in [0.01, 0.05, 0.07]:
    n_fenetre = int(window_size_ / T)
    separate_levdur(n_fenetre, data)

print('Pour la fenêtre rectangulaire:')
print('La fenêtre à 0.01 est trop petite (on sépare trop de fréquences mal) et celle de 0.7 trop grande (la raie glissante est diffuse) mais avec les meilleurs résultats')
