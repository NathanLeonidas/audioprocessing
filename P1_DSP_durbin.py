import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from scipy.signal import find_peaks
from scipy.linalg import solve_toeplitz, toeplitz
import pandas as pd
import librosa

# Construire le chemin relatif vers le fichier audio (changer suivant le fichier voulu)
file = 'voiceP.wav'
script_dir = os.path.dirname(os.path.abspath(__file__))  # Répertoire du script
audio_path = os.path.join(script_dir, 'audio_files', file)

# Lire le fichier audio
data, sample_rate = sf.read(audio_path)
x = data
T = 1 / sample_rate



# Vérifier si le fichier est mono ou stéréo
if len(data.shape) > 1:
    data = data[:, 0]  # Prendre un seul canal si stéréo

def pourcent_bonne_classif(signal, target, error):
    return str(sum(np.abs(signal-target)/np.array(target)<error/100))+"/"+str(len(signal))


def compute_dsp(a, order, freq_range, sigma2):
    exp_term = np.exp(-1j * 2 * np.pi * freq_range[:, None] * np.arange(1, order + 1) / sample_rate)
    sum_term = np.dot(exp_term, a)
    denominator = sample_rate * np.abs(1 + sum_term) ** 2
    return sigma2 / denominator

def find_peaks_simple(x, distancemin):
    # Convertir en numpy array pour la manipulation
    x = np.array(x)
    peak1 = np.argmax(x)
    peak2 = 0

    # Trouver les maxima locaux
    for i in range(1, len(x) - 1):
        if x[i] >= x[i-1] and x[i] >= x[i+1]:
            if x[i] > x[peak2] and i!=peak1:
                peak2 = i

    confidence = x[peak2]/x[peak1]

    return peak1, peak2, confidence



def f0_levdurb(data, n_fenetre, window_size, n_hop,treshold):
    # Créez toutes les fenêtres en une seule opération
    num_frames = (len(data) - n_fenetre) // n_hop
    indices = np.arange(0, num_frames * n_hop, n_hop)[:, None] + np.arange(n_fenetre) #matrice des indices des fenetres
    windows = data[indices]  # (num_frames, n_fenetre)
    print(indices)


    # Initialisation pour stocker les DSP et le suivi des fréquences
    dsp_list = []

    # Calcul de l'autocorrélation et des coefficients AR avec Levinson-Durbin
    order = 300  # Choisir l'ordre du modèle AR (p)
    n_freqs = 8000
    freq_range = np.linspace(0, sample_rate / 2, num=n_freqs)
    list_peak = []

    
    
    for i, x in enumerate(windows):            
            r = np.correlate(x, x, mode='full')[len(x) - 1:] / len(x)
            # Résolution du système T x = B avec T la matrice de Topelitz des coefs autocorrélation
            row = r[:order]
            column = r[:order]
            B = -r[1: order + 1]
            coefs_autoregressifs = solve_toeplitz((row, column), B) # Résoudre le système de Toeplitz pour les coefficients AR
            sigma2 = r[0] + np.dot(r[1:order + 1], coefs_autoregressifs) # Calcul de sigma2
            dsp_ = compute_dsp(coefs_autoregressifs, order, freq_range, sigma2) # Calcul de la DSP pour cette fenêtre7
            dsp = dsp_ / np.max(dsp_)
            dsp_list.append(dsp)  # Normalisation

            #trouver les maximums de la dsp
            list_peak.append(freq_range[np.argmax(dsp)])

            # Afficher la progression
            if i % 10 == 0:
                print(f"{i} / {num_frames} frames traitées.")

    # Visualiser les fréquences dominantes
    plt.figure(figsize=(12, 6))

    # Création de l'image avec les coefficients AR
    img = plt.imshow(np.array(dsp_list).T, origin="lower", aspect="auto",
                     cmap="viridis", interpolation='none',  extent=[0, num_frames, 0, sample_rate / 2])

    # Ajout de la colorbar en associant l'objet `img`
    plt.colorbar(img, label="Amplitude de la DSP")
    plt.title("Estimation autoregressive de la DSP de "+file)
    plt.xlabel("Numero de fenetre")
    plt.ylabel("Estimation de la DSP")

    plt.legend()

    plt.show()


# Paramètres
window_size = 0.02 # Taille de la fenêtre en secondes
hop_size = 0.01  # Décalage entre les fenêtres (en secondes)
treshold = 0.003 # Seuil en dessous duquel c'est du bruit
Fmin = librosa.note_to_hz('C3')
Fmax = librosa.note_to_hz('C6')

#conversion en nbre d'échantillons
n_fenetre = int(window_size / T)
hop_size = int(hop_size / T)

# Calcul avec chaque méthode
times, f0_lev = f0_levdurb(x, n_fenetre, window_size, hop_size,treshold)

#récupération des vraies valeurs
filepath = os.path.join(script_dir, 'documentation_cours', 'veriteterrainflute.txt')
reference = pd.read_csv(filepath, sep='\s+', header=None, names=['debut', 'fin', 'frequence'])
f0_true = []
for i in times:
    row = reference[(reference['debut']<=i) & (reference['fin'] > i)]
    if not row.empty:
        f0_true.append(row['frequence'].values[0])
    else:
        f0_true.append(0)
f0_true = np.array(f0_true)


error=5
print('Avec une erreur de '+str(error)+'%')
print("taux bonne classif méthode durbin "+str(pourcent_bonne_classif(f0_lev,f0_true,error)))