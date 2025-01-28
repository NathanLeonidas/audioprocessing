import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from scipy.signal import find_peaks
from scipy.linalg import solve_toeplitz, toeplitz

# Construire le chemin relatif vers le fichier audio
script_dir = os.path.dirname(os.path.abspath(__file__))  

# Charger le fichier audio avec soundfile
file = 'myson.wav'
file_path = os.path.join(script_dir, 'audio_files', file)
data, sample_rate = sf.read(file_path)
T = 1 / sample_rate



# Vérifier si le fichier est mono ou stéréo
if len(data.shape) > 1:
    data = data[:, 0]  # Prendre un seul canal si stéréo



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



def separate_levdur(n_fenetre, data):
    # Créez toutes les fenêtres en une seule opération
    num_frames = (len(data) - n_fenetre) // n_hop
    indices = np.arange(0, num_frames * n_hop, n_hop)[:, None] + np.arange(n_fenetre) #matrice des indices des fenetres
    windows = data[indices]  # (num_frames, n_fenetre)

    # Calculer l'autocorrélation pour toutes les fenêtres
    autocorr = np.apply_along_axis(
        lambda x: np.correlate(x, x, mode='full')[len(x) - 1:] / len(x),
        axis=1,
        arr=windows
    )

    # Initialisation pour stocker les DSP et le suivi des fréquences
    dsp_list = []
    freqs_a = []
    freqs_b = []
    ampls_a = []
    ampls_b = []
    not_separated_correctly_a = []
    not_separated_correctly_b = []
    n_not_separated_correctly = 0

    # Calcul de l'autocorrélation et des coefficients AR avec Levinson-Durbin
    order = 300  # Choisir l'ordre du modèle AR (p)
    n_freqs = 8000
    freq_range = np.linspace(0, sample_rate / 2, num=n_freqs)

    
    
    for i, r in enumerate(autocorr):
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
            peak1, peak2, confidence = find_peaks_simple(dsp, distancemin)
            ampl1, freq1 = dsp[peak1], freq_range[peak1]
            ampl2, freq2 = dsp[peak2], freq_range[peak2]

            # On rejoint habilement les frequences pour qu'elles collent au plus des 
            # dernieres valeurs (amplitudes et fréquences)
            if len(freqs_a) == 0:
                ampls_a.append(ampl1)
                freqs_a.append(freq1)
                ampls_b.append(ampl2)
                freqs_b.append(freq2)
            elif np.abs(freq1 - freq2) < distancemin:
                #récupère la moyenne des 40 dernieres amplitudes
                meanamps_a = np.mean(ampls_a[-min(len(ampls_a),40):])
                meanamps_b = np.mean(ampls_b[-min(len(ampls_b),40):])
                if np.abs(meanamps_a - ampl1) < np.abs(meanamps_b - ampl1):
                    ampls_a.append(ampl1)
                    freqs_a.append(freq1)
                    ampls_b.append(ampl2)
                    freqs_b.append(freq2)
                else:
                    ampls_a.append(ampl2)
                    freqs_a.append(freq2)
                    ampls_b.append(ampl1)
                    freqs_b.append(freq1)
            else:
                if np.abs(freq1 - freqs_a[-1]) < np.abs(freq1 - freqs_b[-1]):
                    freqs_a.append(freq1)
                    ampls_a.append(ampl1)
                    freqs_b.append(freq2)
                    ampls_b.append(ampl2)
                else:
                    freqs_a.append(freq2)
                    ampls_a.append(ampl2)
                    freqs_b.append(freq1)
                    ampls_b.append(ampl1)

            # Prédire la prochaine fréquence et vérifier la séparation correcte
            if confidence<0.01 or freqs_b[-1]==0 or freqs_a[-1]==0:
                not_separated_correctly_a.append(freqs_a[-1])
                not_separated_correctly_b.append(freqs_b[-1])
                n_not_separated_correctly += 1
            else:
                not_separated_correctly_a.append(-float('inf'))
                not_separated_correctly_b.append(-float('inf'))


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

    # Ajouter les fréquences dominantes
    #plt.scatter(np.arange(num_frames), freqs_a, color="red", label="Fréquences A", s=5)
    #plt.scatter(np.arange(num_frames), freqs_b, color="blue", label="Fréquences B", s=5)
    #plt.scatter(np.arange(num_frames), not_separated_correctly_a, color="black", label="Fréquences A mal séparées", s=20, marker="x")
    #plt.scatter(np.arange(num_frames), not_separated_correctly_b, color="black", label="Fréquences B mal séparées", s=20, marker="x")
    
    #plt.text(200, 1.1 * max(freqs_a[200], freqs_b[200]), str(n_not_separated_correctly) +"/"+str(num_frames) +" problématiques", fontsize=12,
    #         ha='center')

    plt.legend()

    plt.show()


# Paramètres
window_size = 0.05  # Taille de la fenêtre en secondes
hop_size = 0.01  # Décalage entre les fenêtres (en secondes)
distancemin = 10  # en Hz, espace minimal entre deux crêtes que l'on peut bien suivre

# conversion en nombre d'échantillons
n_fenetre = int(window_size / T)
n_hop = int(hop_size / T)
window = np.ones(n_fenetre)

# Essai pour différentes fenêtres
for window_size_ in [0.08]:
    n_fenetre = int(window_size_ / T)
    separate_levdur(n_fenetre, data)
