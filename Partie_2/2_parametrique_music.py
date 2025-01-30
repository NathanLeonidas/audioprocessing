import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, eig
import os

# Charger le signal audio
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(script_dir, 'audio_files', 'croisement.wav')
y, sr = librosa.load(file_path, sr=None)

# Paramètres pour le découpage
frame_size = 2**10  # Taille des fenêtres
hop_size = int(sr * 0.01)  # Décalage entre les fenêtres (10 ms ici)
order = 100  # Ordre du modèle MUSIC

# Découper le signal en fenêtres glissantes
frames = [y[i:i + frame_size] for i in range(0, len(y) - frame_size, hop_size)]

def music_frequencies(signal, order, sr, num_freqs=2):
    """
    Implémente MUSIC pour estimer les fréquences dominantes dans une trame.
    Retourne les `num_freqs` fréquences principales.
    """
    N = len(signal)
    if N <= order:
        raise ValueError("L'ordre du modèle doit être inférieur à la taille du signal.")

    # Matrice de covariance
    R = np.correlate(signal, signal, mode='full')[N - 1:]  # Corrélation auto
    R_matrix = toeplitz(R[:order])  # Matrice Toeplitz de covariance

    # Décomposition en valeurs propres
    eigvals, eigvecs = eig(R_matrix)
    eigvecs = eigvecs[:, np.argsort(eigvals)]  # Trier par valeurs propres croissantes

    # Sous-espace de bruit (petites valeurs propres)
    noise_space = eigvecs[:, :order // 2]

    # Recherche des fréquences dominantes
    freqs = np.linspace(0, sr / 2, 1000)  # Résolution des fréquences pour recherche
    pseudo_spectrum = []
    for f in freqs:
        steering_vector = np.exp(-1j * 2 * np.pi * f * np.arange(order) / sr)
        psd = 1 / np.abs(steering_vector @ noise_space @ noise_space.T.conj() @ steering_vector.T)
        pseudo_spectrum.append(psd)

    pseudo_spectrum = np.array(pseudo_spectrum)
    
    # Trouver les `num_freqs` maxima locaux du pseudo-spectre
    dominant_indices = np.argsort(pseudo_spectrum)[-num_freqs:][::-1]  # Indices des maxima
    dominant_freqs = freqs[dominant_indices]  # Fréquences correspondantes

    return dominant_freqs

# Extraction des deux fréquences dominantes pour chaque trame
freq1, freq2 = [], []
i = 0
for frame in frames:
    i += 1
    if i % 10 == 0:
        print(f"Processing frame {i}...")
    try:
        dominant_freqs = music_frequencies(frame, order, sr, num_freqs=2)
        freq1.append(dominant_freqs[0])
        freq2.append(dominant_freqs[1])
    except ValueError:
        freq1.append(0)
        freq2.append(0)

# Temps pour chaque trame
time_axis = np.arange(len(freq1)) * hop_size / sr

# Affichage des fréquences dominantes
plt.figure(figsize=(10, 6))
plt.plot(time_axis, freq1, label="Fréquence 1 (Hz)", color="blue", marker='x', linestyle='', markersize=2.5)
plt.plot(time_axis, freq2, label="Fréquence 2 (Hz)", color="red", marker='x', linestyle='', markersize=2.5)
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Fréquences des deux sinusoïdes en fonction du temps")
plt.legend()
plt.grid()
plt.show()