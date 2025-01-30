import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os


script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(script_dir, 'audio_files', 'croisement2.wav')

# Charger le signal avec Librosa
y, sr = librosa.load(file_path, sr=None)
hop_size = int(sr * 0.01)  # Décalage de 10 ms en échantillons

# Calculer le spectrogramme avec STFT
D = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_size, window='hann'))

# Affichage
plt.figure(figsize=(10, 6))
librosa.display.specshow(np.abs(D), sr=sr, hop_length=512, y_axis='linear', x_axis='time')
plt.colorbar(format="%+2.0f amplitude")
plt.title("Spectrogramme du signal")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.show()
