import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

# Charger le fichier audio
file_path = "audio_files/myson.wav"  # Ajuste ce chemin si nécessaire
y, sr = librosa.load(file_path, sr=None)

# Affichage du signal temporel
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Représentation temporelle du signal audio")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
