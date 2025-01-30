import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Chemin du fichier audio
file_path = "audio_files/myson.wav"
y, sr = librosa.load(file_path, sr=None)

# Détection de la tonalité perçue
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

# Extraire les fréquences dominantes les plus puissantes
pitch_track = []
for t in range(magnitudes.shape[1]):
    index = magnitudes[:, t].argmax()
    pitch_track.append(pitches[index, t])

pitch_track = np.array(pitch_track)
pitch_track[pitch_track == 0] = np.nan  # Remplace les zéros par NaN pour éviter le bruit

# Affichage des fréquences perçues
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, len(y) / sr, len(pitch_track)), pitch_track, label="Fréquence perçue", color="red")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Évolution de la fréquence perçue dans le temps")
plt.grid()
plt.legend()
plt.show()
