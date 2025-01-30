import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
# Charger le fichier audio

# Construire le chemin relatif vers le fichier audio
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Charger le fichier audio avec soundfile
file = 'myson.wav'
file_path = os.path.join(script_dir, 'audio_files', file)

y, sr = librosa.load(file_path, sr=None)

# Calcul du spectre de puissance
spectrum = np.abs(np.fft.rfft(y))
frequencies = np.fft.rfftfreq(len(y), d=1/sr)

# Détection des pics
peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1)

# Affichage du spectre avec pics dominants
plt.figure(figsize=(12, 6))
plt.plot(frequencies, spectrum, label="Spectre du signal")
plt.scatter(frequencies[peaks], spectrum[peaks], color='red', label="Pics dominants")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.title("Fréquences dominantes du signal")
plt.legend()
plt.grid()
plt.show()
