import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Charger le fichier audio
file_path = "audio_files/myson.wav"  # Ajuste ce chemin si nécessaire
y, sr = librosa.load(file_path, sr=None)

# Calcul du spectre de puissance
power=np.fft.rfft(y)
frequencies=np.fft.rfftfreq(len(y),1/sr)

# Détection des pics
peaks, _ = find_peaks(power, height=np.max(power) * 0.1)


# Affichage du spectre avec pics dominants
plt.figure(figsize=(12, 6))
plt.plot(frequencies, power, label="Spectre du signal")
plt.scatter(frequencies[peaks], power[peaks], color='red', label="Pics dominants")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.title("Fréquences dominantes du signal")
plt.legend()
plt.grid()
plt.show()
