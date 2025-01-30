import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Chemin du fichier audio
file_path = "audio_files/myson.wav"
y, sr = librosa.load(file_path, sr=None)

# Calcul de la STFT
D = librosa.stft(y, n_fft=2048, hop_length=80)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Affichage du spectrogramme
plt.figure(figsize=(12, 6))
librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="log")
plt.colorbar(label="Amplitude (dB)")
plt.title("Spectrogramme du signal (STFT)")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.show()

import scipy.signal as signal
layers = [ (0, 72),     # Couche 1 : 0 - 72 Hz  (graves, potentiellement la fondamentale)
    (72, 134),   # Couche 2 : 72 - 134 Hz (harmoniques basses)
    (134, 200),  # Couche 3 : 134 - 200 Hz (premiers harmoniques clairs)
    (200, 270),  # Couche 4
    (270, 340),  # Couche 5
    (340, 410),  # Couche 6 
    (410, 490),  # Couche 7 
    (490, 560),  # Couche 8 
    (630, 700),  # Couche 9 
    (770, 840)   # Couche 10 
    ]

# Prendre l'évolution de l'énergie dans la bande principale (ex. la première couche)
layer_id = 0
low, high = layers[layer_id]
freq_mask = (librosa.fft_frequencies(sr=sr, n_fft=2048) >= low) & (librosa.fft_frequencies(sr=sr, n_fft=2048) <= high)
sub_spectrogram = S_db[freq_mask, :]
energy_time = np.mean(sub_spectrogram, axis=0)

# Calculer la FFT de cette évolution temporelle
fft_result = np.abs(np.fft.rfft(energy_time))
fft_freqs = np.fft.rfftfreq(len(energy_time), d=512/sr)

# Afficher le spectre du rythme
plt.figure(figsize=(10, 4))
plt.plot(fft_freqs, fft_result, label="Spectre de la variation temporelle")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.title("Analyse fréquentielle des variations temporelles")
plt.legend()
plt.grid()
plt.show()

