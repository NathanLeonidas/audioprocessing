import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd


# Fonction pour détecter la fréquence fondamentale avec FFT
def naive_fft_fundamental(signal, samplerate, window_size, hop_size, treshold):
    f0 = []
    times = []
    for start in range(0, len(signal) - window_size, hop_size):
        end = start + window_size
        window = signal[start:end]
        if np.mean(np.abs(window))>treshold:
            windowed_signal = window * np.hamming(len(window))  # Appliquer une fenêtre de Hamming
            spectrum = np.fft.fft(windowed_signal)[:len(windowed_signal)//2]  # Spectre
            frequencies = np.fft.fftfreq(len(windowed_signal), d=1/samplerate)[:len(windowed_signal)//2]
            magnitude = np.abs(spectrum)
            
            # Trouver le pic maximal dans les fréquences
            fundamental_freq = frequencies[np.argmax(magnitude)]
            f0.append(fundamental_freq)
            
        else:
            f0.append(0)
        times.append((start + window_size//2) / samplerate)
    print("calculated naive fft")
    return np.array(times), np.array(f0)

def autocorrelation_fundamental(signal, samplerate, window_size, hop_size, fmin, fmax, treshold):
    f0 = []
    times = []
    Tmin = int(samplerate / fmax)
    Tmax = int(samplerate / fmin)

    for start in range(0, len(signal) - window_size, hop_size):
        end = start + window_size
        window = signal[start:end]
        if np.mean(np.abs(window))>treshold:
            windowed_signal = window * np.hamming(len(window))  # Appliquer une fenêtre de Hamming

            # Calcul de l'autocorrélation
            autocorr = np.correlate(windowed_signal, windowed_signal, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]  # Garder uniquement la seconde moitié

            # Ignorer les lags en dehors de la plage utile
            autocorr[:Tmin] = 0
            autocorr[Tmax:] = 0

            # Trouver le lag du premier maximum local
            lag = np.argmax(autocorr)
            autocorr[lag]=0

            # Calculer la fréquence fondamentale et vérifier qu'elle est dans la plage
            fundamental_freq = samplerate / lag
            if fmin <= fundamental_freq <= fmax:
                f0.append(fundamental_freq)
            else:
                f0.append(0)  # Fréquence hors de la plage
        else:
            f0.append(0)  # Aucun pic détecté

        times.append((start + window_size//2) / samplerate)
    print("calculated autocorrelation")
    return np.array(times), f0 #np.array(f0)




# Chargement du fichier audio flute
data, samplerate = sf.read('D:\\Ecole\\CS\\METZ2A\\Traitement audio\\audioprocessing\\audio_files\\fluteircam.wav')
x = data
T = 1 / samplerate

# Paramètres
window_size = 0.02 # Taille de la fenêtre en secondes
hop_size = 0.01  # Décalage entre les fenêtres (en secondes)
treshold = 0.003
Fmin = librosa.note_to_hz('C4')
Fmax = librosa.note_to_hz('C6')

#conversion en nbre d'échantillons
window_size = int(window_size / T)
hop_size = int(hop_size / T)

# Calcul avec chaque méthode
times, f0_fft = naive_fft_fundamental(x, samplerate, window_size, hop_size,treshold)
times, f0_autocorr = autocorrelation_fundamental(x, samplerate, window_size, hop_size, Fmin, Fmax,treshold)

#récupération des vraies valeurs
filepath = 'D:\\Ecole\\CS\\METZ2A\\Traitement audio\\audioprocessing\\documentation_cours\\veriteterrainflute.txt'
reference = pd.read_csv(filepath, sep='\s+', header=None, names=['debut', 'fin', 'frequence'])
f0_true = []
for i in times:
    row = reference[(reference['debut']<=i) & (reference['fin'] > i)]
    if not row.empty:
        f0_true.append(row['frequence'].values[0])
    else:
        f0_true.append(0)
f0_true = np.array(f0_true)
print(f0_true)

# Affichage des résultats
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(times, f0_fft, label='FFT naive', color='green')
plt.plot(times, f0_autocorr, label='Autocorrélation', color='purple')
plt.plot(times, f0_true, label='Valeurs données', color='black')
plt.title("Fréquence Fondamentale détectée dans la flute")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence fondamentale (Hz)")
plt.legend()

n=len(f0_autocorr)
print('Bilan pour la flute:')
print('erreur moyenne L1 de la méthode naive FFT:'+str(np.linalg.norm(f0_true - f0_fft, ord=1)/n))
print('erreur moyenne L2 de la méthode naive FFT:'+str(np.linalg.norm(f0_true - f0_fft, ord=2)/np.sqrt(n)))
print('erreur moyenne L1 de la méthode par autocorrelation:'+str(np.linalg.norm(f0_true - f0_autocorr, ord=1)/n))
print('erreur moyenne L2 de la méthode par autocorrelation:'+str(np.linalg.norm(f0_true - f0_autocorr, ord=2)/np.sqrt(n)))

print("On constate que la méthode d'autocorrélation est plus précise.")
print('En effet, la méthode naive ne détecte parfois pas la fondamentale mais les harmoniques.')





# Chargement du fichier audio voix
data, samplerate = sf.read('D:\\Ecole\\CS\\METZ2A\\Traitement audio\\audioprocessing\\audio_files\\voiceP.wav')
x = data
T = 1 / samplerate

# Calcul avec chaque méthode
times, f0_fft = naive_fft_fundamental(x, samplerate, window_size, hop_size,treshold)
times, f0_autocorr = autocorrelation_fundamental(x, samplerate, window_size, hop_size, Fmin, Fmax,treshold)

#récupération des vraies valeurs
filepath = 'D:\\Ecole\\CS\\METZ2A\\Traitement audio\\audioprocessing\\documentation_cours\\veriteterrainvoiceP.txt'
reference = pd.read_csv(filepath, sep='\s+', header=None, names=['debut', 'fin', 'frequence'])
f0_true = []
for i in times:
    row = reference[(reference['debut']<=i) & (reference['fin'] > i)]
    if not row.empty:
        f0_true.append(row['frequence'].values[0])
    else:
        f0_true.append(0)
f0_true = np.array(f0_true)
print(f0_true)

# Affichage des résultats
plt.subplot(2,1,2)
plt.plot(times, f0_fft, label='FFT naive', color='green')
plt.plot(times, f0_autocorr, label='Autocorrélation', color='purple')
plt.plot(times, f0_true, label='Valeurs données', color='black')
plt.title("Fréquence Fondamentale détectée dans la voix")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence fondamentale (Hz)")
plt.legend()
plt.tight_layout()
plt.show()

n=len(f0_autocorr)
print('Bilan pour la voix')
print('erreur moyenne L1 de la méthode naive FFT:'+str(np.linalg.norm(f0_true - f0_fft, ord=1)/n))
print('erreur moyenne L2 de la méthode naive FFT:'+str(np.linalg.norm(f0_true - f0_fft, ord=2)/np.sqrt(n)))
print('erreur moyenne L1 de la méthode par autocorrelation:'+str(np.linalg.norm(f0_true - f0_autocorr, ord=1)/n))
print('erreur moyenne L2 de la méthode par autocorrelation:'+str(np.linalg.norm(f0_true - f0_autocorr, ord=2)/np.sqrt(n)))

print("On constate que la méthode d'autocorrélation est plus précise.")
print('En effet, la méthode naive a du mal a détecter les vibratos légers.')
print('Globalement les deux méthodes sont moins efficaces pour détecter la fondamentale')