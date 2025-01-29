import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from scipy.signal import butter, filtfilt
import os

# Fonction pour détecter la fréquence fondamentale avec FFT
def naive_fft_fundamental(signal, samplerate, window_size, hop_size, treshold):
    f0 = []
    times = []
    for start in range(0, len(signal) - window_size, hop_size):
        end = start + window_size
        window = signal[start:end]
        if np.mean(np.abs(window))>treshold:
            windowed_signal = window #* np.hamming(len(window))  # Appliquer une fenêtre de Hamming
            padding = 2**15
            spectrum = np.fft.fft(windowed_signal,n=padding)[:padding//2]  # Spectre
            frequencies = np.fft.fftfreq(padding, d=1/samplerate)[:padding//2]
            magnitude = np.abs(spectrum)
            
            # Trouver le premier pic maximal dans les fréquences
            peaks=[]
            for i in range(1,len(magnitude)-1):
                if magnitude[i]>magnitude[i+1] and magnitude[i]>magnitude[i-1] and magnitude[i]>0.4*np.max(magnitude):
                    peaks.append(i)
            f0.append(frequencies[peaks[0]])
            
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
            windowed_signal = window  # Appliquer une fenêtre rectangulaire

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

def autocorrelation_fundamental_with_filter(signal, samplerate, window_size, hop_size, fmin, fmax, treshold, vibrato_freq=4.5):
    # Calculer la fréquence fondamentale avec autocorrélation
    times, f0 = autocorrelation_fundamental(signal, samplerate, window_size, hop_size, fmin, fmax, treshold)
    
    # Conception d'un filtre passe-bas pour supprimer le vibrato (environ 5 Hz)
    #il faut reconvertir en duréee
    freq_echantillon_f0 = samplerate / hop_size
    nyquist = freq_echantillon_f0 / 2
    vibrato_cutoff = vibrato_freq / nyquist  # Normaliser la fréquence du vibrato

    # Appliquer un filtre passe-bas (cut-off à vibrato_freq) sur f0
    b, a = butter(2, vibrato_cutoff, btype='low')  # Créer le filtre Butterworth
    f0_filtered = filtfilt(b, a, f0)  # Appliquer le filtre sur la fréquence fondamentale
    print("application d'un passe bas")
    return times, f0_filtered


def autocorrelation_fundamental_enveloppe(signal, samplerate, window_size, hop_size, fmin, fmax, treshold):
    # Calculer la fréquence fondamentale avec autocorrélation
    times, f0 = autocorrelation_fundamental(signal, samplerate, window_size, hop_size, fmin, fmax, treshold)

    envsup = [f0[0]]
    envinf = [f0[0]]
    for i in range(1,len(f0)-1):
        if f0[i]>=f0[i-1] and f0[i]>f0[i+1]:
            envsup.append(f0[i])
        else:
            envsup.append(envsup[-1])
        if f0[i]<=f0[i-1] and f0[i]<f0[i+1]:
            envinf.append(f0[i])
        else:
            envinf.append(envinf[-1])
    envinf.append(envinf[-1])
    envsup.append(envsup[-1])
    f0_moy = (np.array(envinf)+np.array(envsup))/2
    print("calcul de l'enveloppe")
    return times, f0_moy


def pourcent_bonne_classif(signal, target, error):
    return str(sum(np.abs(signal-target)/np.array(target)<error/100))+"/"+str(len(signal))

# Construire le chemin relatif vers le fichier audio
script_dir = os.path.dirname(os.path.abspath(__file__))  # Répertoire du script
audio_path = os.path.join(script_dir, 'audio_files', 'fluteircam.wav')

# Lire le fichier audio
data, samplerate = sf.read(audio_path)
x = data
T = 1 / samplerate

# Paramètres
window_size = 0.02 # Taille de la fenêtre en secondes
hop_size = 0.01  # Décalage entre les fenêtres (en secondes)
treshold = 0.003 # Seuil en dessous duquel c'est du bruit
Fmin = librosa.note_to_hz('C3')
Fmax = librosa.note_to_hz('C6')

#conversion en nbre d'échantillons
window_size = int(window_size / T)
hop_size = int(hop_size / T)

# Calcul avec chaque méthode
times, f0_fft = naive_fft_fundamental(x, samplerate, window_size, hop_size,treshold)
times, f0_autocorr = autocorrelation_fundamental(x, samplerate, window_size, hop_size, Fmin, Fmax,treshold)
times, f0_autocorr_vib = autocorrelation_fundamental_with_filter(x, samplerate, window_size, hop_size, Fmin, Fmax,treshold)
times, f0_autocorr_env = autocorrelation_fundamental_enveloppe(x, samplerate, window_size, hop_size, Fmin, Fmax,treshold)

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


# Affichage des résultats
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(times, f0_fft, label='FFT naive', color='green')
plt.plot(times, f0_autocorr, label='Autocorrélation', color='purple')
plt.plot(times, f0_autocorr_vib, label='Autocorrélation filtrée passe bas', color='red')
plt.plot(times, f0_true, label='Valeurs données', color='black')
plt.title("Fréquence Fondamentale détectée dans la flute")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence fondamentale (Hz)")
plt.legend()

n=len(f0_autocorr)
print('Bilan pour la flute:')
print('erreur moyenne L1 de la méthode naive FFT:'+str(np.linalg.norm(f0_true - f0_fft, ord=1)/n))
print('erreur moyenne L2 de la méthode naive FFT:'+str(np.linalg.norm(f0_true - f0_fft, ord=2)/np.sqrt(n)))
print('erreur moyenne L1 de la méthode par autocorrelation:' + str(np.linalg.norm(f0_true - f0_autocorr, ord=1)/n))
print('erreur moyenne L2 de la méthode par autocorrelation:' + str(np.linalg.norm(f0_true - f0_autocorr, ord=2)/np.sqrt(n)))
print('erreur moyenne L1 de la méthode par autocorrelation filtrée:'+str(np.linalg.norm(f0_true - f0_autocorr_vib, ord=1)/n))
print('erreur moyenne L2 de la méthode par autocorrelation filtrée:'+str(np.linalg.norm(f0_true - f0_autocorr_vib, ord=2)/np.sqrt(n)))
print('erreur moyenne L1 de la méthode par autocorrelation enveloppe:'+str(np.linalg.norm(f0_true - f0_autocorr_env, ord=1)/n))
print('erreur moyenne L2 de la méthode par autocorrelation enveloppe:'+str(np.linalg.norm(f0_true - f0_autocorr_env, ord=2)/np.sqrt(n)))
print('\n')
error=5
print('Avec une erreur de '+str(error)+'%')
print("taux bonne classif méthode naive FFT "+str(pourcent_bonne_classif(f0_fft,f0_true,error)))
print("taux bonne classif méthode autocorrelation "+str(pourcent_bonne_classif(f0_autocorr,f0_true,error)))
print("taux bonne classif méthode autocorrelation filtrée "+str(pourcent_bonne_classif(f0_autocorr_vib,f0_true,error)))
print("taux bonne classif méthode autocorrelation enveloppe "+str(pourcent_bonne_classif(f0_autocorr_env,f0_true,error)))

print('\n')
print("On constate que la méthode d'autocorrélation est plus précise.")
print('En effet, la méthode naive ne détecte parfois pas la fondamentale mais les harmoniques.')
print('De plus le filtrage enleve de la précision puisque la note de la flûte se maintient presque parfaitement')
print('et le gain que l on ferait avec le passse bas est perdu pendant le régime transitoire pour passer d une note à une autre')






# Chargement du fichier audio voix
audio_path = os.path.join(script_dir, 'audio_files', 'voiceP.wav')

# Lire le fichier audio
data, samplerate = sf.read(audio_path)
x = data
T = 1 / samplerate

# Calcul avec chaque méthode
times, f0_fft = naive_fft_fundamental(x, samplerate, window_size, hop_size,treshold)
times, f0_autocorr = autocorrelation_fundamental(x, samplerate, window_size, hop_size, Fmin, Fmax,treshold)
times, f0_autocorr_vib = autocorrelation_fundamental_with_filter(x, samplerate, window_size, hop_size, Fmin, Fmax,treshold)
times, f0_autocorr_env = autocorrelation_fundamental_enveloppe(x, samplerate, window_size, hop_size, Fmin, Fmax,treshold)

#récupération des vraies valeurs
filepath = os.path.join(script_dir, 'documentation_cours', 'veriteterrainvoiceP.txt')
reference = pd.read_csv(filepath, sep='\s+', header=None, names=['debut', 'fin', 'frequence'])
f0_true = []
for i in times:
    row = reference[(reference['debut']<=i) & (reference['fin'] > i)]
    if not row.empty:
        f0_true.append(row['frequence'].values[0])
    else:
        f0_true.append(0)
f0_true = np.array(f0_true)


# Affichage des résultats
plt.subplot(2,1,2)
plt.plot(times, f0_fft, label='FFT naive', color='green')
plt.plot(times, f0_autocorr, label='Autocorrélation', color='purple')
plt.plot(times, f0_autocorr_vib, label='Autocorrélation filtrée passe bas', color='red')
plt.plot(times, f0_true, label='Valeurs données', color='black')
plt.title("Fréquence Fondamentale détectée dans la voix")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence fondamentale (Hz)")
plt.legend()
plt.tight_layout()

n=len(f0_autocorr)
print('\n')
print('Bilan pour la voix')
print('erreur moyenne L1 de la méthode naive FFT:'+str(np.linalg.norm(f0_true - f0_fft, ord=1)/n))
print('erreur moyenne L2 de la méthode naive FFT:'+str(np.linalg.norm(f0_true - f0_fft, ord=2)/np.sqrt(n)))
print('erreur moyenne L1 de la méthode par autocorrelation:'+str(np.linalg.norm(f0_true - f0_autocorr, ord=1)/n))
print('erreur moyenne L2 de la méthode par autocorrelation:'+str(np.linalg.norm(f0_true - f0_autocorr, ord=2)/np.sqrt(n)))
print('erreur moyenne L1 de la méthode par autocorrelation filtrée:'+str(np.linalg.norm(f0_true - f0_autocorr_vib, ord=1)/n))
print('erreur moyenne L2 de la méthode par autocorrelation filtrée:'+str(np.linalg.norm(f0_true - f0_autocorr_vib, ord=2)/np.sqrt(n)))
print('erreur moyenne L1 de la méthode par autocorrelation enveloppe:'+str(np.linalg.norm(f0_true - f0_autocorr_env, ord=1)/n))
print('erreur moyenne L2 de la méthode par autocorrelation enveloppe:'+str(np.linalg.norm(f0_true - f0_autocorr_env, ord=2)/np.sqrt(n)))
print('\n')
print('Avec une erreur de '+str(error)+'%')
print("taux bonne classif méthode naive FFT "+str(pourcent_bonne_classif(f0_fft,f0_true,error)))
print("taux bonne classif méthode autocorrelation "+str(pourcent_bonne_classif(f0_autocorr,f0_true,error)))
print("taux bonne classif méthode autocorrelation filtrée "+str(pourcent_bonne_classif(f0_autocorr_vib,f0_true,error)))
print("taux bonne classif méthode autocorrelation enveloppe "+str(pourcent_bonne_classif(f0_autocorr_env,f0_true,error)))
print('\n')
print("On constate que la méthode d'autocorrélation est plus précise.")
print('En effet, la méthode naive a du mal a détecter les vibratos légers.')
print('Globalement les deux méthodes sont moins efficaces pour détecter la fondamentale')
print('Finalement dans ce cas, ça vaut le coup d uiliser un filtre passe bas: même si on perd de la rapidité en régime transitoire')
print("on la regagne pendant les vibratos")
      





#SPECTRO-CORRELOGRAMME VOIX

Tmin = int(samplerate / Fmax)
Tmax = int(samplerate / Fmin)
# Autocorrélogramme

# Initialiser le tableau pour stocker les autocorrélations
autocorr_matrix = []

# Calculer l'autocorrélation sur chaque fenêtre
for start in range(0, len(x) - window_size, hop_size):
    end = start + window_size
    window = x[start:end]
    windowed_signal = window * np.hamming(len(window))  # Appliquer une fenêtre de Hamming

    # Calculer l'autocorrélation
    autocorr = np.correlate(windowed_signal, windowed_signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / np.max(autocorr)


    # Ajouter au tableau
    autocorr_matrix.append(autocorr)

# Convertir la matrice en numpy array
autocorr_matrix = np.array(autocorr_matrix)
# Calcul des lags correspondants aux fréquences fondamentales détectées
lags_detected = samplerate / np.array(f0_autocorr) * T
lags_detected[f0_autocorr == 0] = 0  # Remplacer les valeurs invalides par 0


# Créer l'échelle de temps et de lag pour le spectrogramme
time_axis = np.arange(len(autocorr_matrix)) * hop_size / samplerate
lag_axis = np.arange(len(autocorr_matrix[0])) * T

# Afficher le spectrogramme avec les lags superposés
plt.figure(figsize=(12, 8))
plt.imshow(
    autocorr_matrix.T,  # bas en haut
    extent=[time_axis[0], time_axis[-1], lag_axis[0], lag_axis[-1]],
    aspect='auto',
    cmap='viridis',
    origin='lower'
)
plt.colorbar(label='Amplitude normalisée')
plt.xlabel('Temps (s)')
plt.ylabel('Lag (s)')
plt.title('Spectrogramme basé sur l\'autocorrélation (voix)')

# Superposer les lags détectés (éliminer les zéros pour éviter les artefacts)
plt.scatter(time_axis, lags_detected, color='red', s=10, label='Lags détectés')

plt.legend()
plt.tight_layout()





#SPECTRO-CORRELOGRAMME FLUTE

# Construire le chemin relatif vers le fichier audio
script_dir = os.path.dirname(os.path.abspath(__file__))  # Répertoire du script
audio_path = os.path.join(script_dir, 'audio_files', 'fluteircam.wav')

# Lire le fichier audio
data, samplerate = sf.read(audio_path)
x = data
T = 1 / samplerate

# Paramètres
window_size = 0.02 # Taille de la fenêtre en secondes
hop_size = 0.01  # Décalage entre les fenêtres (en secondes)
treshold = 0.003 # Seuil en dessous duquel c'est du bruit
Fmin = librosa.note_to_hz('C3')
Fmax = librosa.note_to_hz('C6')

#conversion en nbre d'échantillons
window_size = int(window_size / T)
hop_size = int(hop_size / T)

# Calcul avec chaque méthode
times, f0_fft = naive_fft_fundamental(x, samplerate, window_size, hop_size,treshold)
times, f0_autocorr = autocorrelation_fundamental(x, samplerate, window_size, hop_size, Fmin, Fmax,treshold)

Tmin = int(samplerate / Fmax)
Tmax = int(samplerate / Fmin)

# Initialiser le tableau pour stocker les autocorrélations
autocorr_matrix = []

# Calculer l'autocorrélation sur chaque fenêtre
for start in range(0, len(x) - window_size, hop_size):
    end = start + window_size
    window = x[start:end]
    windowed_signal = window * np.hamming(len(window))  # Appliquer une fenêtre de Hamming

    # Calculer l'autocorrélation
    autocorr = np.correlate(windowed_signal, windowed_signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / np.max(autocorr)


    # Ajouter au tableau
    autocorr_matrix.append(autocorr)

# Convertir la matrice en numpy array
autocorr_matrix = np.array(autocorr_matrix)
# Calcul des lags correspondants aux fréquences fondamentales détectées
lags_detected = samplerate / np.array(f0_autocorr) * T
lags_detected[f0_autocorr == 0] = 0  # Remplacer les valeurs invalides par 0


# Créer l'échelle de temps et de lag pour le spectrogramme
time_axis = np.arange(len(autocorr_matrix)) * hop_size / samplerate
lag_axis = np.arange(len(autocorr_matrix[0])) * T

# Afficher le spectrogramme avec les lags superposés
plt.figure(figsize=(12, 8))
plt.imshow(
    autocorr_matrix.T,  # bas en haut
    extent=[time_axis[0], time_axis[-1], lag_axis[0], lag_axis[-1]],
    aspect='auto',
    cmap='viridis',
    origin='lower'
)
plt.colorbar(label='Amplitude normalisée')
plt.xlabel('Temps (s)')
plt.ylabel('Lag (s)')
plt.title('Spectrogramme basé sur l\'autocorrélation (flute)')

# Superposer les lags détectés (éliminer les zéros pour éviter les artefacts)
plt.scatter(time_axis, lags_detected, color='red', s=10, label='Lags détectés')

plt.legend()
plt.show()