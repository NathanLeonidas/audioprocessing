# 🎵 AUDIOPROCESSING - Analyse et Traitement du Signal Audio

Ce projet implémente différentes méthodes d'analyse et de traitement du signal audio, en utilisant des techniques paramétriques et non paramétriques.

## Structure du Projet

AUDIOPROCESSING/   
│── audio_files/ - Contient les fichiers audio à analyser  
│── documentation_cours/ - Documents et fichiers de référence  
│── images/ - Quelques graphiques et visualisations générés  
│── Partie_1/ - Recherche de la fondamentale  
│── Partie_2/ - Séparation de sinusoïdes    
│── Partie_3/ - Etude de myson  
│── requirements.txt - Liste des dépendances Python  
│── README.md - Documentation du projet  

## Installation et Exécution

1. **Cloner le dépôt**  
   `git clone https://github.com/votre-utilisateur/audioprocessing.git`  
   `cd audioprocessing`  

2. **Créer et activer un environnement virtuel**  
   `python -m venv venv`  
   `source venv/bin/activate` (macOS/Linux)  
   `venv\Scripts\activate` (Windows)  

3. **Installer les dépendances**  
   `pip install -r requirements.txt`  

4. **Exécuter un script**  
   `python Partie_3/3_freq_dominantes.py`  

## Fonctionnalités Principales

- Analyse spectrale avec la FFT complète et la STFT  
- Estimation de la DSP avec MUSIC, Prony, Burg, Levinson-Durbin
- Transformée en ondelettes (CWT) pour une analyse spectrale avancée  
- Visualisation des résultats avec Matplotlib  
- Lecture et traitement des fichiers audio au format WAV  

## Technologies Utilisées

Ce projet repose sur plusieurs bibliothèques Python : NumPy, Librosa, Matplotlib, SciPy, PyWavelets, Soundfile.

---

Auteurs : Paul LEMAIRE, Nathan Lewy 
Date de mise à jour : 30 janvier 2025  
