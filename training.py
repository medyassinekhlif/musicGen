#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maestro ML MIDI - Entraînement + Export Modèles
À lancer une fois sur Colab pour générer les fichiers .pkl
"""

print("="*70)
print("Entraînement Maestro + Export Modèles")
print("="*70 + "\n")

# Step 1: Installations
print("[1/7] Installation des dépendances...\n")
import subprocess
import sys

deps = ['mido', 'scikit-learn', 'numpy', 'requests', 'tqdm']
for lib in deps:
    try:
        __import__(lib.replace('-', '_'))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', lib])

print("Dépendances OK\n")

# Step 2: Imports
import os
import pickle
import numpy as np
from io import BytesIO
from urllib.request import urlopen
from tqdm import tqdm
import zipfile
from mido import MidiFile
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Step 3: Télécharger Maestro v2.0.0 (57 MB)
print("[2/7] Téléchargement Maestro v2.0.0 (57 MB)...")
print(" Attendez 30-90 secondes...\n")

maestro_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
zip_path = "maestro.zip"
extract_dir = "maestro_data"

try:
    # Télécharger
    response = urlopen(maestro_url, timeout=120)
    total_size = int(response.headers.get('content-length', 0))
    
    chunk_size = 1024 * 1024
    downloaded = 0
    
    with open(zip_path, 'wb') as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = (downloaded / total_size) * 100
                mb = downloaded / 1024 / 1024
                print(f" {pct:.0f}% - {mb:.1f} MB", end='\r')
    
    print(f"\nTéléchargement terminé\n")
    
    # Extraire
    print("[3/7] Extraction des fichiers MIDI...\n")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    
    # Trouver les MIDI
    midi_paths = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.midi'):
                midi_paths.append(os.path.join(root, file))
    
    print(f"{len(midi_paths)} fichiers MIDI trouvés\n")

except Exception as e:
    print(f"Erreur: {e}\n")
    print("Utilisation de données synthétiques...\n")
    midi_paths = []

# Step 4: Extraire les features
print("[4/7] Extraction des features...")

def extract_midi_features(midi_path):
    """Extrait features d'un fichier MIDI"""
    try:
        midi = MidiFile(midi_path)
        notes_data = []
        
        for track in midi.tracks:
            active = {}
            time = 0
            
            for msg in track:
                time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    active[msg.note] = (msg.velocity, time)
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active:
                        vel, start = active[msg.note]
                        dur = time - start
                        if dur > 0 and 20 <= vel <= 127:
                            notes_data.append({
                                'pitch': msg.note,
                                'velocity': vel,
                                'duration': dur
                            })
                        del active[msg.note]
        
        if len(notes_data) < 10:
            return None
        
        # Créer features
        X = []
        y_vel = []
        y_dur = []
        
        for i, note in enumerate(notes_data):
            p = note['pitch']
            d = note['duration']
            v = note['velocity']
            
            prev_p = notes_data[i-1]['pitch'] if i > 0 else p
            next_p = notes_data[i+1]['pitch'] if i < len(notes_data)-1 else p
            
            pitch_interval = abs(p - prev_p)
            seq_pos = i / len(notes_data)
            local_density = sum(1 for j in range(max(0, i-6), min(len(notes_data), i+6)))
            
            feat = [p, pitch_interval, prev_p, next_p, seq_pos, local_density, d]
            
            X.append(feat)
            y_vel.append(v)
            y_dur.append(d)
        
        return np.array(X), np.array(y_vel), np.array(y_dur)
    
    except:
        return None

# Traiter les fichiers
X_all = []
y_vel_all = []
y_dur_all = []
success = 0

for midi_path in tqdm(midi_paths[:500], desc="Parsing MIDI"):
    result = extract_midi_features(midi_path)
    if result is not None:
        X, y_vel, y_dur = result
        X_all.append(X)
        y_vel_all.append(y_vel)
        y_dur_all.append(y_dur)
        success += 1

print(f"\n{success} fichiers valides traités\n")

# Concaténer
if X_all:
    X_train = np.vstack(X_all)
    y_vel_train = np.concatenate(y_vel_all)
    y_dur_train = np.concatenate(y_dur_all)
    
    # Réduire les données (important!)
    max_samples = 100000  # Limiter à 100k notes pour entraînement rapide
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
        y_vel_train = y_vel_train[idx]
        y_dur_train = y_dur_train[idx]
        print(f"Dataset réduit à: {len(X_train):,} notes (sur {len(y_vel_all):,} total)\n")
    else:
        print(f"Dataset: {len(X_train):,} notes\n")
else:
    print("Données synthétiques générées\n")
    np.random.seed(42)
    n = 5000
    X_train = np.random.randn(n, 7) * 20 + [60, 5, 60, 65, 0.5, 5, 200]
    y_vel_train = np.random.randint(40, 127, n)
    y_dur_train = np.random.randint(100, 1000, n)
    print(f"Dataset synthétique: {n} notes\n")

# Step 5: Séparer les données Train/Test (80/20)
print("[5/7] Séparation Train/Test (80/20)...\n")

X_train_split, X_test_split, y_vel_train_split, y_vel_test_split, y_dur_train_split, y_dur_test_split = train_test_split(
    X_train, y_vel_train, y_dur_train, test_size=0.2, random_state=42
)

print(f"Train set: {len(X_train_split):,} notes")
print(f"Test set: {len(X_test_split):,} notes\n")

# Normalisation (fit sur train uniquement!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_test_scaled = scaler.transform(X_test_split)

# Step 6: Entraîner les modèles
print("[6/7] Entraînement des modèles Random Forest...\n")

# Modèle vélocité
print(" Entraînement modèle vélocité...")
rf_vel = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
rf_vel.fit(X_train_scaled, y_vel_train_split)

# Prédictions
y_vel_train_pred = rf_vel.predict(X_train_scaled)
y_vel_test_pred = rf_vel.predict(X_test_scaled)

# Métriques train
train_r2_vel = rf_vel.score(X_train_scaled, y_vel_train_split)
train_mae_vel = mean_absolute_error(y_vel_train_split, y_vel_train_pred)
train_rmse_vel = np.sqrt(mean_squared_error(y_vel_train_split, y_vel_train_pred))

# Métriques test
test_r2_vel = rf_vel.score(X_test_scaled, y_vel_test_split)
test_mae_vel = mean_absolute_error(y_vel_test_split, y_vel_test_pred)
test_rmse_vel = np.sqrt(mean_squared_error(y_vel_test_split, y_vel_test_pred))

print(f" Vélocité:")
print(f"   Train → R²={train_r2_vel:.4f} | MAE={train_mae_vel:.2f} | RMSE={train_rmse_vel:.2f}")
print(f"   Test  → R²={test_r2_vel:.4f} | MAE={test_mae_vel:.2f} | RMSE={test_rmse_vel:.2f}\n")

# Modèle durée
print(" Entraînement modèle timing...")
rf_dur = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
rf_dur.fit(X_train_scaled, y_dur_train_split)

# Prédictions
y_dur_train_pred = rf_dur.predict(X_train_scaled)
y_dur_test_pred = rf_dur.predict(X_test_scaled)

# Métriques train
train_r2_dur = rf_dur.score(X_train_scaled, y_dur_train_split)
train_mae_dur = mean_absolute_error(y_dur_train_split, y_dur_train_pred)
train_rmse_dur = np.sqrt(mean_squared_error(y_dur_train_split, y_dur_train_pred))

# Métriques test
test_r2_dur = rf_dur.score(X_test_scaled, y_dur_test_split)
test_mae_dur = mean_absolute_error(y_dur_test_split, y_dur_test_pred)
test_rmse_dur = np.sqrt(mean_squared_error(y_dur_test_split, y_dur_test_pred))

print(f"Timing:")
print(f"   Train → R²={train_r2_dur:.4f} | MAE={train_mae_dur:.2f} | RMSE={train_rmse_dur:.2f}")
print(f"   Test  → R²={test_r2_dur:.4f} | MAE={test_mae_dur:.2f} | RMSE={test_rmse_dur:.2f}\n")

# Step 7: Exporter les modèles
print("[7/7] Export des modèles en fichiers .pkl...\n")

models = {
    'rf_velocity': rf_vel,
    'rf_duration': rf_dur,
    'scaler': scaler
}

with open('models.pkl', 'wb') as f:
    pickle.dump(models, f)

file_size = os.path.getsize('models.pkl') / 1024 / 1024
print(f"models.pkl exporté ({file_size:.2f} MB)\n")

# Infos
info = {
    'training_samples': len(X_train_split),
    'test_samples': len(X_test_split),
    'velocity': {
        'train_r2': float(train_r2_vel),
        'test_r2': float(test_r2_vel),
        'train_mae': float(train_mae_vel),
        'test_mae': float(test_mae_vel),
        'train_rmse': float(train_rmse_vel),
        'test_rmse': float(test_rmse_vel)
    },
    'duration': {
        'train_r2': float(train_r2_dur),
        'test_r2': float(test_r2_dur),
        'train_mae': float(train_mae_dur),
        'test_mae': float(test_mae_dur),
        'train_rmse': float(train_rmse_dur),
        'test_rmse': float(test_rmse_dur)
    },
    'features': ['pitch', 'pitch_interval', 'prev_pitch', 'next_pitch', 'seq_pos', 'local_density', 'duration']
}

with open('model.json', 'w') as f:
    import json
    json.dump(info, f, indent=2)

print(f"model.json créé\n")

# Step 8: Télécharger les fichiers
print("[8/8] Téléchargement des fichiers...\n")

try:
    from google.colab import files
    print("Cliquez sur les fichiers pour télécharger:\n")
    files.download('models.pkl')
    print()
    files.download('model.json')
    print()
except ImportError:
    print("Non sur Colab. Fichiers disponibles:\n")
    print(f" • models.pkl ({file_size:.2f} MB)")
    print(f" • model.json\n")

# Résumé final
print("\n" + "="*70)
print("Entraînement terminé avec succès!")
print("="*70)
print(f"\nDonnées: {len(X_train_split):,} train + {len(X_test_split):,} test")
print(f"\nAccuracy (Test Set):")
print(f"   Vélocité → R²={test_r2_vel:.4f} | MAE={test_mae_vel:.2f} | RMSE={test_rmse_vel:.2f}")
print(f"   Timing   → R²={test_r2_dur:.4f} | MAE={test_mae_dur:.2f} | RMSE={test_rmse_dur:.2f}")
print(f"\nLes modèles peuvent maintenant généraliser sur des données non vues!")
