



# Documentation Technique : Databricks + Azure Blob Storage pour Fichiers Audio

## 1. Architecture du Système

### 1.1 Vue d'ensemble
Cette solution permet de stocker, traiter et analyser des fichiers audio en utilisant Databricks comme plateforme de traitement et Azure Blob Storage comme système de stockage.

```
[Fichiers Audio] → [Azure Blob Storage] → [Databricks] → [Traitement/Analyse]
```

### 1.2 Composants principaux
- **Azure Blob Storage** : Stockage des fichiers audio bruts et traités
- **Databricks** : Plateforme de traitement et d'analyse
- **Azure Key Vault** : Gestion sécurisée des secrets et clés d'accès
- **Delta Lake** : Stockage des métadonnées et résultats

## 2. Prérequis

### 2.1 Ressources Azure
- Compte Azure actif
- Compte de stockage Azure (Storage Account)
- Workspace Databricks
- Azure Key Vault (optionnel mais recommandé)

### 2.2 Permissions requises
- Contributeur sur le Storage Account
- Accès au Workspace Databricks
- Permissions de lecture/écriture sur les conteneurs Blob

## 3. Configuration du Stockage Azure

### 3.1 Création du Storage Account

```bash
# Via Azure CLI
az storage account create \
  --name mystorageaccount \
  --resource-group myResourceGroup \
  --location westeurope \
  --sku Standard_LRS \
  --kind StorageV2
```

### 3.2 Création du conteneur pour fichiers audio

```bash
# Créer un conteneur pour les fichiers audio
az storage container create \
  --name audio-files \
  --account-name mystorageaccount \
  --public-access off
```

### 3.3 Récupération des clés d'accès

```bash
# Obtenir la clé d'accès
az storage account keys list \
  --account-name mystorageaccount \
  --resource-group myResourceGroup \
  --query '[0].value' \
  --output tsv
```

## 4. Configuration Databricks

### 4.1 Méthode 1 : Montage avec Access Key

```python
# Configuration des credentials
storage_account_name = "mystorageaccount"
storage_account_key = "votre_cle_dacces"
container_name = "audio-files"

# Configuration du montage
mount_point = "/mnt/audio-storage"

# Vérifier si déjà monté
if not any(mount.mountPoint == mount_point for mount in dbutils.fs.mounts()):
    dbutils.fs.mount(
        source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net",
        mount_point = mount_point,
        extra_configs = {
            f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_key
        }
    )
    print(f"Montage réussi à {mount_point}")
else:
    print(f"Le point de montage {mount_point} existe déjà")
```

### 4.2 Méthode 2 : Montage avec SAS Token (Plus sécurisé)

```python
# Génération du SAS Token dans Azure CLI
# az storage container generate-sas --account-name mystorageaccount --name audio-files --permissions rwdl --expiry 2025-12-31

sas_token = "votre_sas_token"
storage_account_name = "mystorageaccount"
container_name = "audio-files"

mount_point = "/mnt/audio-storage"

if not any(mount.mountPoint == mount_point for mount in dbutils.fs.mounts()):
    dbutils.fs.mount(
        source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net",
        mount_point = mount_point,
        extra_configs = {
            f"fs.azure.sas.{container_name}.{storage_account_name}.blob.core.windows.net": sas_token
        }
    )
```

### 4.3 Méthode 3 : Service Principal (Recommandé pour Production)

```python
# Configuration avec Service Principal
configs = {
    "fs.azure.account.auth.type": "OAuth",
    "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    "fs.azure.account.oauth2.client.id": "<application-id>",
    "fs.azure.account.oauth2.client.secret": "<service-credential>",
    "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/<directory-id>/oauth2/token"
}

mount_point = "/mnt/audio-storage"
container_name = "audio-files"
storage_account_name = "mystorageaccount"

if not any(mount.mountPoint == mount_point for mount in dbutils.fs.mounts()):
    dbutils.fs.mount(
        source = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/",
        mount_point = mount_point,
        extra_configs = configs
    )
```

### 4.4 Utilisation d'Azure Key Vault (Meilleure pratique)

```python
# 1. Créer un secret scope dans Databricks
# databricks secrets create-scope --scope azure-key-vault-secrets --scope-backend-type AZURE_KEYVAULT \
# --resource-id /subscriptions/<subscription-id>/resourceGroups/<rg-name>/providers/Microsoft.KeyVault/vaults/<vault-name> \
# --dns-name https://<vault-name>.vault.azure.net/

# 2. Utiliser les secrets dans le code
storage_account_name = "mystorageaccount"
storage_account_key = dbutils.secrets.get(scope="azure-key-vault-secrets", key="storage-account-key")
container_name = "audio-files"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    storage_account_key
)
```

## 5. Opérations sur les Fichiers Audio

### 5.1 Lister les fichiers audio

```python
# Lister tous les fichiers dans le conteneur
files = dbutils.fs.ls("/mnt/audio-storage/")

# Filtrer les fichiers audio
audio_files = [f for f in files if f.name.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

# Afficher les informations
for file in audio_files:
    print(f"Nom: {file.name}, Taille: {file.size} bytes, Path: {file.path}")
```

### 5.2 Charger et lire un fichier audio

```python
# Installation des bibliothèques nécessaires
%pip install pydub librosa soundfile

import librosa
import numpy as np
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType

# Fonction pour charger un fichier audio
def load_audio_file(file_path):
    """
    Charge un fichier audio et retourne le signal audio et le taux d'échantillonnage
    """
    # Convertir le chemin DBFS en chemin système
    local_path = file_path.replace("dbfs:", "/dbfs")
    
    # Charger l'audio
    audio_data, sample_rate = librosa.load(local_path, sr=None)
    
    return audio_data, sample_rate

# Exemple d'utilisation
audio_path = "/mnt/audio-storage/sample_audio.wav"
audio_data, sr = load_audio_file(audio_path)
print(f"Durée: {len(audio_data)/sr:.2f} secondes")
print(f"Taux d'échantillonnage: {sr} Hz")
```

### 5.3 Traitement en batch avec Spark

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Créer un DataFrame avec les chemins des fichiers audio
audio_paths = [f.path for f in dbutils.fs.ls("/mnt/audio-storage/") 
               if f.name.endswith(('.wav', '.mp3'))]

df_paths = spark.createDataFrame([(path,) for path in audio_paths], ["file_path"])

# Fonction UDF pour extraire les métadonnées audio
@udf(returnType=StructType([
    StructField("duration", FloatType(), True),
    StructField("sample_rate", IntegerType(), True),
    StructField("channels", IntegerType(), True)
]))
def extract_audio_metadata(file_path):
    try:
        import librosa
        local_path = file_path.replace("dbfs:", "/dbfs")
        y, sr = librosa.load(local_path, sr=None, mono=False)
        
        duration = librosa.get_duration(y=y, sr=sr)
        channels = 1 if y.ndim == 1 else y.shape[0]
        
        return (float(duration), int(sr), int(channels))
    except Exception as e:
        return (None, None, None)

# Appliquer l'extraction de métadonnées
df_metadata = df_paths.withColumn("metadata", extract_audio_metadata(col("file_path")))
df_metadata = df_metadata.select(
    col("file_path"),
    col("metadata.duration").alias("duration_seconds"),
    col("metadata.sample_rate").alias("sample_rate"),
    col("metadata.channels").alias("channels")
)

display(df_metadata)
```

### 5.4 Extraction de caractéristiques audio

```python
import librosa
import numpy as np

def extract_audio_features(file_path):
    """
    Extrait les caractéristiques audio d'un fichier
    """
    local_path = file_path.replace("dbfs:", "/dbfs")
    y, sr = librosa.load(local_path, sr=22050)
    
    # Extraction des caractéristiques
    features = {
        'duration': len(y) / sr,
        'sample_rate': sr,
        'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist(),
        'spectral_centroid': float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
        'spectral_rolloff': float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean()),
        'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(y).mean()),
        'rms_energy': float(librosa.feature.rms(y=y).mean())
    }
    
    return features

# Exemple d'utilisation
audio_path = "/mnt/audio-storage/sample_audio.wav"
features = extract_audio_features(audio_path)
print(features)
```

### 5.5 Sauvegarder les résultats dans Delta Lake

```python
from pyspark.sql.types import *

# Définir le schéma pour les caractéristiques audio
schema = StructType([
    StructField("file_path", StringType(), True),
    StructField("filename", StringType(), True),
    StructField("duration", FloatType(), True),
    StructField("sample_rate", IntegerType(), True),
    StructField("spectral_centroid", FloatType(), True),
    StructField("spectral_rolloff", FloatType(), True),
    StructField("zero_crossing_rate", FloatType(), True),
    StructField("rms_energy", FloatType(), True),
    StructField("mfcc_features", ArrayType(FloatType()), True),
    StructField("processed_timestamp", StringType(), True)
])

# Traiter tous les fichiers et créer un DataFrame
from datetime import datetime

results = []
for audio_file in audio_files[:10]:  # Traiter les 10 premiers fichiers
    try:
        features = extract_audio_features(audio_file.path)
        results.append({
            'file_path': audio_file.path,
            'filename': audio_file.name,
            'duration': features['duration'],
            'sample_rate': features['sample_rate'],
            'spectral_centroid': features['spectral_centroid'],
            'spectral_rolloff': features['spectral_rolloff'],
            'zero_crossing_rate': features['zero_crossing_rate'],
            'rms_energy': features['rms_energy'],
            'mfcc_features': features['mfcc'],
            'processed_timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Erreur lors du traitement de {audio_file.name}: {e}")

# Créer le DataFrame
df_features = spark.createDataFrame(results, schema=schema)

# Sauvegarder dans Delta Lake
delta_path = "/mnt/audio-storage/delta/audio_features"
df_features.write.format("delta").mode("overwrite").save(delta_path)

print(f"Données sauvegardées dans {delta_path}")
```

## 6. Pipeline de Traitement Automatisé

### 6.1 Notebook de traitement automatique

```python
# Pipeline complet de traitement
class AudioProcessingPipeline:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
    
    def list_audio_files(self):
        """Liste tous les fichiers audio non traités"""
        files = dbutils.fs.ls(self.input_path)
        audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
        return [f for f in files if f.name.lower().endswith(audio_extensions)]
    
    def process_file(self, file_path):
        """Traite un fichier audio individuel"""
        try:
            features = extract_audio_features(file_path)
            return {
                'status': 'success',
                'file_path': file_path,
                'features': features
            }
        except Exception as e:
            return {
                'status': 'error',
                'file_path': file_path,
                'error': str(e)
            }
    
    def run(self):
        """Execute le pipeline complet"""
        print("Démarrage du pipeline de traitement...")
        
        # Lister les fichiers
        audio_files = self.list_audio_files()
        print(f"Fichiers trouvés: {len(audio_files)}")
        
        # Traiter les fichiers
        results = []
        for file in audio_files:
            print(f"Traitement de {file.name}...")
            result = self.process_file(file.path)
            results.append(result)
        
        # Sauvegarder les résultats
        success_results = [r for r in results if r['status'] == 'success']
        if success_results:
            df = spark.createDataFrame(success_results)
            df.write.format("delta").mode("append").save(self.output_path)
            print(f"Traitement terminé: {len(success_results)} fichiers réussis")
        
        return results

# Utilisation du pipeline
pipeline = AudioProcessingPipeline(
    input_path="/mnt/audio-storage/raw/",
    output_path="/mnt/audio-storage/delta/processed_audio"
)
results = pipeline.run()
```

### 6.2 Job Databricks planifié

```python
# Configuration d'un job via Databricks API
# À exécuter depuis la CLI ou un script Python

job_config = {
    "name": "Audio Processing Job",
    "tasks": [
        {
            "task_key": "process_audio_files",
            "notebook_task": {
                "notebook_path": "/Users/your-email@domain.com/AudioProcessingNotebook",
                "base_parameters": {}
            },
            "new_cluster": {
                "spark_version": "13.3.x-scala2.12",
                "node_type_id": "Standard_DS3_v2",
                "num_workers": 2
            }
        }
    ],
    "schedule": {
        "quartz_cron_expression": "0 0 2 * * ?",  # Tous les jours à 2h du matin
        "timezone_id": "Europe/Paris"
    }
}
```

## 7. Bonnes Pratiques

### 7.1 Sécurité
- Toujours utiliser Azure Key Vault pour stocker les secrets
- Utiliser des Service Principals avec des permissions minimales
- Activer le chiffrement au repos et en transit
- Utiliser des SAS tokens avec date d'expiration

### 7.2 Performance
- Utiliser Delta Lake pour un accès optimisé aux métadonnées
- Partitionner les données par date ou catégorie
- Utiliser le cache Databricks pour les données fréquemment accédées
- Optimiser la taille des clusters selon le volume de données

### 7.3 Organisation des données
```
/audio-storage/
├── raw/                 # Fichiers audio bruts
│   ├── 2024/
│   └── 2025/
├── processed/           # Fichiers traités
│   └── converted/
├── delta/              # Tables Delta Lake
│   ├── audio_features/
│   └── audio_metadata/
└── logs/               # Logs de traitement
```

## 8. Monitoring et Logs

```python
# Configuration du logging
import logging
from datetime import datetime

# Créer un logger
logger = logging.getLogger('AudioProcessing')
logger.setLevel(logging.INFO)

# Fonction pour logger dans Azure Blob
def log_to_blob(message, level="INFO"):
    timestamp = datetime.now().isoformat()
    log_entry = f"{timestamp} | {level} | {message}\n"
    
    log_path = f"/mnt/audio-storage/logs/{datetime.now().strftime('%Y-%m-%d')}.log"
    
    try:
        dbutils.fs.put(log_path, log_entry, overwrite=False)
    except:
        pass

# Utilisation
log_to_blob("Pipeline démarré", "INFO")
```

## 9. Dépannage

### 9.1 Problèmes courants

**Erreur de montage**
```python
# Démonter et remonter
dbutils.fs.unmount("/mnt/audio-storage")
# Puis remonter avec la configuration correcte
```

**Erreur d'accès aux fichiers**
```python
# Vérifier les permissions
dbutils.fs.ls("/mnt/audio-storage/")
```

**Erreur de mémoire**
```python
# Augmenter la mémoire du driver
spark.conf.set("spark.driver.memory", "8g")
```

## 10. Références

- [Documentation Azure Blob Storage](https://docs.microsoft.com/azure/storage/blobs/)
- [Documentation Databricks](https://docs.databricks.com/)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Delta Lake Documentation](https://docs.delta.io/)

