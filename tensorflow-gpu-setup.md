# Procédure d'installation TensorFlow GPU pour développeurs

## Table des matières
1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis système](#prérequis-système)
3. [Installation par système d'exploitation](#installation-par-système-dexploitation)
   - [Linux/Ubuntu](#linuxubuntu)
   - [Windows WSL2](#windows-wsl2)
   - [macOS](#macos)
4. [Vérification de l'installation](#vérification-de-linstallation)
5. [Dépannage](#dépannage)
6. [Bonnes pratiques](#bonnes-pratiques)

## Vue d'ensemble

Cette procédure vous guide pour configurer un environnement de développement TensorFlow avec support GPU utilisant les dernières versions disponibles (TensorFlow 2.17+). L'installation se fait dans un environnement virtuel isolé pour éviter les conflits de dépendances.

### Versions recommandées (Juin 2025)
- **TensorFlow**: 2.17+ 
- **Python**: 3.9-3.12
- **CUDA**: 12.3+ (pour TensorFlow 2.17)
- **cuDNN**: 8.9+

## Prérequis système

### Matériel requis
- **GPU NVIDIA** avec architecture CUDA 6.0+ (génération Pascal ou plus récente)
- **RAM**: Minimum 8GB, recommandé 16GB+
- **Espace disque**: Minimum 10GB d'espace libre

### Logiciels de base
- **Python 3.9-3.12** installé
- **Git** (optionnel, mais recommandé)
- **Conda** ou **Python venv** pour la gestion d'environnements

---

## Installation par système d'exploitation

## Linux/Ubuntu

### Étape 1: Préparation de l'environnement

#### Option A: Avec Conda (Recommandé)
```bash
# Installer Miniconda si pas déjà fait
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Créer l'environnement
conda create --name tf-gpu python=3.11
conda activate tf-gpu
```

#### Option B: Avec venv
```bash
# Créer l'environnement virtuel
python3.11 -m venv tf-gpu-env
source tf-gpu-env/bin/activate

# Mettre à jour pip
pip install --upgrade pip
```

### Étape 2: Installation des drivers NVIDIA
```bash
# Vérifier la version du driver
nvidia-smi

# Si nécessaire, installer/mettre à jour les drivers
sudo apt update
sudo apt install nvidia-driver-535  # ou version plus récente
sudo reboot
```

### Étape 3: Installation TensorFlow avec support GPU
```bash
# Activer l'environnement
conda activate tf-gpu  # ou source tf-gpu-env/bin/activate

# Installation simple avec CUDA intégré
pip install 'tensorflow[and-cuda]'
```

---

## Windows WSL2

### Important: Support GPU natif Windows abandonné
⚠️ **Note critique**: Le support GPU natif pour Windows a été abandonné après TensorFlow 2.10. **WSL2 est désormais la méthode recommandée**.

### Étape 1: Installation WSL2
```powershell
# Dans PowerShell en tant qu'administrateur
wsl --install -d Ubuntu-24.04
```

### Étape 2: Installation des drivers NVIDIA
1. Télécharger et installer les **drivers NVIDIA pour WSL** depuis le site NVIDIA
2. **Ne PAS installer CUDA dans WSL** - les drivers Windows suffisent

### Étape 3: Configuration dans WSL2
```bash
# Entrer dans WSL2
wsl

# Vérifier la détection GPU
nvidia-smi

# Installer Python et venv
sudo apt update
sudo apt install python3.11-venv python3-pip

# Créer l'environnement
python3.11 -m venv tf-gpu
source tf-gpu/bin/activate

# Installation TensorFlow
pip install --upgrade pip
pip install 'tensorflow[and-cuda]'
```

### Alternative Windows Native (TensorFlow ≤2.10 seulement)
```cmd
# Créer environnement conda
conda create --name tf-gpu python=3.10
conda activate tf-gpu

# Installer CUDA et cuDNN via conda
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Installer TensorFlow (version limitée)
pip install "tensorflow<2.11"
```

---

## macOS

### Support GPU sur macOS
- **GPU NVIDIA**: Non supporté nativement depuis macOS 10.14
- **Apple Silicon (M1/M2/M3)**: Utiliser **tensorflow-metal** pour l'accélération GPU

### Installation pour Apple Silicon
```bash
# Créer environnement conda (recommandé pour M1/M2/M3)
conda create --name tf-metal python=3.11
conda activate tf-metal

# Installation TensorFlow optimisé pour macOS
pip install tensorflow-macos
pip install tensorflow-metal

# Vérification
python -c "import tensorflow as tf; print('GPU disponible:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

### Installation CPU uniquement (Intel Mac)
```bash
# Environnement virtuel
python3 -m venv tf-cpu
source tf-cpu/bin/activate

# Installation TensorFlow CPU
pip install --upgrade pip
pip install tensorflow

# Test
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

---

## Vérification de l'installation

### Script de test complet
Créer un fichier `test_tensorflow_gpu.py`:

```python
import tensorflow as tf
import sys

print("=== Test TensorFlow GPU ===")
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

# Vérifier les devices disponibles
print(f"GPU disponibles: {len(tf.config.list_physical_devices('GPU'))}")
print("Devices physiques:", tf.config.list_physical_devices())

# Test de calcul sur GPU
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU détecté!")
    
    # Test de performance simple
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    print("✅ Calcul GPU réussi!")
    
    # Informations détaillées GPU
    gpu_details = tf.config.experimental.get_device_details(
        tf.config.list_physical_devices('GPU')[0]
    )
    print(f"GPU: {gpu_details}")
else:
    print("❌ Aucun GPU détecté")
    print("Mode CPU uniquement")

# Test d'entraînement simple
print("\n=== Test d'entraînement ===")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print("Entraînement d'un modèle test...")
model.fit(x_train[:1000], y_train[:1000], epochs=2, verbose=1)
print("✅ Entraînement terminé avec succès!")
```

Exécuter le test:
```bash
python test_tensorflow_gpu.py
```

---

## Dépannage

### Problèmes courants et solutions

#### 1. GPU non détecté
**Symptôme**: `tf.config.list_physical_devices('GPU')` retourne une liste vide

**Solutions**:
```bash
# Vérifier drivers NVIDIA
nvidia-smi

# Réinstaller TensorFlow
pip uninstall tensorflow
pip install 'tensorflow[and-cuda]'

# Vérifier variables d'environnement (si problème persiste)
echo $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
```

#### 2. Erreurs de compatibilité CUDA/cuDNN
**Solutions**:
```bash
# Vérifier les versions installées
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

# Réinstaller avec version spécifique
pip install tensorflow==2.17.0
```

#### 3. Erreurs de mémoire GPU
**Solutions**:
```python
# Limiter la croissance mémoire GPU
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

#### 4. Performance lente sur macOS M1/M2/M3
**Solutions**:
```bash
# Vérifier tensorflow-metal
pip install --upgrade tensorflow-metal

# Utiliser optimisations spécifiques
export TF_METAL_DEVICE_VERBOSE=1
```

### Commandes de diagnostic

```bash
# Informations système
uname -a
python --version

# Drivers NVIDIA
nvidia-smi

# Variables d'environnement importantes
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
echo $PATH

# Packages installés
pip list | grep -E "(tensorflow|cuda|cudnn)"

# Test détaillé TensorFlow
python -c "
import tensorflow as tf
print('Build info:', tf.sysconfig.get_build_info())
print('Devices:', tf.config.list_physical_devices())
"
```

---

## Bonnes pratiques

### 1. Gestion des environnements
- **Toujours utiliser des environnements virtuels isolés**
- Nommer clairement les environnements (`tf-gpu-prod`, `tf-dev`, etc.)
- Documenter les versions dans un fichier `requirements.txt`

```bash
# Exporter l'environnement
pip freeze > requirements.txt

# Recréer l'environnement
pip install -r requirements.txt
```

### 2. Gestion de la mémoire GPU
```python
# Configuration recommandée en début de script
import tensorflow as tf

# Permettre la croissance dynamique de la mémoire
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
```

### 3. Monitoring et optimisation
```python
# Profiling GPU
tf.profiler.experimental.start('logdir')
# ... votre code d'entraînement ...
tf.profiler.experimental.stop()

# Utilisation mémoire
print("GPU Memory:", tf.config.experimental.get_memory_info('GPU:0'))
```

### 4. Script d'activation automatique
Créer un script `activate_tf_env.sh`:

```bash
#!/bin/bash
# Script d'activation de l'environnement TensorFlow GPU

echo "🚀 Activation environnement TensorFlow GPU"

# Activer l'environnement (adapter selon votre méthode)
source ~/tf-gpu-env/bin/activate  # pour venv
# conda activate tf-gpu  # pour conda

# Vérifications
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "GPU: $(python -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices("GPU")))')"

# Variables d'environnement optionnelles
export TF_CPP_MIN_LOG_LEVEL=2  # Réduire les logs
export CUDA_VISIBLE_DEVICES=0  # Utiliser seulement le premier GPU

echo "✅ Environnement prêt!"
```

### 5. Maintenance et mises à jour
```bash
# Mise à jour régulière (faire une sauvegarde avant)
pip install --upgrade tensorflow

# Nettoyage cache
pip cache purge

# Vérification après mise à jour
python test_tensorflow_gpu.py
```

---

## Notes importantes

1. **Compatibilité**: Toujours vérifier la matrice de compatibilité TensorFlow/CUDA avant installation
2. **Performances**: Les premières exécutions peuvent être plus lentes (compilation JIT)
3. **Mémoire**: Configurer la croissance mémoire GPU pour éviter les erreurs OOM
4. **Mises à jour**: Tester les nouvelles versions dans un environnement séparé
5. **Documentation**: Maintenir un journal des configurations qui fonctionnent

Pour les questions spécifiques ou problèmes non couverts, consulter:
- [Documentation officielle TensorFlow](https://www.tensorflow.org/install)
- [Forums TensorFlow](https://www.tensorflow.org/community)
- [Stack Overflow tag tensorflow-gpu](https://stackoverflow.com/questions/tagged/tensorflow-gpu)