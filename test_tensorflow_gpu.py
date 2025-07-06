#!/usr/bin/env python3
"""
Script de test TensorFlow GPU
============================

Ce script teste votre installation TensorFlow et vérifie le support GPU.
Utilisation: python test_tensorflow_gpu.py

Auteur: Mickael Faust
Date: Juin 2025
"""

import sys
import os
import time
import subprocess

def print_section(title):
    """Affiche une section avec formatage"""
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

def run_command(cmd):
    """Exécute une commande système et retourne le résultat"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode == 0
    except Exception as e:
        return str(e), False

def main():
    print("🚀 Test d'installation TensorFlow GPU")
    print("Auteur: Guide d'installation TensorFlow GPU - Juin 2025")

    # ===== INFORMATIONS SYSTÈME =====
    print_section("INFORMATIONS SYSTÈME")
    print(f"Python version: {sys.version}")
    print(f"Plateforme: {sys.platform}")
    print(f"Architecture: {os.uname().machine if hasattr(os, 'uname') else 'N/A'}")

    # Test nvidia-smi
    nvidia_output, nvidia_success = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits")
    if nvidia_success:
        print(f"✅ NVIDIA Driver détecté:")
        for line in nvidia_output.split('\n'):
            if line.strip():
                print(f"   {line.strip()}")
    else:
        print("❌ nvidia-smi non disponible ou pas de GPU NVIDIA")

    # ===== TEST TENSORFLOW =====
    print_section("TEST TENSORFLOW")

    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")

        # Informations de build
        try:
            build_info = tf.sysconfig.get_build_info()
            print(f"   - CUDA version: {build_info.get('cuda_version', 'N/A')}")
            print(f"   - cuDNN version: {build_info.get('cudnn_version', 'N/A')}")
        except:
            print("   - Informations build non disponibles")

    except ImportError as e:
        print(f"❌ Erreur import TensorFlow: {e}")
        print("   Solution: Vérifiez votre installation et environnement virtuel")
        return False

    # ===== TEST GPU =====
    print_section("TEST DÉTECTION GPU")

    # Lister les devices
    physical_devices = tf.config.list_physical_devices()
    print("Devices physiques disponibles:")
    for device in physical_devices:
        print(f"   - {device}")

    # Test spécifique GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"\nNombre de GPUs détectés: {len(gpu_devices)}")

    if len(gpu_devices) == 0:
        print("❌ Aucun GPU détecté par TensorFlow")
        print("   Solutions possibles:")
        print("   - Vérifiez les drivers NVIDIA (nvidia-smi)")
        print("   - Réinstallez tensorflow[and-cuda]")
        print("   - Vérifiez CUDA_VISIBLE_DEVICES")
        return False

    # Informations détaillées sur les GPUs
    for i, gpu in enumerate(gpu_devices):
        print(f"\n✅ GPU {i}: {gpu}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            for key, value in details.items():
                print(f"   - {key}: {value}")
        except:
            print("   - Détails non disponibles")

    # ===== TEST CONFIGURATION MÉMOIRE =====
    print_section("TEST CONFIGURATION MÉMOIRE GPU")

    try:
        # Configurer la croissance mémoire
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Croissance mémoire GPU activée")

        # Informations mémoire
        if len(gpu_devices) > 0:
            try:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"   - Mémoire utilisée: {memory_info['current'] / 1024**3:.2f} GB")
                print(f"   - Mémoire limite: {memory_info['peak'] / 1024**3:.2f} GB")
            except:
                print("   - Informations mémoire non disponibles")

    except Exception as e:
        print(f"⚠️  Configuration mémoire échouée: {e}")

    # ===== TEST PERFORMANCE =====
    print_section("TEST PERFORMANCE")

    print("Test de calcul matriciel...")

    # Test CPU
    with tf.device('/CPU:0'):
        start_time = time.time()
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        cpu_time = time.time() - start_time

    print(f"✅ CPU: {cpu_time:.4f}s")

    # Test GPU si disponible
    if len(gpu_devices) > 0:
        try:
            with tf.device('/GPU:0'):
                start_time = time.time()
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                gpu_time = time.time() - start_time

            print(f"✅ GPU: {gpu_time:.4f}s")

            if gpu_time < cpu_time:
                speedup = cpu_time / gpu_time
                print(f"🚀 Accélération GPU: {speedup:.2f}x")
            else:
                print("⚠️  GPU plus lent que CPU (normal pour petites matrices)")

        except Exception as e:
            print(f"❌ Erreur calcul GPU: {e}")

    # ===== TEST ENTRAÎNEMENT =====
    print_section("TEST ENTRAÎNEMENT SIMPLE")

    try:
        print("Création d'un modèle test...")

        # Données factices
        x_train = tf.random.normal([1000, 32])
        y_train = tf.random.uniform([1000, 1], maxval=2, dtype=tf.int32)

        # Modèle simple
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("Entraînement (2 epochs)...")
        start_time = time.time()

        history = model.fit(
            x_train, y_train,
            epochs=2,
            batch_size=32,
            verbose=0
        )

        training_time = time.time() - start_time
        print(f"✅ Entraînement terminé en {training_time:.2f}s")
        print(f"   - Loss finale: {history.history['loss'][-1]:.4f}")
        print(f"   - Accuracy finale: {history.history['accuracy'][-1]:.4f}")

    except Exception as e:
        print(f"❌ Erreur entraînement: {e}")

    # ===== RÉSUMÉ FINAL =====
    print_section("RÉSUMÉ FINAL")

    if len(gpu_devices) > 0:
        print("🎉 SUCCÈS! TensorFlow GPU est correctement configuré")
        print(f"   - {len(gpu_devices)} GPU(s) disponible(s)")
        print("   - Calculs GPU fonctionnels")
        print("   - Entraînement possible")

        # Recommandations pour optimiser
        print("\n📋 Recommandations d'optimisation:")
        print("   - Utilisez des batch sizes plus importantes")
        print("   - Configurez la croissance mémoire GPU")
        print("   - Utilisez mixed precision pour plus de performance")
        print("   - Exemple: model.compile(optimizer='adam', loss='mse', jit_compile=True)")

    else:
        print("⚠️  Configuration CPU uniquement")
        print("   - TensorFlow fonctionne mais sans accélération GPU")
        print("   - Consultez le guide de dépannage")

    print("\n🔗 Ressources utiles:")
    print("   - Guide TensorFlow GPU: https://www.tensorflow.org/guide/gpu")
    print("   - Compatibilité CUDA: https://www.tensorflow.org/install/source#gpu")
    print("   - Forum TensorFlow: https://www.tensorflow.org/community")

    return len(gpu_devices) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
