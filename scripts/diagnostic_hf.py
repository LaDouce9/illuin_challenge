"""
Script de diagnostic pour Hugging Face / sentence-transformers sous Windows
"""
import sys
import os
import platform
from pathlib import Path

print("=" * 80)
print("DIAGNOSTIC HUGGING FACE / SENTENCE-TRANSFORMERS")
print("=" * 80)

# 1. Informations système
print("\n[1] INFORMATIONS SYSTÈME")
print("-" * 80)
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")
print(f"Chemin du projet (cwd): {Path.cwd()}")
print(f"Chemin du projet (abs): {Path.cwd().resolve()}")

# 2. Variables d'environnement Hugging Face
print("\n[2] VARIABLES D'ENVIRONNEMENT HUGGING FACE")
print("-" * 80)
hf_vars = [
    'HF_HOME',
    'HF_HUB_CACHE',
    'HF_DATASETS_CACHE',
    'TRANSFORMERS_CACHE',
    'HUGGINGFACE_HUB_CACHE',
    'HF_DATASETS_OFFLINE',
    'TRANSFORMERS_OFFLINE'
]

for var in hf_vars:
    value = os.environ.get(var, '(non définie)')
    print(f"{var:30s} = {value}")

# 3. Chemins de cache par défaut
print("\n[3] CHEMINS DE CACHE PAR DÉFAUT")
print("-" * 80)
user_home = Path.home()
default_cache = user_home / ".cache" / "huggingface"
print(f"User home: {user_home}")
print(f"Cache par défaut: {default_cache}")
print(f"Cache existe: {default_cache.exists()}")

if default_cache.exists():
    print(f"Contenu du cache:")
    try:
        for item in list(default_cache.iterdir())[:10]:  # Limiter à 10 items
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
    except Exception as e:
        print(f"  ERREUR lors de la lecture: {e}")

# 4. Test d'import huggingface_hub
print("\n[4] TEST IMPORT huggingface_hub")
print("-" * 80)
try:
    import huggingface_hub
    print(f"[OK] Import reussi")
    print(f"   Version: {huggingface_hub.__version__}")
    print(f"   Chemin: {huggingface_hub.__file__}")
except Exception as e:
    print(f"[ERREUR] Import echoue")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {e}")
    print(f"\n   Stacktrace complète:")
    import traceback
    traceback.print_exc()

# 5. Test d'import sentence_transformers
print("\n[5] TEST IMPORT sentence_transformers")
print("-" * 80)
try:
    from sentence_transformers import SentenceTransformer
    print(f"[OK] Import reussi")
    print(f"   Version: {SentenceTransformer.__module__}")
except Exception as e:
    print(f"[ERREUR] Import echoue")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {e}")
    print(f"\n   Stacktrace complète:")
    import traceback
    traceback.print_exc()

# 6. Vérification des dépendances
print("\n[6] VERSIONS DES DÉPENDANCES")
print("-" * 80)
packages = [
    'huggingface_hub',
    'filelock',
    'transformers',
    'sentence-transformers',
    'torch'
]

for pkg in packages:
    try:
        mod = __import__(pkg.replace('-', '_'))
        version = getattr(mod, '__version__', 'N/A')
        print(f"{pkg:25s} = {version}")
    except ImportError:
        print(f"{pkg:25s} = (non installé)")
    except Exception as e:
        print(f"{pkg:25s} = (erreur: {e})")

print("\n" + "=" * 80)
print("FIN DU DIAGNOSTIC")
print("=" * 80)


"""
import sys
import os
import platform
from pathlib import Path

print("=" * 80)
print("DIAGNOSTIC HUGGING FACE / SENTENCE-TRANSFORMERS")
print("=" * 80)

# 1. Informations système
print("\n[1] INFORMATIONS SYSTÈME")
print("-" * 80)
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")
print(f"Chemin du projet (cwd): {Path.cwd()}")
print(f"Chemin du projet (abs): {Path.cwd().resolve()}")

# 2. Variables d'environnement Hugging Face
print("\n[2] VARIABLES D'ENVIRONNEMENT HUGGING FACE")
print("-" * 80)
hf_vars = [
    'HF_HOME',
    'HF_HUB_CACHE',
    'HF_DATASETS_CACHE',
    'TRANSFORMERS_CACHE',
    'HUGGINGFACE_HUB_CACHE',
    'HF_DATASETS_OFFLINE',
    'TRANSFORMERS_OFFLINE'
]

for var in hf_vars:
    value = os.environ.get(var, '(non définie)')
    print(f"{var:30s} = {value}")

# 3. Chemins de cache par défaut
print("\n[3] CHEMINS DE CACHE PAR DÉFAUT")
print("-" * 80)
user_home = Path.home()
default_cache = user_home / ".cache" / "huggingface"
print(f"User home: {user_home}")
print(f"Cache par défaut: {default_cache}")
print(f"Cache existe: {default_cache.exists()}")

if default_cache.exists():
    print(f"Contenu du cache:")
    try:
        for item in list(default_cache.iterdir())[:10]:  # Limiter à 10 items
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
    except Exception as e:
        print(f"  ERREUR lors de la lecture: {e}")

# 4. Test d'import huggingface_hub
print("\n[4] TEST IMPORT huggingface_hub")
print("-" * 80)
try:
    import huggingface_hub
    print(f"[OK] Import reussi")
    print(f"   Version: {huggingface_hub.__version__}")
    print(f"   Chemin: {huggingface_hub.__file__}")
except Exception as e:
    print(f"[ERREUR] Import echoue")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {e}")
    print(f"\n   Stacktrace complète:")
    import traceback
    traceback.print_exc()

# 5. Test d'import sentence_transformers
print("\n[5] TEST IMPORT sentence_transformers")
print("-" * 80)
try:
    from sentence_transformers import SentenceTransformer
    print(f"[OK] Import reussi")
    print(f"   Version: {SentenceTransformer.__module__}")
except Exception as e:
    print(f"[ERREUR] Import echoue")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {e}")
    print(f"\n   Stacktrace complète:")
    import traceback
    traceback.print_exc()

# 6. Vérification des dépendances
print("\n[6] VERSIONS DES DÉPENDANCES")
print("-" * 80)
packages = [
    'huggingface_hub',
    'filelock',
    'transformers',
    'sentence-transformers',
    'torch'
]

for pkg in packages:
    try:
        mod = __import__(pkg.replace('-', '_'))
        version = getattr(mod, '__version__', 'N/A')
        print(f"{pkg:25s} = {version}")
    except ImportError:
        print(f"{pkg:25s} = (non installé)")
    except Exception as e:
        print(f"{pkg:25s} = (erreur: {e})")

print("\n" + "=" * 80)
print("FIN DU DIAGNOSTIC")
print("=" * 80)

