# Explication du Cache Docker

## Pourquoi le premier build est long ?

### Étape 1 : Build de l'image Docker (première fois uniquement)
- **Téléchargement de l'image Python** : ~30 MB
- **Installation des dépendances** : 5-10 minutes
  - `sentence-transformers` et ses dépendances (transformers, torch, etc.)
  - `scikit-learn`, `xgboost`, `pandas`, etc.
  - Toutes les bibliothèques listées dans `pyproject.toml`
- **Copie des fichiers** : ~177 MB (selon votre projet)

### Étape 2 : Premier run de la CLI
- **Téléchargement du modèle Hugging Face** : ~80-90 MB
  - Modèle : `sentence-transformers/all-MiniLM-L6-v2`
  - Téléchargé dans le volume persistant `hf_cache`

## Pourquoi les fois suivantes sont rapides ?

### Cache Docker (layers)
Docker met en cache chaque étape du Dockerfile :
```
Layer 1: Image Python de base          → ✅ Déjà en cache
Layer 2: Installation de uv          → ✅ Déjà en cache
Layer 3: Copie pyproject.toml         → ✅ Déjà en cache (si pas modifié)
Layer 4: Installation dépendances     → ✅ Déjà en cache (si pas modifié)
Layer 5: Copie du code source         → ⚠️ Rebuild si code modifié
```

**Si vous modifiez seulement le code source** : seules les couches après `COPY . .` sont reconstruites.

**Si vous modifiez `pyproject.toml`** : les dépendances sont réinstallées (mais le cache Docker accélère).

### Volume persistant (modèles Hugging Face)
Le volume `hf_cache` persiste entre les runs :
- **Premier run** : Télécharge le modèle (~80-90 MB, ~30 secondes)
- **Runs suivants** : Utilise le cache, chargement instantané

## Vérifier le cache

### Voir les images Docker
```bash
docker images | grep illuin
```

### Voir les volumes
```bash
docker volume ls | grep hf_cache
```

### Voir la taille du cache Hugging Face
```bash
docker volume inspect illuin_challenge_hf_cache
```

## Optimisations possibles

### 1. Build en arrière-plan (si vous modifiez souvent le code)
```bash
docker-compose build
```

Puis utilisez l'image déjà construite :
```bash
docker-compose run --rm cli --input sample.json
```

### 2. Pré-télécharger le modèle (optionnel)
Créer un script `scripts/preload_model.py` :
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully")
```

Puis :
```bash
docker-compose run --rm cli python scripts/preload_model.py
```

### 3. Utiliser BuildKit (accélère les builds)
```bash
DOCKER_BUILDKIT=1 docker-compose build
```

## Temps estimés

| Action | Première fois | Fois suivantes |
|--------|---------------|----------------|
| Build Docker | 5-10 min | 10-30 sec (si code modifié) |
| Téléchargement modèle | 30-60 sec | 0 sec (cache) |
| Exécution CLI | 5-10 sec | 5-10 sec |

## Quand le cache est invalidé ?

- ✅ **Code source modifié** : Seule la dernière couche est reconstruite
- ✅ **Dépendances modifiées** (`pyproject.toml`) : Réinstallation des dépendances
- ✅ **Dockerfile modifié** : Rebuild à partir de la modification
- ❌ **Modèle Hugging Face** : Toujours en cache (volume persistant)

## Forcer un rebuild complet

Si vous voulez tout reconstruire depuis zéro :
```bash
docker-compose build --no-cache
```

## Conclusion

**Premier build** : Long (5-10 min) mais nécessaire une seule fois.

**Builds suivants** : Rapides grâce au cache Docker (10-30 sec si seulement le code change).

**Runs suivants** : Très rapides, le modèle est en cache.

