# Docker Compose - Guide d'utilisation

Ce fichier `docker-compose.yml` permet d'exécuter facilement les notebooks Jupyter et la CLI de prédiction dans un environnement Docker isolé, avec persistance du cache Hugging Face.

## Services disponibles

### 1. Service Jupyter (notebooks)

Démarre Jupyter Lab pour travailler sur les notebooks.

**Démarrer :**
```bash
docker-compose up jupyter
```

**Accès :**
- URL : http://localhost:8888
- Token : `illuin2024`

**Arrêter :**
```bash
docker-compose down
```

### 2. Service CLI (prédictions)

Service pour exécuter la CLI de prédiction.

**Utilisation :**

#### Prédiction sur un fichier unique
```bash
docker-compose run --rm cli --input data/raw/code_classification_dataset/sample_1.json
```

#### Prédiction sur un répertoire
```bash
docker-compose run --rm cli --input-dir data/raw/code_classification_dataset/ --output predictions.json
```

#### Prédiction avec configuration personnalisée
```bash
docker-compose run --rm cli --input sample.json --config scripts/prediction_config.json --output results.json
```

#### Prédiction avec chemins personnalisés
```bash
docker-compose run --rm cli \
  --input sample.json \
  --model models/logreg_baseline_all_features.pkl \
  --artifacts-dir data/processed \
  --models-dir models \
  --output predictions.json
```

## Volumes montés

- **Projet** : Le répertoire courant est monté dans `/app` (modifications en temps réel)
- **Cache Hugging Face** : Volume persistant `hf_cache` → `/app/.cache/huggingface`
  - Les modèles téléchargés sont conservés entre les exécutions
  - Évite de re-télécharger les modèles à chaque run

## Variables d'environnement

Les variables suivantes sont définies automatiquement :
- `HF_HOME=/app/.cache/huggingface`
- `HF_HUB_CACHE=/app/.cache/huggingface/hub`
- `HF_DATASETS_CACHE=/app/.cache/huggingface/datasets`

## Commandes utiles

### Voir les logs du service Jupyter
```bash
docker-compose logs -f jupyter
```

### Reconstruire l'image
```bash
docker-compose build
```

### Nettoyer les volumes (supprime le cache Hugging Face)
```bash
docker-compose down -v
```

### Exécuter une commande Python arbitraire
```bash
docker-compose run --rm cli python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

### Accéder au shell du conteneur
```bash
docker-compose run --rm cli bash
```

## Avantages

1. **Environnement isolé** : Pas de conflit avec l'environnement local
2. **Cache persistant** : Les modèles Hugging Face sont conservés
3. **Reproductibilité** : Même environnement pour tous
4. **Pas de problème OneDrive** : Fonctionne parfaitement sous Windows
5. **Facilité d'utilisation** : Commandes simples avec docker-compose

## Dépannage

### Le cache ne persiste pas
Vérifiez que le volume `hf_cache` existe :
```bash
docker volume ls | grep hf_cache
```

### Reconstruire après modification des dépendances
```bash
docker-compose build --no-cache
docker-compose up jupyter
```

### Vérifier les variables d'environnement dans le conteneur
```bash
docker-compose run --rm cli env | grep HF_
```

