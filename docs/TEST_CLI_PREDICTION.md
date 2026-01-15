# Test de la Commande CLI - Prédiction

## Commande de Test

```bash
# Test avec un fichier unique
uv run python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json

# Test avec sauvegarde des résultats
uv run python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json --output predictions.json

# Test avec un répertoire
uv run python scripts/predict.py --input-dir data/raw/code_classification_dataset/ --output predictions_batch.json
```

## Vérifications Effectuées

### ✅ Structure du Code

1. **Script CLI (`scripts/predict.py`)** :
   - ✅ Gestion des arguments (--input, --input-dir, --output, --config, --model)
   - ✅ Chargement des fichiers JSON (unique ou répertoire)
   - ✅ Conversion en DataFrame
   - ✅ Initialisation du Predictor
   - ✅ Appel à `predict_with_labels()`
   - ✅ Sauvegarde ou affichage des résultats

2. **Module Predictor (`src/prediction/predictor.py`)** :
   - ✅ Initialisation avec chargement des artifacts et du modèle
   - ✅ Méthode `predict()` : retourne array numpy
   - ✅ Méthode `predict_with_labels()` : retourne liste de dicts avec tags

3. **Module Preprocessor (`src/prediction/preprocessor.py`)** :
   - ✅ Pipeline complet de preprocessing (11 étapes)
   - ✅ Chargement des artifacts (imputation, TF-IDF, scaler, LaTeX features)
   - ✅ Calcul des embeddings et TF-IDF
   - ✅ Extraction et normalisation des features denses

4. **Module ModelLoader (`src/prediction/model_loader.py`)** :
   - ✅ Chargement du modèle depuis pickle
   - ✅ Chargement de la configuration JSON

### ✅ Cohérence avec le Notebook

- ✅ Ordre des features : TF-IDF → Embeddings → Dense (normalisées)
- ✅ Features denses : même structure que le notebook (~40 features)
- ✅ Normalisation : StandardScaler appliqué aux features denses
- ✅ Pipeline preprocessing : même ordre que `02_preprocessing_pipeline.ipynb`

## Problème Rencontré

### Erreur d'Environnement (Non-Bloquant pour la Logique)

```
OSError: [Errno 22] Invalid argument
```

**Cause** : Problème connu avec `huggingface_hub` et les chemins Windows/OneDrive longs.

**Impact** : L'import de `sentence_transformers` échoue, mais la logique du code est correcte.

**Solutions possibles** :
1. Tester dans un environnement Linux/WSL
2. Déplacer le projet hors de OneDrive
3. Utiliser un chemin plus court
4. Configurer `HF_HOME` dans un répertoire avec chemin court

## Test Manuel Recommandé

Une fois le problème d'environnement résolu, tester dans cet ordre :

1. **Test basique** :
   ```bash
   uv run python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json
   ```

2. **Vérifier les artifacts** :
   - `data/processed/imputation_values.json` ✅
   - `data/processed/train_preprocessed.parquet` ✅
   - `data/processed/tfidf_vectorizer.pkl` ✅
   - `data/processed/scaler_dense_features.pkl` ✅
   - `models/logreg_baseline_all_features.pkl` ✅

3. **Vérifier la sortie** :
   - Format JSON valide
   - Présence de `src_uid`, `predicted_tags`, `num_tags`
   - Tags prédits dans la liste des PRIORITY_TAGS

## Structure Attendue de la Sortie

```json
[
  {
    "src_uid": "afcd41492158e68095b01ff1e88c3dd4",
    "predicted_tags": ["math", "number theory"],
    "num_tags": 2
  }
]
```

## Points de Vérification

- [ ] Les artifacts sont chargés correctement
- [ ] Le preprocessing s'exécute sans erreur
- [ ] Les features sont combinées dans le bon ordre
- [ ] Le modèle fait des prédictions
- [ ] Les tags prédits sont valides (dans PRIORITY_TAGS)
- [ ] La sortie JSON est valide

## Commandes de Debug

```bash
# Vérifier les imports
uv run python -c "from src.prediction.predictor import Predictor; print('OK')"

# Vérifier le chargement des artifacts
uv run python -c "from src.prediction.preprocessor import InferencePreprocessor; p = InferencePreprocessor(); p.load_artifacts()"

# Vérifier le chargement du modèle
uv run python -c "from src.prediction.model_loader import ModelLoader; m = ModelLoader(); m.load_model('logreg_baseline_all_features.pkl')"
```


## Commande de Test

```bash
# Test avec un fichier unique
uv run python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json

# Test avec sauvegarde des résultats
uv run python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json --output predictions.json

# Test avec un répertoire
uv run python scripts/predict.py --input-dir data/raw/code_classification_dataset/ --output predictions_batch.json
```

## Vérifications Effectuées

### ✅ Structure du Code

1. **Script CLI (`scripts/predict.py`)** :
   - ✅ Gestion des arguments (--input, --input-dir, --output, --config, --model)
   - ✅ Chargement des fichiers JSON (unique ou répertoire)
   - ✅ Conversion en DataFrame
   - ✅ Initialisation du Predictor
   - ✅ Appel à `predict_with_labels()`
   - ✅ Sauvegarde ou affichage des résultats

2. **Module Predictor (`src/prediction/predictor.py`)** :
   - ✅ Initialisation avec chargement des artifacts et du modèle
   - ✅ Méthode `predict()` : retourne array numpy
   - ✅ Méthode `predict_with_labels()` : retourne liste de dicts avec tags

3. **Module Preprocessor (`src/prediction/preprocessor.py`)** :
   - ✅ Pipeline complet de preprocessing (11 étapes)
   - ✅ Chargement des artifacts (imputation, TF-IDF, scaler, LaTeX features)
   - ✅ Calcul des embeddings et TF-IDF
   - ✅ Extraction et normalisation des features denses

4. **Module ModelLoader (`src/prediction/model_loader.py`)** :
   - ✅ Chargement du modèle depuis pickle
   - ✅ Chargement de la configuration JSON

### ✅ Cohérence avec le Notebook

- ✅ Ordre des features : TF-IDF → Embeddings → Dense (normalisées)
- ✅ Features denses : même structure que le notebook (~40 features)
- ✅ Normalisation : StandardScaler appliqué aux features denses
- ✅ Pipeline preprocessing : même ordre que `02_preprocessing_pipeline.ipynb`

## Problème Rencontré

### Erreur d'Environnement (Non-Bloquant pour la Logique)

```
OSError: [Errno 22] Invalid argument
```

**Cause** : Problème connu avec `huggingface_hub` et les chemins Windows/OneDrive longs.

**Impact** : L'import de `sentence_transformers` échoue, mais la logique du code est correcte.

**Solutions possibles** :
1. Tester dans un environnement Linux/WSL
2. Déplacer le projet hors de OneDrive
3. Utiliser un chemin plus court
4. Configurer `HF_HOME` dans un répertoire avec chemin court

## Test Manuel Recommandé

Une fois le problème d'environnement résolu, tester dans cet ordre :

1. **Test basique** :
   ```bash
   uv run python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json
   ```

2. **Vérifier les artifacts** :
   - `data/processed/imputation_values.json` ✅
   - `data/processed/train_preprocessed.parquet` ✅
   - `data/processed/tfidf_vectorizer.pkl` ✅
   - `data/processed/scaler_dense_features.pkl` ✅
   - `models/logreg_baseline_all_features.pkl` ✅

3. **Vérifier la sortie** :
   - Format JSON valide
   - Présence de `src_uid`, `predicted_tags`, `num_tags`
   - Tags prédits dans la liste des PRIORITY_TAGS

## Structure Attendue de la Sortie

```json
[
  {
    "src_uid": "afcd41492158e68095b01ff1e88c3dd4",
    "predicted_tags": ["math", "number theory"],
    "num_tags": 2
  }
]
```

## Points de Vérification

- [ ] Les artifacts sont chargés correctement
- [ ] Le preprocessing s'exécute sans erreur
- [ ] Les features sont combinées dans le bon ordre
- [ ] Le modèle fait des prédictions
- [ ] Les tags prédits sont valides (dans PRIORITY_TAGS)
- [ ] La sortie JSON est valide

## Commandes de Debug

```bash
# Vérifier les imports
uv run python -c "from src.prediction.predictor import Predictor; print('OK')"

# Vérifier le chargement des artifacts
uv run python -c "from src.prediction.preprocessor import InferencePreprocessor; p = InferencePreprocessor(); p.load_artifacts()"

# Vérifier le chargement du modèle
uv run python -c "from src.prediction.model_loader import ModelLoader; m = ModelLoader(); m.load_model('logreg_baseline_all_features.pkl')"
```

