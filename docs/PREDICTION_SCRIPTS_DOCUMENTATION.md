# Documentation - Scripts de Prédiction CLI

## Vue d'ensemble

Ce document décrit l'architecture, le fonctionnement et l'utilisation des scripts de prédiction en ligne de commande (CLI) pour le projet de classification de code.

## Arborescence des fichiers

```
illuin_challenge/
├── scripts/
│   ├── predict.py                    # Script CLI principal
│   ├── prediction_config.json        # Configuration (modèle, artifacts)
│   └── README_PREDICTION.md          # Guide d'utilisation
│
├── src/
│   └── prediction/
│       ├── __init__.py               # Module exports
│       ├── preprocessor.py           # Pipeline de preprocessing pour inference
│       ├── model_loader.py           # Chargement des modèles
│       └── predictor.py              # Orchestration preprocessing + prédiction
│
└── data/processed/                   # Artifacts de preprocessing
    ├── imputation_values.json        # Valeurs d'imputation (train)
    ├── train_preprocessed.parquet    # Pour extraire les noms de features LaTeX
    ├── tfidf_vectorizer.pkl          # Vectorizer TF-IDF (fitté sur train)
    └── scaler_dense_features.pkl     # StandardScaler pour features denses
```

## Architecture

### 1. Module `src/prediction/preprocessor.py`

**Classe : `InferencePreprocessor`**

Responsabilité : Appliquer le pipeline de preprocessing complet sur les données d'inférence, en utilisant les artifacts sauvegardés lors de l'entraînement pour éviter tout data leakage.

**Méthodes principales :**

- `__init__(artifacts_dir, embeddings_model_name)` : Initialise le preprocessor
- `load_artifacts()` : Charge tous les artifacts nécessaires depuis le disque
- `preprocess(df)` : Applique le pipeline complet et retourne les features

**Pipeline de preprocessing (11 étapes) :**

1. **Text pattern cleaning** : Correction des patterns de texte (NoteIN → Note: In, etc.)
2. **Translation** : Traduction vers l'anglais des 4 colonnes textuelles
3. **Numeric variable conversion** : Conversion de `time_limit` en secondes, gestion des valeurs invalides de `difficulty`
4. **Text/LaTeX separation** : Extraction des features LaTeX numériques (nb_blocks, nb_symbols, density, etc.)
5. **LaTeX binary features** : Création des features binaires `has_*` en utilisant les mêmes symboles que le train
6. **Text length features** : Calcul des longueurs de texte, ratios, etc.
7. **Unified document creation** : Création du document unifié (avec et sans LaTeX)
8. **Missing value imputation** : Imputation avec les valeurs médianes du train
9. **Dense features extraction** : Extraction et normalisation des features denses
10. **Embeddings computation** : Calcul des embeddings avec SentenceTransformer
11. **TF-IDF computation** : Transformation TF-IDF avec le vectorizer du train

**Points critiques (pas de data leakage) :**

- ✅ Imputation : Utilise les valeurs médianes calculées sur le train uniquement
- ✅ LaTeX features : Utilise uniquement les symboles sélectionnés lors de l'entraînement
- ✅ TF-IDF : Utilise le vectorizer fitté sur le train uniquement
- ✅ Normalisation : Utilise le StandardScaler fitté sur le train uniquement
- ✅ Embeddings : Même modèle que l'entraînement

### 2. Module `src/prediction/model_loader.py`

**Classe : `ModelLoader`**

Responsabilité : Charger les modèles entraînés et leurs configurations.

**Méthodes principales :**

- `__init__(models_dir)` : Initialise le loader
- `load_model(model_path)` : Charge un modèle depuis un fichier pickle
- `load_config(config_path)` : Charge la configuration depuis un fichier JSON

### 3. Module `src/prediction/predictor.py`

**Classe : `Predictor`**

Responsabilité : Orchestrer le preprocessing et la prédiction pour fournir une interface simple.

**Méthodes principales :**

- `__init__(artifacts_dir, models_dir, embeddings_model_name)` : Initialise le predictor
- `initialize(model_path)` : Charge les artifacts et le modèle
- `predict(df, return_proba)` : Fait les prédictions (retourne array numpy)
- `predict_with_labels(df, priority_tags)` : Fait les prédictions avec noms de labels (retourne liste de dicts)

**Combinaison des features :**

L'ordre de combinaison est identique à l'entraînement :
1. TF-IDF (sparse matrix)
2. Embeddings (converti en sparse pour stacking)
3. Dense features (converti en sparse pour stacking)

Résultat : `X_full = hstack([X_tfidf, X_embeddings_sparse, X_dense_sparse])`

### 4. Script CLI `scripts/predict.py`

**Fonctionnalités :**

- ✅ **Gestion d'un fichier unique** : `--input sample.json`
- ✅ **Gestion d'un répertoire** : `--input-dir data/predictions/` (charge tous les `.json`)
- ✅ **Sortie vers fichier** : `--output predictions.json`
- ✅ **Sortie vers stdout** : Par défaut si `--output` non spécifié
- ✅ **Configuration personnalisée** : `--config custom_config.json`
- ✅ **Chemins personnalisés** : `--artifacts-dir`, `--models-dir`, `--model`

**Fonctions utilitaires :**

- `load_input_files(input_path)` : Charge un fichier unique ou tous les fichiers d'un répertoire
- `save_predictions(predictions, output_path)` : Sauvegarde les prédictions en JSON

**Format d'entrée :**

Fichier(s) JSON avec la même structure que les données d'entraînement :
```json
{
  "prob_desc_description": "...",
  "prob_desc_input_spec": "...",
  "prob_desc_output_spec": "...",
  "prob_desc_notes": "...",
  "difficulty": 1200,
  "prob_desc_time_limit": "3 seconds",
  "src_uid": "...",
  ...
}
```

**Format de sortie :**

Liste de dictionnaires avec les prédictions :
```json
[
  {
    "src_uid": "sample_1",
    "predicted_tags": ["math", "number theory"],
    "num_tags": 2
  },
  {
    "src_uid": "sample_2",
    "predicted_tags": ["graphs", "trees"],
    "num_tags": 2
  }
]
```

## Gestion des multiples fichiers

**Oui, la gestion des multiples fichiers est implémentée :**

1. **Via `--input-dir`** : Le script charge automatiquement tous les fichiers `.json` d'un répertoire
2. **Via `load_input_files()`** : La fonction détecte si le chemin est un fichier ou un répertoire
3. **Traitement en batch** : Tous les fichiers sont convertis en un seul DataFrame et traités ensemble par le preprocessor
4. **Prédictions groupées** : Les prédictions sont faites en une seule passe pour tous les échantillons

**Avantages du traitement en batch :**

- Efficacité : Les embeddings et TF-IDF sont calculés en batch
- Cohérence : Tous les échantillons utilisent les mêmes artifacts
- Performance : Meilleure utilisation de la mémoire et du CPU

## Flux d'exécution

```
1. Chargement des fichiers d'entrée
   └─> load_input_files() → Liste de dicts

2. Conversion en DataFrame
   └─> pd.DataFrame(input_data)

3. Initialisation du Predictor
   └─> Predictor.initialize()
       ├─> InferencePreprocessor.load_artifacts()
       │   ├─> Charge imputation_values.json
       │   ├─> Charge train_preprocessed.parquet (pour has_* features)
       │   ├─> Charge tfidf_vectorizer.pkl
       │   ├─> Charge scaler_dense_features.pkl
       │   └─> Charge SentenceTransformer model
       └─> ModelLoader.load_model()
           └─> Charge le modèle pickle

4. Preprocessing
   └─> InferencePreprocessor.preprocess(df)
       ├─> 11 étapes de preprocessing
       └─> Retourne (df_processed, X_tfidf, X_embeddings, X_dense)

5. Combinaison des features
   └─> hstack([X_tfidf, X_embeddings, X_dense])

6. Prédiction
   └─> model.predict(X_full) ou model.predict_proba(X_full)

7. Formatage des résultats
   └─> predict_with_labels() → Liste de dicts avec tags

8. Sauvegarde/Affichage
   └─> save_predictions() ou print vers stdout
```

## Exemples d'utilisation

### Exemple 1 : Fichier unique
```bash
python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json
```

### Exemple 2 : Répertoire complet
```bash
python scripts/predict.py --input-dir data/raw/predictions/ --output results.json
```

### Exemple 3 : Configuration personnalisée
```bash
python scripts/predict.py \
  --input sample.json \
  --config custom_config.json \
  --output predictions.json
```

### Exemple 4 : Chemins personnalisés
```bash
python scripts/predict.py \
  --input sample.json \
  --model models/best_model.pkl \
  --artifacts-dir custom/processed/ \
  --models-dir custom/models/
```

## Dépendances et artifacts requis

### Artifacts de preprocessing (dans `data/processed/`) :

1. **`imputation_values.json`** : Obligatoire
   - Contient les valeurs médianes pour `difficulty` et `time_limit_seconds`
   - Format : `{"difficulty": 1700.0, "time_limit_seconds": 2.0}`

2. **`train_preprocessed.parquet`** : Obligatoire
   - Utilisé pour extraire la liste des features `has_*` créées lors de l'entraînement
   - Permet de garantir la cohérence des features entre train et inference

3. **`tfidf_vectorizer.pkl`** : Recommandé
   - Vectorizer TF-IDF fitté sur le train
   - Si absent, les features TF-IDF ne seront pas calculées

4. **`scaler_dense_features.pkl`** : Recommandé
   - StandardScaler pour normaliser les features denses
   - Si absent, les features denses ne seront pas normalisées

### Modèle entraîné (dans `models/`) :

- Fichier pickle contenant le modèle (OneVsRestClassifier avec le modèle de base)
- Doit être compatible avec les features produites par le preprocessor

## Vérifications et tests recommandés

### 1. Vérification des artifacts
```python
from pathlib import Path
artifacts_dir = Path("data/processed")
required = [
    "imputation_values.json",
    "train_preprocessed.parquet",
    "tfidf_vectorizer.pkl",
    "scaler_dense_features.pkl"
]
for artifact in required:
    assert (artifacts_dir / artifact).exists(), f"Missing: {artifact}"
```

### 2. Test du preprocessing
```python
from src.prediction.preprocessor import InferencePreprocessor
import pandas as pd

preprocessor = InferencePreprocessor()
preprocessor.load_artifacts()

# Test avec un échantillon
df_test = pd.DataFrame([{...}])  # Un échantillon de test
df_proc, X_tfidf, X_emb, X_dense = preprocessor.preprocess(df_test)

assert X_emb.shape[0] == 1
assert X_emb.shape[1] == 384  # Dimension des embeddings all-MiniLM-L6-v2
```

### 3. Test de la prédiction complète
```python
from src.prediction.predictor import Predictor
import pandas as pd

predictor = Predictor()
predictor.initialize(model_path="models/best_model.pkl")

df_test = pd.DataFrame([{...}])  # Échantillon de test
predictions = predictor.predict_with_labels(df_test)

assert len(predictions) == 1
assert "predicted_tags" in predictions[0]
```

## Points d'attention

1. **Ordre des features** : L'ordre de combinaison (TF-IDF, embeddings, dense) doit être identique à l'entraînement
2. **Features LaTeX** : Les features `has_*` doivent être exactement les mêmes que celles du train
3. **Normalisation** : Les features denses doivent être normalisées avec le même scaler que le train
4. **Embeddings** : Le modèle d'embeddings doit être le même (all-MiniLM-L6-v2)
5. **TF-IDF** : Le vectorizer doit être exactement celui fitté sur le train

## Améliorations futures possibles

1. **Gestion des erreurs** : Meilleure gestion des cas où certains artifacts sont manquants
2. **Validation des inputs** : Vérification de la structure des fichiers JSON d'entrée
3. **Logging** : Ajout d'un système de logging pour tracer les étapes
4. **Tests unitaires** : Ajout de tests pour chaque module
5. **Support batch processing** : Traitement par lots pour très gros volumes
6. **Cache des embeddings** : Mise en cache des embeddings calculés pour éviter les recalculs


## Vue d'ensemble

Ce document décrit l'architecture, le fonctionnement et l'utilisation des scripts de prédiction en ligne de commande (CLI) pour le projet de classification de code.

## Arborescence des fichiers

```
illuin_challenge/
├── scripts/
│   ├── predict.py                    # Script CLI principal
│   ├── prediction_config.json        # Configuration (modèle, artifacts)
│   └── README_PREDICTION.md          # Guide d'utilisation
│
├── src/
│   └── prediction/
│       ├── __init__.py               # Module exports
│       ├── preprocessor.py           # Pipeline de preprocessing pour inference
│       ├── model_loader.py           # Chargement des modèles
│       └── predictor.py              # Orchestration preprocessing + prédiction
│
└── data/processed/                   # Artifacts de preprocessing
    ├── imputation_values.json        # Valeurs d'imputation (train)
    ├── train_preprocessed.parquet    # Pour extraire les noms de features LaTeX
    ├── tfidf_vectorizer.pkl          # Vectorizer TF-IDF (fitté sur train)
    └── scaler_dense_features.pkl     # StandardScaler pour features denses
```

## Architecture

### 1. Module `src/prediction/preprocessor.py`

**Classe : `InferencePreprocessor`**

Responsabilité : Appliquer le pipeline de preprocessing complet sur les données d'inférence, en utilisant les artifacts sauvegardés lors de l'entraînement pour éviter tout data leakage.

**Méthodes principales :**

- `__init__(artifacts_dir, embeddings_model_name)` : Initialise le preprocessor
- `load_artifacts()` : Charge tous les artifacts nécessaires depuis le disque
- `preprocess(df)` : Applique le pipeline complet et retourne les features

**Pipeline de preprocessing (11 étapes) :**

1. **Text pattern cleaning** : Correction des patterns de texte (NoteIN → Note: In, etc.)
2. **Translation** : Traduction vers l'anglais des 4 colonnes textuelles
3. **Numeric variable conversion** : Conversion de `time_limit` en secondes, gestion des valeurs invalides de `difficulty`
4. **Text/LaTeX separation** : Extraction des features LaTeX numériques (nb_blocks, nb_symbols, density, etc.)
5. **LaTeX binary features** : Création des features binaires `has_*` en utilisant les mêmes symboles que le train
6. **Text length features** : Calcul des longueurs de texte, ratios, etc.
7. **Unified document creation** : Création du document unifié (avec et sans LaTeX)
8. **Missing value imputation** : Imputation avec les valeurs médianes du train
9. **Dense features extraction** : Extraction et normalisation des features denses
10. **Embeddings computation** : Calcul des embeddings avec SentenceTransformer
11. **TF-IDF computation** : Transformation TF-IDF avec le vectorizer du train

**Points critiques (pas de data leakage) :**

- ✅ Imputation : Utilise les valeurs médianes calculées sur le train uniquement
- ✅ LaTeX features : Utilise uniquement les symboles sélectionnés lors de l'entraînement
- ✅ TF-IDF : Utilise le vectorizer fitté sur le train uniquement
- ✅ Normalisation : Utilise le StandardScaler fitté sur le train uniquement
- ✅ Embeddings : Même modèle que l'entraînement

### 2. Module `src/prediction/model_loader.py`

**Classe : `ModelLoader`**

Responsabilité : Charger les modèles entraînés et leurs configurations.

**Méthodes principales :**

- `__init__(models_dir)` : Initialise le loader
- `load_model(model_path)` : Charge un modèle depuis un fichier pickle
- `load_config(config_path)` : Charge la configuration depuis un fichier JSON

### 3. Module `src/prediction/predictor.py`

**Classe : `Predictor`**

Responsabilité : Orchestrer le preprocessing et la prédiction pour fournir une interface simple.

**Méthodes principales :**

- `__init__(artifacts_dir, models_dir, embeddings_model_name)` : Initialise le predictor
- `initialize(model_path)` : Charge les artifacts et le modèle
- `predict(df, return_proba)` : Fait les prédictions (retourne array numpy)
- `predict_with_labels(df, priority_tags)` : Fait les prédictions avec noms de labels (retourne liste de dicts)

**Combinaison des features :**

L'ordre de combinaison est identique à l'entraînement :
1. TF-IDF (sparse matrix)
2. Embeddings (converti en sparse pour stacking)
3. Dense features (converti en sparse pour stacking)

Résultat : `X_full = hstack([X_tfidf, X_embeddings_sparse, X_dense_sparse])`

### 4. Script CLI `scripts/predict.py`

**Fonctionnalités :**

- ✅ **Gestion d'un fichier unique** : `--input sample.json`
- ✅ **Gestion d'un répertoire** : `--input-dir data/predictions/` (charge tous les `.json`)
- ✅ **Sortie vers fichier** : `--output predictions.json`
- ✅ **Sortie vers stdout** : Par défaut si `--output` non spécifié
- ✅ **Configuration personnalisée** : `--config custom_config.json`
- ✅ **Chemins personnalisés** : `--artifacts-dir`, `--models-dir`, `--model`

**Fonctions utilitaires :**

- `load_input_files(input_path)` : Charge un fichier unique ou tous les fichiers d'un répertoire
- `save_predictions(predictions, output_path)` : Sauvegarde les prédictions en JSON

**Format d'entrée :**

Fichier(s) JSON avec la même structure que les données d'entraînement :
```json
{
  "prob_desc_description": "...",
  "prob_desc_input_spec": "...",
  "prob_desc_output_spec": "...",
  "prob_desc_notes": "...",
  "difficulty": 1200,
  "prob_desc_time_limit": "3 seconds",
  "src_uid": "...",
  ...
}
```

**Format de sortie :**

Liste de dictionnaires avec les prédictions :
```json
[
  {
    "src_uid": "sample_1",
    "predicted_tags": ["math", "number theory"],
    "num_tags": 2
  },
  {
    "src_uid": "sample_2",
    "predicted_tags": ["graphs", "trees"],
    "num_tags": 2
  }
]
```

## Gestion des multiples fichiers

**Oui, la gestion des multiples fichiers est implémentée :**

1. **Via `--input-dir`** : Le script charge automatiquement tous les fichiers `.json` d'un répertoire
2. **Via `load_input_files()`** : La fonction détecte si le chemin est un fichier ou un répertoire
3. **Traitement en batch** : Tous les fichiers sont convertis en un seul DataFrame et traités ensemble par le preprocessor
4. **Prédictions groupées** : Les prédictions sont faites en une seule passe pour tous les échantillons

**Avantages du traitement en batch :**

- Efficacité : Les embeddings et TF-IDF sont calculés en batch
- Cohérence : Tous les échantillons utilisent les mêmes artifacts
- Performance : Meilleure utilisation de la mémoire et du CPU

## Flux d'exécution

```
1. Chargement des fichiers d'entrée
   └─> load_input_files() → Liste de dicts

2. Conversion en DataFrame
   └─> pd.DataFrame(input_data)

3. Initialisation du Predictor
   └─> Predictor.initialize()
       ├─> InferencePreprocessor.load_artifacts()
       │   ├─> Charge imputation_values.json
       │   ├─> Charge train_preprocessed.parquet (pour has_* features)
       │   ├─> Charge tfidf_vectorizer.pkl
       │   ├─> Charge scaler_dense_features.pkl
       │   └─> Charge SentenceTransformer model
       └─> ModelLoader.load_model()
           └─> Charge le modèle pickle

4. Preprocessing
   └─> InferencePreprocessor.preprocess(df)
       ├─> 11 étapes de preprocessing
       └─> Retourne (df_processed, X_tfidf, X_embeddings, X_dense)

5. Combinaison des features
   └─> hstack([X_tfidf, X_embeddings, X_dense])

6. Prédiction
   └─> model.predict(X_full) ou model.predict_proba(X_full)

7. Formatage des résultats
   └─> predict_with_labels() → Liste de dicts avec tags

8. Sauvegarde/Affichage
   └─> save_predictions() ou print vers stdout
```

## Exemples d'utilisation

### Exemple 1 : Fichier unique
```bash
python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json
```

### Exemple 2 : Répertoire complet
```bash
python scripts/predict.py --input-dir data/raw/predictions/ --output results.json
```

### Exemple 3 : Configuration personnalisée
```bash
python scripts/predict.py \
  --input sample.json \
  --config custom_config.json \
  --output predictions.json
```

### Exemple 4 : Chemins personnalisés
```bash
python scripts/predict.py \
  --input sample.json \
  --model models/best_model.pkl \
  --artifacts-dir custom/processed/ \
  --models-dir custom/models/
```

## Dépendances et artifacts requis

### Artifacts de preprocessing (dans `data/processed/`) :

1. **`imputation_values.json`** : Obligatoire
   - Contient les valeurs médianes pour `difficulty` et `time_limit_seconds`
   - Format : `{"difficulty": 1700.0, "time_limit_seconds": 2.0}`

2. **`train_preprocessed.parquet`** : Obligatoire
   - Utilisé pour extraire la liste des features `has_*` créées lors de l'entraînement
   - Permet de garantir la cohérence des features entre train et inference

3. **`tfidf_vectorizer.pkl`** : Recommandé
   - Vectorizer TF-IDF fitté sur le train
   - Si absent, les features TF-IDF ne seront pas calculées

4. **`scaler_dense_features.pkl`** : Recommandé
   - StandardScaler pour normaliser les features denses
   - Si absent, les features denses ne seront pas normalisées

### Modèle entraîné (dans `models/`) :

- Fichier pickle contenant le modèle (OneVsRestClassifier avec le modèle de base)
- Doit être compatible avec les features produites par le preprocessor

## Vérifications et tests recommandés

### 1. Vérification des artifacts
```python
from pathlib import Path
artifacts_dir = Path("data/processed")
required = [
    "imputation_values.json",
    "train_preprocessed.parquet",
    "tfidf_vectorizer.pkl",
    "scaler_dense_features.pkl"
]
for artifact in required:
    assert (artifacts_dir / artifact).exists(), f"Missing: {artifact}"
```

### 2. Test du preprocessing
```python
from src.prediction.preprocessor import InferencePreprocessor
import pandas as pd

preprocessor = InferencePreprocessor()
preprocessor.load_artifacts()

# Test avec un échantillon
df_test = pd.DataFrame([{...}])  # Un échantillon de test
df_proc, X_tfidf, X_emb, X_dense = preprocessor.preprocess(df_test)

assert X_emb.shape[0] == 1
assert X_emb.shape[1] == 384  # Dimension des embeddings all-MiniLM-L6-v2
```

### 3. Test de la prédiction complète
```python
from src.prediction.predictor import Predictor
import pandas as pd

predictor = Predictor()
predictor.initialize(model_path="models/best_model.pkl")

df_test = pd.DataFrame([{...}])  # Échantillon de test
predictions = predictor.predict_with_labels(df_test)

assert len(predictions) == 1
assert "predicted_tags" in predictions[0]
```

## Points d'attention

1. **Ordre des features** : L'ordre de combinaison (TF-IDF, embeddings, dense) doit être identique à l'entraînement
2. **Features LaTeX** : Les features `has_*` doivent être exactement les mêmes que celles du train
3. **Normalisation** : Les features denses doivent être normalisées avec le même scaler que le train
4. **Embeddings** : Le modèle d'embeddings doit être le même (all-MiniLM-L6-v2)
5. **TF-IDF** : Le vectorizer doit être exactement celui fitté sur le train

## Améliorations futures possibles

1. **Gestion des erreurs** : Meilleure gestion des cas où certains artifacts sont manquants
2. **Validation des inputs** : Vérification de la structure des fichiers JSON d'entrée
3. **Logging** : Ajout d'un système de logging pour tracer les étapes
4. **Tests unitaires** : Ajout de tests pour chaque module
5. **Support batch processing** : Traitement par lots pour très gros volumes
6. **Cache des embeddings** : Mise en cache des embeddings calculés pour éviter les recalculs

