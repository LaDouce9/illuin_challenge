# Documentation - Preprocessing Pipeline
## Code Classification Challenge

**Date**: Janvier 2026  
**Notebook**: `04_preprocessing_pipeline.ipynb`  
**Module**: `src/utils/preprocessing.py`

---

## Table des Matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture du Pipeline](#2-architecture-du-pipeline)
3. [Étapes Détaillées](#3-étapes-détaillées)
4. [Fonctions Utilitaires](#4-fonctions-utilitaires)
5. [Features Créées](#5-features-créées)
6. [Validation et Qualité](#6-validation-et-qualité)
7. [Utilisation en Production](#7-utilisation-en-production)

---

## 1. Vue d'ensemble

### Objectif

Le pipeline de preprocessing prépare le dataset brut pour la modélisation en appliquant une série de transformations qui :
- Nettoient et normalisent les données textuelles
- Extraient des features discriminantes
- Gèrent les valeurs manquantes
- Encodent la target pour le multi-label learning

### Principes de Conception

**Modularité**: Chaque étape est une fonction indépendante réutilisable

**Traçabilité**: Chaque transformation est documentée et affiche des statistiques

**Production-ready**: Séparation fit/transform pour éviter le data leakage
- **Mode Training** (fit): Calcule les statistiques sur les données d'entraînement
- **Mode Inference** (transform): Applique les statistiques pré-calculées

**Robustesse**: Validation à chaque étape pour détecter les anomalies

---

## 2. Architecture du Pipeline

### Ordre des Opérations

```
1. Text Cleaning (Pattern Correction)
   ↓
2. Translation (English Normalization)
   ↓
3. Near-Duplicate Detection and Removal
   ↓
4. Numeric Variable Conversion
   ↓
5. Text/LaTeX Separation
   ↓
6. LaTeX Feature Extraction
   ↓
7. Text Length Features
   ↓
8. Text Concatenation (Unified Document)
   ↓
9. Target Encoding
   ↓
10. Missing Value Imputation
   ↓
11. Validation and Summary
```

### Justification de l'Ordre

1. **Nettoyage avant traduction**: Les patterns nettoyés améliorent la qualité de la traduction
2. **Traduction avant déduplication**: Les duplicates peuvent être dans différentes langues
3. **Déduplication avant split**: Évite le data leakage (duplicates dans train et test)
4. **Séparation LaTeX avant concaténation**: Permet des analyses granulaires
5. **Target encoding en fin**: On sait exactement quels échantillons sont conservés
6. **Imputation en dernier**: Utilise les statistiques finales du dataset

---

## 3. Étapes Détaillées

### Étape 1: Text Cleaning (Pattern Correction)

**Objectif**: Corriger les erreurs de formatage courantes dans les textes

**Fonction**: `clean_text_patterns()`

**Transformations**:
```python
"NoteIN"   → "Note: In"
"NoteThe"  → "Note: The"
"NoteA"    → "Note: A"
# ... autres patterns
```

**Colonne affectée**: `prob_desc_notes`

**Sortie**: DataFrame avec `prob_desc_notes` nettoyé

**Mode**: Transform only (pas de fit nécessaire)

---

### Étape 2: Translation (English Normalization)

**Objectif**: Normaliser toutes les descriptions en anglais

**Fonction**: `translate_column()` (depuis `translation_helpers.py`)

**Colonnes traitées**:
- `prob_desc_description`
- `prob_desc_input_spec`
- `prob_desc_output_spec`
- `prob_desc_notes`

**Caractéristiques**:
- Détection automatique de la langue source
- Préservation du LaTeX pendant la traduction
- Gestion des erreurs (affichage du taux de succès)
- Pas de fallback (si erreur, colonne traduite vide)

**Colonnes créées**:
- `prob_desc_description_translated`
- `prob_desc_input_spec_translated`
- `prob_desc_output_spec_translated`
- `prob_desc_notes_translated`

**Mode**: Transform only

**Temps d'exécution**: ~10-30 minutes (selon API rate limits)

---

### Étape 3: Near-Duplicate Detection and Removal

**Objectif**: Identifier et supprimer les descriptions quasi-identiques

**Fonction**: `detect_near_duplicates()` (depuis `advanced_eda.py`)

**Méthode**:
1. Normalisation agressive du texte (lowercase, lettres uniquement, suppression LaTeX)
2. Hashing MD5 des textes normalisés
3. Identification des groupes de hash identiques

**Colonne analysée**: `prob_desc_description_translated`

**Colonne créée**: `prob_desc_description_translated_hash`

**Stratégie de déduplication**:
- Conserver tous les échantillons avec hash unique
- Conserver le **premier** échantillon de chaque groupe de duplicates

**Impact attendu**: Réduction de 1-5% du dataset

**Mode**: Transform only

---

### Étape 4: Numeric Variable Conversion

#### 4.1 Time Limit Conversion

**Objectif**: Convertir les chaînes de temps en valeurs numériques (secondes)

**Fonction**: `convert_time_limit_column()` (depuis `numeric_analysis.py`)

**Formats gérés**:
- Anglais: `"2 seconds"`, `"1 second"`, `"0.5 seconds"`, `"2.0 s"`
- Russe: `"1 секунда"`, `"2 секунды"`

**Colonne source**: `prob_desc_time_limit`

**Colonne créée**: `time_limit_seconds` (float)

**Gestion des erreurs**: Valeurs non parsables → NaN (seront imputées)

#### 4.2 Difficulty Cleaning

**Objectif**: Remplacer les valeurs invalides (-1) par NaN

**Fonction**: `handle_difficulty_invalid_values()`

**Colonne affectée**: `difficulty`

**Transformation**: `-1` → `NaN`

**Mode**: Transform only

---

### Étape 5: Text/LaTeX Separation

**Objectif**: Extraire le texte propre et les métriques LaTeX séparément

**Fonction**: `preprocess_text_full()` (depuis `text_analysis.py`)

**Colonne traitée**: `prob_desc_description_translated`

**Colonnes créées**:
- `clean_description`: Texte sans LaTeX (lettres, chiffres, espaces uniquement)
- `latex_features_desc`: Dictionnaire de métriques LaTeX
- `nb_latex_blocks`: Nombre de blocs LaTeX (`$$$...$$$`)
- `nb_latex_symbols`: Nombre de symboles LaTeX (`\command`)
- `latex_density`: Ratio (longueur LaTeX / longueur totale)
- `latex_symbols_density`: Ratio (longueur symboles / longueur totale)

**Mode**: Transform only

---

### Étape 6: LaTeX Feature Extraction

**Objectif**: Créer des features binaires pour les symboles LaTeX les plus fréquents

**Fonctions**:
1. `extract_all_latex_symbols()` (depuis `latex_analysis.py`)
2. `extract_latex_binary_features()`

**Processus**:
1. Extraction de tous les symboles LaTeX uniques du dataset
2. Création d'une matrice binaire (échantillons × symboles)
3. Sélection des top N symboles (fréquence ≥ seuil)
4. Création de features binaires `has_<symbol>`

**Paramètres**:
- `top_n=30`: Nombre de symboles à conserver
- `min_frequency=10`: Fréquence minimale requise

**Colonnes créées** (exemples):
- `has_le`, `has_ldots`, `has_cdot`, `has_frac`, `has_sum`, `has_prod`, etc.

**Mode**: Fit/Transform
- **Fit** (training): Identifier les top N symboles
- **Transform** (inference): Appliquer les symboles pré-sélectionnés

---

### Étape 7: Text Length Features

**Objectif**: Créer des features basées sur la longueur des textes

**Fonction**: `create_text_length_features()`

**Colonne traitée**: `prob_desc_description_translated`

**Features créées**:
- `prob_desc_description_translated_char_length`: Nombre de caractères
- `prob_desc_description_translated_word_count`: Nombre de mots
- `prob_desc_description_translated_latex_ratio`: Ratio de LaTeX (réutilise `latex_density`)

**Mode**: Transform only

---

### Étape 8: Text Concatenation (Unified Document)

**Objectif**: Créer un document unique en concaténant tous les champs textuels

**Fonction**: `create_unified_document()` (depuis `text_analysis.py`)

**Format**:
```
[DESC] <prob_desc_description_translated>
[IN] <prob_desc_input_spec_translated>
[OUT] <prob_desc_output_spec_translated>
[NOTES] <prob_desc_notes_translated>
[SAMPLE_IN] <prob_desc_sample_inputs>
[SAMPLE_OUT] <prob_desc_sample_outputs>
```

**Tokens spéciaux**:
- `[NOTES_MISSING]`: Si `prob_desc_notes` est vide
- `[SAMPLE_MISSING]`: Si samples manquants

**Colonne créée**: `unified_document`

**Utilité**: Prêt pour TF-IDF, embeddings, ou transformers

**Mode**: Transform only

---

### Étape 9: Target Encoding

**Objectif**: Créer la colonne target et son encoding multi-label

**Fonctions**:
1. `create_priority_tags_column()`
2. `encode_multilabel_target()`

#### 9.1 Priority Tags Column

**Transformation**: Filtrer uniquement les tags prioritaires

**Tags prioritaires**:
```python
['math', 'graphs', 'strings', 'number theory',
 'trees', 'geometry', 'games', 'probabilities']
```

**Colonne source**: `tags` (liste de tous les tags)

**Colonne créée**: `tags_priority` (liste filtrée)

**Exemples**:
```python
tags = ['math', 'graphs', 'implementation']
tags_priority = ['math', 'graphs']

tags = ['implementation', 'brute force']
tags_priority = []  # Aucun tag prioritaire
```

**Décision importante**: Les échantillons sans tags prioritaires sont **conservés** (négatifs utiles)

#### 9.2 Multi-Label Binary Encoding

**Transformation**: Créer une colonne binaire par tag prioritaire

**Colonnes créées** (8 colonnes):
- `target_math`
- `target_graphs`
- `target_strings`
- `target_number_theory`
- `target_trees`
- `target_geometry`
- `target_games`
- `target_probabilities`

**Valeurs**: 0 (absent) ou 1 (présent)

**Exemple**:
```python
tags_priority = ['math', 'graphs']
→ target_math=1, target_graphs=1, target_strings=0, ...
```

**Mode**: Transform only

---

### Étape 10: Missing Value Imputation

**Objectif**: Imputer les valeurs manquantes dans les colonnes numériques

**Fonction**: `impute_missing_values()`

**Stratégie**: Imputation par la médiane globale

**Colonnes traitées**:
- `difficulty`
- `time_limit_seconds`

**Mode**: Fit/Transform
- **Fit** (training): Calculer la médiane sur les données d'entraînement
- **Transform** (inference): Appliquer la médiane pré-calculée

**Sortie**:
```python
df, fill_values = impute_missing_values(df, columns=['difficulty', 'time_limit_seconds'])

# fill_values = {'difficulty': 1500.0, 'time_limit_seconds': 2.0}
```

**Important**: Sauvegarder `fill_values` pour l'inférence

---

### Étape 11: Validation and Summary

**Objectif**: Vérifier la qualité du preprocessing et afficher un résumé

**Fonction**: `validate_preprocessing()`

**Vérifications**:
- Présence des colonnes requises
- Types de données corrects
- Valeurs manquantes résiduelles
- Cohérence des données

**Fonction**: `get_preprocessing_summary()`

**Affichage**:
- Nombre d'échantillons et de features
- Utilisation mémoire
- Distribution des targets
- Valeurs manquantes résiduelles

**Sortie**: Rapport de validation (warnings et errors)

---

## 4. Fonctions Utilitaires

### Module: `src/utils/preprocessing.py`

| Fonction | Description | Mode |
|----------|-------------|------|
| `clean_text_patterns()` | Correction de patterns textuels | Transform |
| `handle_difficulty_invalid_values()` | Remplacement -1 → NaN | Transform |
| `impute_missing_values()` | Imputation de valeurs manquantes | Fit/Transform |
| `create_priority_tags_column()` | Filtrage des tags prioritaires | Transform |
| `encode_multilabel_target()` | Encoding multi-label binaire | Transform |
| `extract_latex_binary_features()` | Features binaires LaTeX | Fit/Transform |
| `create_text_length_features()` | Features de longueur de texte | Transform |
| `remove_duplicate_rows()` | Suppression de duplicates | Transform |
| `validate_preprocessing()` | Validation du preprocessing | - |
| `print_preprocessing_step()` | Affichage formaté d'une étape | - |
| `get_preprocessing_summary()` | Résumé final du dataset | - |

### Dépendances Externes

**Modules existants réutilisés**:
- `src/utils/translation_helpers.py`: `translate_column()`
- `src/utils/numeric_analysis.py`: `convert_time_limit_column()`
- `src/utils/advanced_eda.py`: `detect_near_duplicates()`
- `src/utils/text_analysis.py`: `preprocess_text_full()`, `create_unified_document()`
- `src/utils/latex_analysis.py`: `extract_all_latex_symbols()`

---

## 5. Features Créées

### Récapitulatif

Le pipeline crée **~60-80 nouvelles features** à partir des 21 colonnes initiales.

### Catégories de Features

#### **Textuelles** (5 features)
| Feature | Description | Type |
|---------|-------------|------|
| `clean_description` | Texte sans LaTeX | string |
| `unified_document` | Document concaténé | string |
| `*_translated` (×4) | Colonnes traduites | string |

#### **LaTeX** (34+ features)
| Feature | Description | Type |
|---------|-------------|------|
| `nb_latex_blocks` | Nombre de blocs LaTeX | int |
| `nb_latex_symbols` | Nombre de symboles LaTeX | int |
| `latex_density` | Densité LaTeX | float |
| `latex_symbols_density` | Densité des symboles | float |
| `has_<symbol>` (×30) | Présence de symbole | int (0/1) |

#### **Longueur de Texte** (3 features)
| Feature | Description | Type |
|---------|-------------|------|
| `*_char_length` | Nombre de caractères | int |
| `*_word_count` | Nombre de mots | int |
| `*_latex_ratio` | Ratio LaTeX | float |

#### **Numériques** (2 features)
| Feature | Description | Type |
|---------|-------------|------|
| `time_limit_seconds` | Limite de temps (s) | float |
| `difficulty` | Difficulté (imputée) | float |

#### **Target** (9 features)
| Feature | Description | Type |
|---------|-------------|------|
| `tags_priority` | Tags prioritaires (liste) | list |
| `target_<tag>` (×8) | Encoding binaire multi-label | int (0/1) |

### Features pour la Modélisation

**Features numériques** (~40 features):
- LaTeX: `nb_latex_blocks`, `nb_latex_symbols`, `latex_density`, `has_*` (×30)
- Longueur: `*_char_length`, `*_word_count`
- Autres: `difficulty`, `time_limit_seconds`

**Features textuelles** (1-2 features):
- `unified_document` → À transformer en TF-IDF ou embeddings
- `clean_description` → Alternative si on veut séparer

**Target** (8 features):
- `target_math`, `target_graphs`, ..., `target_probabilities`

---

## 6. Validation et Qualité

### Checks Automatiques

**Présence des colonnes requises**:
- `tags_priority`
- `unified_document`
- `clean_description`
- `difficulty`
- `time_limit_seconds`

**Types de données**:
- Numériques: `difficulty`, `time_limit_seconds`, features LaTeX
- Textuelles: `unified_document`, `clean_description`

**Valeurs manquantes**:
- Après imputation: 0 NaN dans `difficulty` et `time_limit_seconds`
- Textes: Possibles NaN résiduels (signalés en warning)

### Métriques de Qualité

**Réduction du dataset**:
- Attendu: 1-5% (removal de near-duplicates)
- Samples finaux: ~4,700-4,900

**Distribution des targets**:
- Vérification du déséquilibre (ratio max/min attendu ~15:1)
- Vérification de la présence de tous les tags

**Utilisation mémoire**:
- Dataset preprocessé: ~50-150 MB (selon features)

---

## 7. Utilisation en Production

### Workflow Standard

#### Phase 1: Training

```python
# 1. Load raw data
df = load_dataset(DATA_DIR)

# 2. Apply preprocessing pipeline (fit mode)
df = preprocess_training_data(df)  # Fit + Transform

# 3. Save imputation values for inference
save_imputation_values(fill_values, 'imputation_values.json')

# 4. Train/test split (AFTER preprocessing)
X_train, X_test, y_train, y_test = group_split(df, group_col='code_uid')

# 5. Feature engineering (TF-IDF, embeddings)
# ...

# 6. Model training
# ...
```

#### Phase 2: Inference

```python
# 1. Load new data
df_new = load_new_data()

# 2. Load imputation values
fill_values = load_imputation_values('imputation_values.json')

# 3. Apply preprocessing pipeline (transform mode)
df_new = preprocess_inference_data(df_new, fill_values=fill_values)

# 4. Feature engineering (transform mode)
# ...

# 5. Prediction
# ...
```

### Fichiers à Sauvegarder

**Pour le preprocessing**:
1. `preprocessed_dataset.parquet`: Dataset preprocessé
2. `imputation_values.json`: Valeurs d'imputation (médiane)

**Pour le feature engineering** (étape suivante):
3. `tfidf_vectorizer.pkl`: Vectorizer TF-IDF fitté
4. `latex_symbols_list.json`: Liste des symboles LaTeX sélectionnés
5. `feature_names.json`: Liste des features finales

---

## Résumé

### Points Clés

✅ **Pipeline modulaire et réutilisable**: Chaque fonction est indépendante

✅ **Production-ready**: Séparation fit/transform pour éviter le data leakage

✅ **Robuste**: Validation à chaque étape

✅ **Documenté**: Chaque transformation est tracée et expliquée

### Statistiques Finales

**Input**:
- 4,982 échantillons
- 21 colonnes

**Output**:
- ~4,700-4,900 échantillons (après déduplication)
- ~80-100 colonnes (nouvelles features)
- 0 valeurs manquantes dans les colonnes critiques

### Prochaines Étapes

1. **Train/Test Split**: GroupSplit sur `code_uid` (80/20)
2. **Feature Engineering**: TF-IDF sur `unified_document`
3. **Feature Selection**: Sélection des top features
4. **Modeling**: Logistic Regression, XGBoost, BERT
5. **Evaluation**: Hamming Loss, F1 Micro/Macro/Samples

---

*Document généré le 2026-01-12 à partir du notebook `04_preprocessing_pipeline.ipynb`*

