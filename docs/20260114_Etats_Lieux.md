# 📊 Résumé des Travaux - Code Classification Challenge

**Date**: 14 Janvier 2026  
**Objectif Global**: Classification multi-label de problèmes algorithmiques selon 8 tags prioritaires  
**Tags prioritaires**: `math`, `graphs`, `strings`, `number theory`, `trees`, `geometry`, `games`, `probabilities`

---

## 📁 Architecture du Projet

```
illuin_challenge/
├── data/
│   ├── raw/code_classification_dataset/     # Dataset brut (4,982 samples)
│   └── processed/                            # Datasets preprocessés
│       ├── train_preprocessed.parquet       # Train: 3,183 samples, 79 features
│       ├── test_preprocessed.parquet        # Test: 796 samples, 78 features
│       ├── imputation_values.json           # Valeurs d'imputation (medians train)
│       └── mlb_encoder.pkl                  # MultiLabelBinarizer fitted sur train
├── notebooks/
│   ├── 01_eda.ipynb                         # Analyse exploratoire
│   ├── 02_preprocessing_pipeline.ipynb      # Pipeline de preprocessing
│   ├── 03_tfidf_v1.ipynb                    # Modèle TF-IDF
│   └── 04_embeddings_v1.ipynb               # Modèle Embeddings (en cours)
├── src/utils/                               # Fonctions utilitaires
└── docs/embeddings/                         # Visualisations embeddings
```

---

## 🔢 Vue d'Ensemble des Données

### Dataset Initial
- **Samples**: 4,982
- **Colonnes**: 21
- **Near-duplicates retirés**: 9 (détection sur `prob_desc_description_translated`)
- **Samples finaux**: 4,973

### Découpage Train/Test

#### 🎯 **Stratégie de Split** (CRITIQUE pour éviter le data leakage)

**Split Unique - Iterative Stratification** ✅ ACTUEL
- **Train: 3,183 samples (64% du total)**
- **Test: 796 samples (16% du total)**
- **Méthode**: `MultilabelStratifiedShuffleSplit`
- **Random State**: 42 (reproductibilité)
- **Différence moyenne de distribution**: 0.045% (excellente stratification)

**Caractéristiques du split:**
- Appliqué directement après la déduplication sur les 4,973 samples
- Garantit une distribution équilibrée des 8 labels multi-label entre train et test
- Split ratio: 80/20 (train/test)

**⚠️ IMPORTANT POUR LA MODÉLISATION:**
- **Toutes les opérations de fitting (imputation, TF-IDF, embeddings) DOIVENT être fittées sur `df_train` UNIQUEMENT**
- Les valeurs fittées sont ensuite appliquées à `df_test` (jamais l'inverse)
- Le split est réalisé AVANT toute opération de feature engineering avec fitting

### Distribution des Labels (après stratification)

| Label | Train Count | Train % | Test Count | Test % | Diff |
|-------|-------------|---------|------------|--------|------|
| **math** | 902 | 28.34% | 225 | 28.27% | 0.07% |
| **graphs** | 355 | 11.15% | 89 | 11.18% | 0.03% |
| **strings** | 274 | 8.61% | 68 | 8.54% | 0.07% |
| **number theory** | 228 | 7.16% | 57 | 7.16% | 0.00% |
| **trees** | 204 | 6.41% | 51 | 6.41% | 0.00% |
| **geometry** | 102 | 3.20% | 26 | 3.27% | 0.06% |
| **games** | 70 | 2.20% | 17 | 2.14% | 0.06% |
| **probabilities** | 62 | 1.95% | 15 | 1.88% | 0.06% |

**Moyenne de labels par document**: ~0.69 labels/doc (multi-label peu dense)

---

## 📓 Détail des Notebooks

---

### 01_eda.ipynb - Analyse Exploratoire des Données

#### 🎯 Objectif
Comprendre la structure, la qualité et les caractéristiques du dataset brut avant toute transformation.

#### 📥 Input
- `data/raw/code_classification_dataset/` (4,982 samples, 21 colonnes)

#### 📊 Analyses Réalisées

**1. Analyse de la Qualité des Données**
- Taux de valeurs manquantes par colonne
- Distribution des langages de programmation
- Distribution de la difficulté (`difficulty`)
- Valeurs aberrantes détectées (`difficulty = -1`)

**2. Analyse des Tags**
- 8 tags prioritaires identifiés
- Distribution des tags (déséquilibrée: math 29%, probabilities 2%)
- Analyse multi-label: ~1 tag par problème en moyenne

**3. Analyse Textuelle**
- Distribution des longueurs de texte (description, input_spec, output_spec, notes)
- Détection de LaTeX: ~55% des descriptions contiennent du LaTeX
- Analyse de la langue: 99.6% anglais, 0.4% autres langues

**4. Analyse LaTeX**
- Identification des patterns LaTeX: `$...$`, `$$...$$`, `$$$...$$$`, `\[...\]`, `\(...\)`
- Extraction des commandes LaTeX fréquentes: `\le`, `\ldots`, `\frac`, `\sum`, etc.
- Densité LaTeX par document

**5. Near-Duplicates**
- Détection de 9 groupes de near-duplicates (18 samples concernés)
- Normalisation de texte pour la détection

#### 📤 Output
- Statistiques descriptives (console)
- Identification des problèmes de qualité
- Liste des tags prioritaires → `src/config.py`

#### 🔑 Insights Clés
- Dataset principalement en anglais (traduction nécessaire pour <1%)
- LaTeX présent dans >50% des descriptions (traitement spécial requis)
- Distribution des tags très déséquilibrée (stratification critique)
- Valeurs manquantes: `prob_desc_notes` (27%), autres colonnes <2%

---

### 02_preprocessing_pipeline.ipynb - Pipeline de Preprocessing Complet

#### 🎯 Objectif
Préparer le dataset pour la modélisation en appliquant toutes les transformations nécessaires.

#### 📥 Input
- `data/raw/code_classification_dataset/` (4,982 samples)

#### 📤 Output
- `data/processed/train_preprocessed.parquet` (3,183 samples, 79 features)
- `data/processed/test_preprocessed.parquet` (796 samples, 78 features)
- `data/processed/imputation_values.json` (valeurs de médiane pour inference)
- `data/processed/mlb_encoder.pkl` (MultiLabelBinarizer fitted sur train)

---

#### 🛠️ Pipeline Détaillé (11 Étapes)

---

##### **ÉTAPE 1: Text Pattern Cleaning**

**Application**: Train + Test (pas de fitting)

**Opération**:
- Correction des patterns malformés dans `prob_desc_notes`
- Exemples: `NoteIN` → `Note: In`, `NoteThe` → `Note: The`

**Fonction**: `clean_text_patterns()`

**Colonnes modifiées**: `prob_desc_notes`

---

##### **ÉTAPE 2: Translation (English Normalization)**

**Application**: Train + Test (pas de fitting)

**Colonnes traduites** (si non-anglais détecté):
- `prob_desc_description` → `prob_desc_description_translated`
- `prob_desc_input_spec` → `prob_desc_input_spec_translated`
- `prob_desc_output_spec` → `prob_desc_output_spec_translated`
- `prob_desc_notes` → `prob_desc_notes_translated`

**Résultats**:
- Description: 18/4982 traduits (0.4%)
- Input spec: 30/4982 traduits (0.6%)
- Output spec: 28/4982 traduits (0.6%)
- Notes: 21/4982 traduits (0.4%)

**Fonction**: `translate_column()` avec `googletrans`

**Conservation du LaTeX**: Les patterns LaTeX sont préservés durant la traduction

---

##### **ÉTAPE 3: Near-Duplicate Detection & Removal**

**Application**: Dataset complet (avant split)

**Méthode**:
1. Normalisation du texte (lowercase, remove punctuation, strip whitespace)
2. Hashage SHA256 de `prob_desc_description_translated`
3. Détection des groupes avec hash identique
4. Suppression des duplicates (keep='first')

**Résultats**:
- 9 groupes de near-duplicates détectés
- 9 samples supprimés (0.18%)
- Dataset final: 4,973 samples

**Fonction**: `detect_near_duplicates()`

**Colonnes créées**: `prob_desc_description_translated_hash`

---

##### **ÉTAPE 4: Train/Test Split (Iterative Stratification)**

**📍 SPLIT CRITIQUE - Emplacement dans le pipeline**

**Application**: Après la déduplication, AVANT toute opération de fitting

**Méthode**: `MultilabelStratifiedShuffleSplit`
- Garantit une distribution équilibrée des 8 labels multi-label
- Test size: 20%
- Random state: 42

**Résultats**:
- **Train**: 3,183 samples (80%)
- **Test**: 796 samples (20%)
- Différence moyenne de distribution: 0.045%

**⚠️ SÉPARATION TRAIN/TEST**:
À partir de cette étape, **toutes les opérations suivantes sont appliquées séparément** sur train et test:
- **Operations SANS fitting**: Appliquées indépendamment aux 2 datasets
- **Operations AVEC fitting**: Fit sur train UNIQUEMENT, puis apply sur train et test

---

##### **ÉTAPE 5: Numeric Variable Conversion**

**Application**: Train + Test (pas de fitting, transformation déterministe)

**5.1 Time Limit Conversion**
- Colonne source: `prob_desc_time_limit` (string: "1 second", "2 seconds")
- Colonne créée: `time_limit_seconds` (float)
- Parsing: Extraction du nombre et conversion en secondes
- Valeurs uniques: 0.5s à 15.0s (majorité: 1s et 2s)

**5.2 Difficulty Cleaning**
- Valeurs invalides (`-1`) remplacées par `NaN`
- Train: 0 valeurs invalides
- Test: 0 valeurs invalides
- Imputation réalisée plus tard (Étape 10)

**Fonctions**: 
- `convert_time_limit_column()`
- `handle_difficulty_invalid_values()`

---

##### **ÉTAPE 6: Text/LaTeX Separation**

**Application**: Train + Test (pas de fitting)

**Colonne traitée**: `prob_desc_description_translated`

**Opérations de `preprocess_text_full()`**:

1. **Extraction des symboles LaTeX**:
   - Pattern: `\\([a-zA-Z]+)` → Capture `\sum`, `\frac`, `\le`, etc.
   - Stockage de la liste des symboles

2. **Suppression des blocs LaTeX**:
   - `$$$...$$$` → `LATEXBLOCK`
   - `$$...$$` → `LATEXBLOCK`
   - `$...$` → `LATEXBLOCK`
   - `\[...\]` → `LATEXBLOCK`
   - `\(...\)` → `LATEXBLOCK`

3. **Suppression des commandes LaTeX**:
   - `\command` → ` ` (espace)

4. **Normalisation**:
   - Espaces multiples → espace unique
   - Lowercase
   - Strip

**Colonnes créées**:
- `clean_description`: Texte sans LaTeX (string)
- `nb_latex_blocks`: Nombre de blocs LaTeX (int)
- `nb_latex_symbols`: Nombre de commandes LaTeX (int)
- `latex_density`: Ratio de caractères LaTeX / total (float)
- `latex_symbols_density`: Symboles LaTeX / mots (float)
- `latex_features_desc`: Dict complet des features LaTeX

**Résultats**:
- Train: 1,782/3,183 samples avec LaTeX (56.0%)
- Test: 428/796 samples avec LaTeX (53.8%)

**Fonction**: `preprocess_text_full()`

---

##### **ÉTAPE 7: LaTeX Feature Extraction (Binary Features)**

**Application**: FIT sur train, APPLY sur train + test

**Méthode**:

1. **Extraction de tous les symboles LaTeX** (train + test séparément):
   - Train: 98 symboles uniques détectés
   - Test: 62 symboles uniques détectés

2. **Sélection des top symboles** (FIT sur train):
   - Top N: 30 symboles les plus fréquents
   - Min frequency: 10 occurrences minimum
   - Symboles sélectionnés depuis le TRAIN uniquement

3. **Création de features binaires**:
   - Format: `has_{symbol}` (ex: `has_le`, `has_frac`, `has_sum`)
   - Train: 31 features créées
   - Test: 30 features créées (symboles du train appliqués au test)

**Top 10 symboles LaTeX** (exemples):
- `has_le`: 581 train (18.3%), 140 test (17.6%)
- `has_ldots`: 286 train (9.0%), 60 test (7.5%)
- `has_dots`: 256 train (8.0%), 70 test (8.8%)
- `has_leq`: 172 train (5.4%), 47 test (5.9%)
- `has_cdot`: 161 train (5.1%), 35 test (4.4%)

**Fonction**: `extract_latex_binary_features()`

**⚠️ Note**: Les symboles sont fittés sur train, donc certains symboles présents dans test mais absents de train ne seront pas détectés

---

##### **ÉTAPE 8: Text Length Features**

**Application**: Train + Test (pas de fitting, calculs déterministes)

**Colonne traitée**: `prob_desc_description_translated`

**Features créées** (4 features):

1. `prob_desc_description_translated_char_length`:
   - Nombre total de caractères
   - Train mean: 958.57, Test mean: 936.37

2. `prob_desc_description_translated_word_count`:
   - Nombre de mots (split sur whitespace)
   - Train mean: 168.76, Test mean: 165.41

3. `prob_desc_description_translated_numeric_ratio`:
   - Ratio de chiffres (0-9) dans le texte
   - Train mean: 0.01, Test mean: 0.01

4. `prob_desc_description_translated_latex_ratio`:
   - Copie de `latex_density` (déjà calculée)
   - Train mean: 0.10, Test mean: 0.10

**Fonction**: `create_text_length_features()`

---

##### **ÉTAPE 9: Text Concatenation (Unified Document)**

**Application**: Train + Test (pas de fitting)

**But**: Créer un document unique concaténant toutes les sections du problème

**9.1 Unified Document (avec LaTeX)**

**Colonnes concaténées** (avec balises):
- `[DESC]` + `prob_desc_description_translated`
- `[INPUT]` + `prob_desc_input_spec_translated`
- `[OUTPUT]` + `prob_desc_output_spec_translated`
- `[NOTE]` + `prob_desc_notes_translated`
- `[SAMPLE_INPUT]` + `prob_desc_sample_inputs`
- `[SAMPLE_OUTPUT]` + `prob_desc_sample_outputs`

**Colonne créée**: `unified_document`

**Longueur moyenne**:
- Train: 2,046 caractères
- Test: 1,990 caractères

**Fonction**: `create_unified_document()`

---

**9.2 Unified Document (sans LaTeX)**

**Opération**:
- Remplacement de tous les blocs LaTeX par le token `[LATEX]`
- Suppression des commandes LaTeX restantes

**Colonne créée**: `unified_document_without_latex`

**Résultats**:
- Train: 2,046 → 1,964 chars (réduction 4.0%)
- Test: 1,990 → 1,917 chars (réduction 3.7%)
- Documents avec [LATEX] token: 96% (train), 96% (test)

**Fonction**: `remove_latex_from_text(replacement_token='[LATEX]')`

**💡 Usage**:
- `unified_document`: Pour modèles capables de comprendre le LaTeX (ex: SPECTER)
- `unified_document_without_latex`: Pour modèles standards (TF-IDF, BERT classique)

---

##### **ÉTAPE 10: Target Encoding (Multi-Label)**

**Application**: FIT sur train, APPLY sur train + test

**10.1 Priority Tags Column**

**Opération**:
- Filtrage des tags pour ne garder que les 8 tags prioritaires
- Colonne source: `tags` (liste de tags)
- Colonne créée: `tags_priority` (liste filtrée)

**Résultats**:
- Train: 1,715/3,183 avec au moins 1 tag prioritaire (53.9%)
- Test: 428/796 avec au moins 1 tag prioritaire (53.8%)
- Les samples sans tag prioritaire sont gardés comme exemples négatifs

**Fonction**: `create_priority_tags_column()`

---

**10.2 Multi-Label Binary Encoding**

**Opération** (FIT sur train):
1. Fit `MultiLabelBinarizer` sur `df_train['tags_priority']`
2. Transform train et test avec le même encoder
3. Création de 8 colonnes binaires: `target_{tag}`

**Classes détectées**: 
```python
['games', 'geometry', 'graphs', 'math', 'number theory', 
 'probabilities', 'strings', 'trees']
```

**Colonnes créées** (8 colonnes target):
- `target_games`
- `target_geometry`
- `target_graphs`
- `target_math`
- `target_number_theory`
- `target_probabilities`
- `target_strings`
- `target_trees`

**Distribution** (voir tableau au début du document)

**⚠️ IMPORTANT**: Le `MultiLabelBinarizer` est sauvegardé dans `mlb_encoder.pkl` pour être utilisé en inference

---

##### **ÉTAPE 11: Missing Value Imputation**

**Application**: FIT sur train, APPLY sur train + test

**Colonnes imputées**:
- `difficulty`
- `time_limit_seconds`

**Stratégie**: Médiane (calculée sur train uniquement)

**Valeurs d'imputation** (fittées sur train):
```json
{
  "difficulty": 1700.0,
  "time_limit_seconds": 2.0
}
```

**Résultats**:
- Train difficulty: 0 valeurs imputées
- Test difficulty: 0 valeurs imputées
- Train time_limit: 0 valeurs imputées
- Test time_limit: 0 valeurs imputées

**⚠️ IMPORTANT**: Les valeurs de médiane sont sauvegardées dans `imputation_values.json` pour être utilisées en inference sur de nouveaux samples

**Fonction**: `impute_missing_values()` (custom, basée sur medians)

---

#### 📊 Résumé des Features Créées (79 features au total)

**Features Originales** (21):
- `prob_desc_*` (description, input_spec, output_spec, notes, etc.)
- `difficulty`, `time_limit`, `tags`, `src_uid`, `lang`, etc.

**Features de Traduction** (4):
- `*_translated` pour les 4 colonnes textuelles principales

**Features LaTeX** (5 + 31):
- `nb_latex_blocks`, `nb_latex_symbols`, `latex_density`, `latex_symbols_density`, `latex_features_desc`
- 31 features binaires `has_{symbol}`

**Features de Longueur de Texte** (4):
- `*_char_length`, `*_word_count`, `*_numeric_ratio`, `*_latex_ratio`

**Features de Document Unifié** (2):
- `unified_document`, `unified_document_without_latex`

**Features de Target** (8 + 1):
- 8 colonnes `target_{tag}` binaires
- `tags_priority` (liste filtrée)

**Features Numériques Transformées** (1):
- `time_limit_seconds` (conversion depuis string)

**Features Techniques** (1):
- `prob_desc_description_translated_hash` (pour near-duplicate detection)

**⚠️ Note**: Train a 79 features, Test a 78 (1 feature binaire LaTeX manquante car symbole absent du train)

---

### 03_tfidf_v1.ipynb - Modèle Baseline TF-IDF

#### 🎯 Objectif
Créer un modèle baseline multi-label classification basé sur TF-IDF + Classifieurs traditionnels.

#### 📥 Input
- `data/processed/train_preprocessed.parquet`
- `data/processed/test_preprocessed.parquet`

#### 🛠️ Approche

**1. Vectorisation TF-IDF**
- Colonne utilisée: `unified_document_without_latex`
- Paramètres TF-IDF:
  - `max_features`: 5000-10000
  - `ngram_range`: (1, 2) ou (1, 3)
  - `min_df`, `max_df`: Filtrage des termes trop rares/fréquents

**2. Modèles Testés**
- Logistic Regression (One-vs-Rest)
- Random Forest
- XGBoost
- LightGBM

**3. Métriques d'Évaluation**
- Precision, Recall, F1-score par label
- Macro/Micro averages
- Hamming Loss
- Subset Accuracy

#### 📤 Output
- Modèles sauvegardés (`.pkl`)
- Matrice TF-IDF sauvegardée
- Rapport de performance par label

#### 🔑 Insights
- Baseline pour comparer avec les approches deep learning
- Identification des labels faciles vs difficiles

---

### 04_embeddings_v1.ipynb - Modèle Embeddings (Sentence-Transformers)

#### 🎯 Objectif
Tester différents embedders et identifier les colonnes textuelles les plus discriminantes pour la classification.

#### 📥 Input
- `data/processed/train_preprocessed.parquet`

#### 🛠️ Analyses

**Section 2.1: Comparaison de 3 Embedders (sur `prob_desc_description` uniquement)**

**Embedders testés**:
1. `all-MiniLM-L6-v2` (384 dim)
2. `all-mpnet-base-v2` (768 dim)
3. `allenai-specter` (768 dim)

**Métriques de comparaison**:
- **Mean Δmean**: Différence moyenne de similarité cosine (positifs vs négatifs)
- **Centroid Similarity**: Similarité moyenne entre centroids de classes
- **Performance par tag**: Δmean individuel pour chaque des 8 tags

**Résultats**:

| Modèle | Dimensions | Mean Δmean ↑ | Centroid Sim ↓ |
|--------|-----------|--------------|----------------|
| **all-MiniLM-L6-v2** ✅ | 384 | **0.1470** | **0.7255** |
| all-mpnet-base-v2 | 768 | 0.1319 | 0.7533 |
| allenai-specter ❌ | 768 | 0.0372 | 0.9654 |

**Conclusion Section 2.1**:
- **Modèle recommandé**: `all-MiniLM-L6-v2`
- Meilleure séparation positifs/négatifs
- Moins de confusion inter-classes
- Plus léger et rapide (384 dim vs 768)

---

**Section 2.2: Analyse par Colonne Textuelle** 🔄 **EN COURS D'EXÉCUTION**

**Objectif**: Identifier quelle(s) colonne(s) contiennent le plus d'information discriminante

**Modèle utilisé**: `all-MiniLM-L6-v2` (gagnant de la Section 2.1)

**Colonnes testées**:
- `prob_desc_description_translated`
- `prob_desc_input_spec_translated`
- `prob_desc_output_spec_translated`
- `prob_desc_notes_translated`

**Métriques par colonne**:
- Mean Δmean (moyenne simple)
- **Weighted Δmean** (moyenne pondérée par le nombre de positifs par label)
- Centroid Similarity
- Coverage (% de documents non-vides)
- Δmean par (Tag × Colonne) → Heatmap 8×4

**Analyses prévues**:
1. Tableau comparatif trié par Mean Δmean
2. Graphiques:
   - Mean Δmean par colonne (barres horizontales)
   - Weighted Δmean par colonne (barres horizontales)
   - Coverage vs Performance (scatter plot)
   - Centroid Similarity par colonne
   - **Heatmap Δmean par (Tag × Colonne)** → Identifier si certains tags sont mieux capturés par certaines colonnes

**Questions à répondre**:
- Quelle colonne est la plus discriminante globalement ?
- Y a-t-il des complémentarités ? (ex: "graphs" mieux dans input_spec, "math" mieux dans description)
- Trade-off coverage vs qualité ? (notes peut être sparse mais discriminant)
- Recommandation pour le document unifié: garder toutes les colonnes ou filtrer ?

#### 📤 Output (prévu)
- Embeddings par colonne: `minilm_{colonne}_embeddings_train.npy`
- Centroids par colonne
- Heatmaps de similarité
- Graphiques comparatifs: `docs/embeddings/column_comparison_complete.png`

---

## 🎯 Variables Explicatives Disponibles pour la Modélisation

### 📊 Vue d'Ensemble

**Total Features**: 79 (train) / 78 (test)

---

### 1️⃣ **Features Textuelles Brutes** (pour TF-IDF / Embeddings)

**Colonnes disponibles** (toutes traduites en anglais):

| Colonne | Description | Coverage | Usage Recommandé |
|---------|-------------|----------|------------------|
| `prob_desc_description_translated` | Description du problème | 100% | ✅ **Principale** |
| `prob_desc_input_spec_translated` | Spécification des entrées | 99.3% | ✅ Complémentaire |
| `prob_desc_output_spec_translated` | Spécification des sorties | 98.4% | ✅ Complémentaire |
| `prob_desc_notes_translated` | Notes additionnelles | 72.9% | ⚠️ Sparse mais informatif |
| `unified_document` | Concaténation avec LaTeX | 100% | ✅ Pour modèles LaTeX-aware |
| `unified_document_without_latex` | Concaténation avec token [LATEX] | 100% | ✅ Pour modèles standards |
| `clean_description` | Description sans LaTeX | 100% | ✅ Alternative sans LaTeX |

**💡 Recommandations**:
- **TF-IDF**: Utiliser `unified_document_without_latex` (concaténation + LaTeX tokenisé)
- **Embeddings génériques**: Utiliser `unified_document_without_latex`
- **Embeddings spécialisés**: Tester colonnes individuelles puis concaténation pondérée
- **SPECTER**: Utiliser `unified_document` (avec LaTeX brut)

---

### 2️⃣ **Features Numériques LaTeX** (6 features)

| Feature | Type | Description | Mean (train) |
|---------|------|-------------|--------------|
| `nb_latex_blocks` | int | Nombre de blocs LaTeX ($$$, $$, $) | Variable |
| `nb_latex_symbols` | int | Nombre de commandes LaTeX (\sum, \frac, etc.) | Variable |
| `latex_density` | float | Ratio caractères LaTeX / total | 0.10 |
| `latex_symbols_density` | float | Symboles LaTeX / mots | Variable |
| `latex_features_desc` | dict | Dict complet (pas directement utilisable) | - |

**💡 Usage**:
- Features discriminantes pour certains tags (ex: "math" corrélé avec `latex_density`)
- Peuvent être utilisées comme features additionnelles dans un modèle hybride

---

### 3️⃣ **Features Binaires LaTeX** (31 features)

**Format**: `has_{symbol}` (ex: `has_le`, `has_frac`, `has_sum`)

**Top 10 symboles les plus fréquents**:
- `has_le`: 18.3% des documents (train)
- `has_ldots`: 9.0%
- `has_dots`: 8.0%
- `has_leq`: 5.4%
- `has_cdot`: 5.1%
- `has_times`: 3.6%
- `has_ne`: 3.5%
- `has_frac`: 3.4%
- `has_ge`: 2.9%
- `has_sum`: 2.1%

**💡 Usage**:
- Features interprétables pour certains tags (ex: `has_sum` pour "math")
- Peuvent améliorer les modèles traditionnels (Logistic Regression, XGBoost)
- Moins utiles pour deep learning (déjà capturé par embeddings)

---

### 4️⃣ **Features de Longueur de Texte** (4 features)

| Feature | Mean (train) | Mean (test) |
|---------|--------------|-------------|
| `prob_desc_description_translated_char_length` | 958.57 | 936.37 |
| `prob_desc_description_translated_word_count` | 168.76 | 165.41 |
| `prob_desc_description_translated_numeric_ratio` | 0.01 | 0.01 |
| `prob_desc_description_translated_latex_ratio` | 0.10 | 0.10 |

**💡 Usage**:
- Features de complexité du problème
- Peuvent aider à détecter les outliers
- Faible pouvoir discriminant seuls, mais utiles en combinaison

---

### 5️⃣ **Features Numériques Générales** (2 features)

| Feature | Type | Imputation | Mean (train) |
|---------|------|------------|--------------|
| `difficulty` | float | Médiane (1700.0) | 1700.0 |
| `time_limit_seconds` | float | Médiane (2.0) | 2.0 |

**💡 Usage**:
- Peuvent être corrélées avec certains tags
- `difficulty` peut aider à détecter les problèmes complexes

---

### 6️⃣ **Features Target** (8 colonnes binaires)

**Format**: `target_{tag}` (0 ou 1)

**Labels**:
- `target_math`, `target_graphs`, `target_strings`, `target_number_theory`
- `target_trees`, `target_geometry`, `target_games`, `target_probabilities`

**⚠️ ATTENTION**: Ces colonnes sont les **targets** à prédire, pas des features !

---

### 7️⃣ **Features TF-IDF** (à créer)

**Source**: `unified_document_without_latex`

**Paramètres recommandés**:
```python
TfidfVectorizer(
    max_features=10000,       # À tuner (5000-20000)
    ngram_range=(1, 2),       # Unigrammes + bigrammes
    min_df=5,                 # Ignorer termes trop rares
    max_df=0.7,               # Ignorer termes trop fréquents
    sublinear_tf=True,        # log(tf)
    norm='l2'                 # Normalisation L2
)
```

**Dimensions**: 5000-20000 features (sparse)

**💡 Usage**:
- FIT sur train uniquement
- Transform train et test avec le même vectorizer
- Modèles recommandés: Logistic Regression, XGBoost, LightGBM

---

### 8️⃣ **Features Embeddings** (à créer)

**Source**: Colonnes textuelles (description, input_spec, output_spec, notes)

**Modèle recommandé**: `all-MiniLM-L6-v2` (384 dimensions)

**Stratégies possibles**:

**Stratégie 1: Document Unifié**
- Embedder `unified_document_without_latex`
- Dimensions: 384
- Simple et rapide

**Stratégie 2: Colonnes Individuelles** (recommandé selon résultats Section 2.2)
- Embedder chaque colonne séparément
- Concaténer les embeddings: 384 × 4 = 1536 dimensions
- Ou pondérer selon l'importance de chaque colonne

**Stratégie 3: Mean/Max Pooling**
- Embedder chaque colonne
- Mean/Max pooling pour obtenir 384 dimensions

**💡 Usage**:
- Calculer embeddings sur train et test
- Sauvegarder matrices d'embeddings (`.npy`)
- Modèles recommandés: Logistic Regression, XGBoost, Neural Network

---

## 🎯 Stratégie de Modélisation Recommandée

### Phase 1: Modèles Baselines (Rapides à tester)

**1.1 TF-IDF + Logistic Regression**
- Features: TF-IDF (10k features) + 42 features numériques/binaires
- Modèle: `OneVsRestClassifier(LogisticRegression())`
- Temps: ~5-10 min
- But: Baseline solide et interprétable

**1.2 TF-IDF + XGBoost**
- Features: TF-IDF + features numériques
- Modèle: XGBoost multi-label (ou One-vs-Rest)
- Temps: ~20-30 min
- But: Meilleure performance que Logistic Regression

---

### Phase 2: Modèles Embeddings (Plus longs mais meilleurs)

**2.1 Embeddings + Logistic Regression**
- Features: Embeddings (384-1536 dim) + features numériques
- Modèle: `OneVsRestClassifier(LogisticRegression())`
- Temps: ~10-15 min (après calcul des embeddings)
- But: Tester la qualité des embeddings

**2.2 Embeddings + Neural Network**
- Features: Embeddings + features numériques
- Architecture: Dense layers (384 → 256 → 128 → 8)
- Activation finale: Sigmoid (multi-label)
- Loss: Binary Cross-Entropy
- Temps: ~30-60 min
- But: Meilleure modélisation des interactions non-linéaires

---

### Phase 3: Modèles Hybrides (Combinaison TF-IDF + Embeddings)

**3.1 Stacking**
- Niveau 1: TF-IDF + Logistic, Embeddings + Logistic, XGBoost
- Niveau 2: Meta-modèle (Logistic ou XGBoost)
- Temps: ~1-2h
- But: Combiner les forces de chaque approche

**3.2 Feature Concatenation**
- Features: TF-IDF (top 5000) + Embeddings (384) + features numériques
- Modèle: XGBoost ou Neural Network
- Temps: ~1h
- But: Utiliser toute l'information disponible

---

## ⚙️ Calibrage du Modèle - Bonnes Pratiques

### 🔒 Règles Critiques (Éviter le Data Leakage)

**1. Toujours FIT sur TRAIN uniquement**
- TF-IDF Vectorizer: `fit(df_train['text'])`
- Embedder: Calculer embeddings sur train, puis test
- Scaler (si utilisé): `fit(X_train)`
- Imputation: Médiane calculée sur train

**2. Jamais toucher au TEST avant l'évaluation finale**
- Le test sert UNIQUEMENT à évaluer la performance finale
- Toutes les décisions de modélisation se font sur train/validation

**3. Cross-Validation sur TRAIN**
- Utiliser `StratifiedKFold` (ou `MultilabelStratifiedKFold` si disponible)
- 5-10 folds
- But: Sélection d'hyperparamètres, comparaison de modèles

---

### 📊 Split Train/Validation (pour tuning)

**Option 1: Holdout Validation (recommandé pour vitesse)**
```python
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in msss.split(X_train, y_train):
    X_train_split, X_val = X_train[train_idx], X_train[val_idx]
    y_train_split, y_val = y_train[train_idx], y_train[val_idx]
```

**Option 2: K-Fold Cross-Validation (recommandé pour robustesse)**
```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(mskf.split(X_train, y_train)):
    # Train sur train_idx, évaluer sur val_idx
    pass
```

---

### 🎛️ Hyperparameter Tuning

**Méthode recommandée**: Grid Search ou Random Search avec Cross-Validation

**Exemple TF-IDF + Logistic Regression**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'tfidf__max_features': [5000, 10000, 15000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__min_df': [2, 5, 10],
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l1', 'l2']
}

# GridSearchCV avec cross-validation multi-label
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=mskf,  # MultilabelStratifiedKFold
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)
```

**Scoring metrics pour multi-label**:
- `f1_macro`: Moyenne des F1 par label (équilibrée)
- `f1_micro`: F1 global (favorise les labels fréquents)
- `f1_weighted`: F1 pondéré par fréquence
- Custom scorer: `make_scorer(custom_metric)`

---

### 📈 Métriques d'Évaluation

**Métriques par Label** (8 labels):
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=label_names))
```

**Métriques Globales**:
```python
from sklearn.metrics import hamming_loss, jaccard_score

# Hamming Loss (↓ mieux): % de labels mal prédits
hamming = hamming_loss(y_test, y_pred)

# Jaccard Score (↑ mieux): Intersection / Union
jaccard = jaccard_score(y_test, y_pred, average='samples')

# Subset Accuracy (↑ mieux): % de samples parfaitement prédits
subset_acc = (y_test == y_pred).all(axis=1).mean()
```

---

## 📁 Fichiers Importants pour la Modélisation

### Input (Déjà créés)

| Fichier | Description | Taille | Usage |
|---------|-------------|--------|-------|
| `train_preprocessed.parquet` | Dataset train | 3,183 samples | Entraînement + validation |
| `test_preprocessed.parquet` | Dataset test | 796 samples | Évaluation finale |
| `imputation_values.json` | Médianes train | - | Inference sur nouveaux samples |
| `mlb_encoder.pkl` | MultiLabelBinarizer | - | Inference sur nouveaux samples |

### Output (À créer durant modélisation)

| Fichier | Description | Usage |
|---------|-------------|-------|
| `tfidf_vectorizer.pkl` | TF-IDF fitted sur train | Inference |
| `embeddings_train.npy` | Embeddings train (3183, 384) | Modélisation |
| `embeddings_test.npy` | Embeddings test (796, 384) | Évaluation |
| `model_*.pkl` | Modèles entraînés | Inference |
| `results_*.json` | Métriques de performance | Comparaison |

---

## 🚀 Prochaines Étapes

### Immédiat (En cours)
1. ✅ Attendre fin de l'analyse Section 2.2 (colonnes textuelles)
2. ✅ Analyser les résultats de la heatmap (Tag × Colonne)

### Court Terme
3. **Décider stratégie embeddings**:
   - Document unifié OU colonnes séparées ?
   - Pondération des colonnes selon Δmean ?

4. **Créer features TF-IDF**:
   - Vectorizer sur `unified_document_without_latex`
   - Sauvegarder matrice sparse + vectorizer

5. **Créer features Embeddings**:
   - all-MiniLM-L6-v2 sur colonnes sélectionnées
   - Sauvegarder matrices .npy

### Moyen Terme
6. **Entraîner modèles baselines**:
   - TF-IDF + Logistic Regression
   - TF-IDF + XGBoost
   - Embeddings + Logistic Regression

7. **Comparer performances**:
   - F1-macro, F1 par label, Hamming Loss
   - Identifier labels difficiles

8. **Tuning hyperparamètres**:
   - Grid Search avec Cross-Validation
   - Sélectionner meilleur modèle

### Long Terme
9. **Modèles avancés**:
   - Neural Networks (Dense layers)
   - Stacking / Ensemble
   - Fine-tuning d'embedders (si ressources suffisantes)

10. **Évaluation finale**:
    - Performance sur `test_preprocessed.parquet`
    - Analyse des erreurs
    - Rapport final

---

## 📝 Notes Importantes

### ⚠️ Points d'Attention

1. **Data Leakage Prevention**:
   - TOUTES les opérations de fitting doivent être sur train uniquement
   - Valeurs d'imputation, encoders, vectorizers: fit(train) puis transform(test)

2. **Stratification Multi-Label**:
   - Utiliser `iterstrat.ml_stratifiers` pour la validation
   - Éviter `StratifiedKFold` standard (ne gère pas le multi-label)

3. **Imbalance des Labels**:
   - Math: 28% vs Probabilities: 2% (ratio 14:1)
   - Envisager class weights ou oversampling si besoin

4. **Features Sparse**:
   - `prob_desc_notes`: 27% de valeurs manquantes
   - TF-IDF: Matrices très sparse (~99%)
   - Utiliser sparse matrices pour économiser mémoire

5. **Computational Cost**:
   - Embeddings: ~5-10 min pour 3,979 samples
   - TF-IDF: Quasi-instantané
   - Neural Networks: ~30-60 min d'entraînement

---

## 🎓 Récapitulatif Méthodologique

### Découpage des Données

```
Dataset Initial (4,982)
    ↓
Déduplication (-9)
    ↓
Dataset Propre (4,973)
    ↓
[SPLIT UNIQUE] Iterative Stratification (80/20)
    ↓
┌─────────────────┬──────────────────┐
│   Train (3,183) │   Test (796)     │  ← Ne JAMAIS toucher avant éval finale
│        ↓        │                  │
│   [Toutes les   │                  │
│   opérations    │                  │
│   de fitting]   │                  │
│        ↓        │                  │
│   Entraînement  │                  │
│   + Validation  │                  │
│   (avec CV)     │                  │
└─────────────────┴──────────────────┘
```

### Workflow de Modélisation

```
1. Charger train_preprocessed.parquet
2. Séparer features (X) et targets (y)
3. Créer validation split (MultilabelStratifiedShuffleSplit)
4. Fit preprocessing sur X_train (TF-IDF, Scaler, etc.)
5. Transform X_train, X_val, X_test
6. Entraîner modèle sur (X_train, y_train)
7. Tuner hyperparamètres avec Cross-Validation
8. Évaluer sur X_val
9. Itérer (steps 4-8) jusqu'à satisfaction
10. Évaluation FINALE sur test UNIQUEMENT
```

---

## 📚 Ressources et Dépendances

### Packages Principaux
```python
# Data manipulation
pandas, numpy

# NLP
sentence-transformers (embeddings)
scikit-learn (TF-IDF, models)

# Multi-label
iterative-stratification (MultilabelStratifiedKFold)

# Models
xgboost, lightgbm
tensorflow/keras (si Neural Networks)

# Metrics
sklearn.metrics
```

### Documentation Utile
- Sentence-Transformers: https://www.sbert.net/
- Iterative Stratification: https://github.com/trent-b/iterative-stratification
- Multi-label Classification: https://scikit-learn.org/stable/modules/multiclass.html

---

## ✅ Checklist Pré-Modélisation

- [x] Dataset préprocessé et sauvegardé
- [x] Train/Test split réalisé avec stratification
- [x] Imputation values sauvegardées
- [x] MultiLabelBinarizer sauvegardé
- [x] Features textuelles nettoyées et standardisées
- [x] Features numériques/binaires créées
- [ ] TF-IDF vectorizer créé et sauvegardé
- [ ] Embeddings calculés et sauvegardés
- [ ] Stratégie de validation définie
- [ ] Métriques d'évaluation choisies
- [ ] Pipeline de modélisation testé sur petit échantillon

---

**Document généré le**: 14 Janvier 2026  
**Version**: 1.0  
**Auteur**: Assistant IA  
**Statut**: Section 2.2 (Embeddings) en cours d'exécution


**Date**: 14 Janvier 2026  
**Objectif Global**: Classification multi-label de problèmes algorithmiques selon 8 tags prioritaires  
**Tags prioritaires**: `math`, `graphs`, `strings`, `number theory`, `trees`, `geometry`, `games`, `probabilities`

---

## 📁 Architecture du Projet

```
illuin_challenge/
├── data/
│   ├── raw/code_classification_dataset/     # Dataset brut (4,982 samples)
│   └── processed/                            # Datasets preprocessés
│       ├── train_preprocessed.parquet       # Train: 3,183 samples, 79 features
│       ├── test_preprocessed.parquet        # Test: 796 samples, 78 features
│       ├── imputation_values.json           # Valeurs d'imputation (medians train)
│       └── mlb_encoder.pkl                  # MultiLabelBinarizer fitted sur train
├── notebooks/
│   ├── 01_eda.ipynb                         # Analyse exploratoire
│   ├── 02_preprocessing_pipeline.ipynb      # Pipeline de preprocessing
│   ├── 03_tfidf_v1.ipynb                    # Modèle TF-IDF
│   └── 04_embeddings_v1.ipynb               # Modèle Embeddings (en cours)
├── src/utils/                               # Fonctions utilitaires
└── docs/embeddings/                         # Visualisations embeddings
```

---

## 🔢 Vue d'Ensemble des Données

### Dataset Initial
- **Samples**: 4,982
- **Colonnes**: 21
- **Near-duplicates retirés**: 9 (détection sur `prob_desc_description_translated`)
- **Samples finaux**: 4,973

### Découpage Train/Test

#### 🎯 **Stratégie de Split** (CRITIQUE pour éviter le data leakage)

**Split Unique - Iterative Stratification** ✅ ACTUEL
- **Train: 3,183 samples (64% du total)**
- **Test: 796 samples (16% du total)**
- **Méthode**: `MultilabelStratifiedShuffleSplit`
- **Random State**: 42 (reproductibilité)
- **Différence moyenne de distribution**: 0.045% (excellente stratification)

**Caractéristiques du split:**
- Appliqué directement après la déduplication sur les 4,973 samples
- Garantit une distribution équilibrée des 8 labels multi-label entre train et test
- Split ratio: 80/20 (train/test)

**⚠️ IMPORTANT POUR LA MODÉLISATION:**
- **Toutes les opérations de fitting (imputation, TF-IDF, embeddings) DOIVENT être fittées sur `df_train` UNIQUEMENT**
- Les valeurs fittées sont ensuite appliquées à `df_test` (jamais l'inverse)
- Le split est réalisé AVANT toute opération de feature engineering avec fitting

### Distribution des Labels (après stratification)

| Label | Train Count | Train % | Test Count | Test % | Diff |
|-------|-------------|---------|------------|--------|------|
| **math** | 902 | 28.34% | 225 | 28.27% | 0.07% |
| **graphs** | 355 | 11.15% | 89 | 11.18% | 0.03% |
| **strings** | 274 | 8.61% | 68 | 8.54% | 0.07% |
| **number theory** | 228 | 7.16% | 57 | 7.16% | 0.00% |
| **trees** | 204 | 6.41% | 51 | 6.41% | 0.00% |
| **geometry** | 102 | 3.20% | 26 | 3.27% | 0.06% |
| **games** | 70 | 2.20% | 17 | 2.14% | 0.06% |
| **probabilities** | 62 | 1.95% | 15 | 1.88% | 0.06% |

**Moyenne de labels par document**: ~0.69 labels/doc (multi-label peu dense)

---

## 📓 Détail des Notebooks

---

### 01_eda.ipynb - Analyse Exploratoire des Données

#### 🎯 Objectif
Comprendre la structure, la qualité et les caractéristiques du dataset brut avant toute transformation.

#### 📥 Input
- `data/raw/code_classification_dataset/` (4,982 samples, 21 colonnes)

#### 📊 Analyses Réalisées

**1. Analyse de la Qualité des Données**
- Taux de valeurs manquantes par colonne
- Distribution des langages de programmation
- Distribution de la difficulté (`difficulty`)
- Valeurs aberrantes détectées (`difficulty = -1`)

**2. Analyse des Tags**
- 8 tags prioritaires identifiés
- Distribution des tags (déséquilibrée: math 29%, probabilities 2%)
- Analyse multi-label: ~1 tag par problème en moyenne

**3. Analyse Textuelle**
- Distribution des longueurs de texte (description, input_spec, output_spec, notes)
- Détection de LaTeX: ~55% des descriptions contiennent du LaTeX
- Analyse de la langue: 99.6% anglais, 0.4% autres langues

**4. Analyse LaTeX**
- Identification des patterns LaTeX: `$...$`, `$$...$$`, `$$$...$$$`, `\[...\]`, `\(...\)`
- Extraction des commandes LaTeX fréquentes: `\le`, `\ldots`, `\frac`, `\sum`, etc.
- Densité LaTeX par document

**5. Near-Duplicates**
- Détection de 9 groupes de near-duplicates (18 samples concernés)
- Normalisation de texte pour la détection

#### 📤 Output
- Statistiques descriptives (console)
- Identification des problèmes de qualité
- Liste des tags prioritaires → `src/config.py`

#### 🔑 Insights Clés
- Dataset principalement en anglais (traduction nécessaire pour <1%)
- LaTeX présent dans >50% des descriptions (traitement spécial requis)
- Distribution des tags très déséquilibrée (stratification critique)
- Valeurs manquantes: `prob_desc_notes` (27%), autres colonnes <2%

---

### 02_preprocessing_pipeline.ipynb - Pipeline de Preprocessing Complet

#### 🎯 Objectif
Préparer le dataset pour la modélisation en appliquant toutes les transformations nécessaires.

#### 📥 Input
- `data/raw/code_classification_dataset/` (4,982 samples)

#### 📤 Output
- `data/processed/train_preprocessed.parquet` (3,183 samples, 79 features)
- `data/processed/test_preprocessed.parquet` (796 samples, 78 features)
- `data/processed/imputation_values.json` (valeurs de médiane pour inference)
- `data/processed/mlb_encoder.pkl` (MultiLabelBinarizer fitted sur train)

---

#### 🛠️ Pipeline Détaillé (11 Étapes)

---

##### **ÉTAPE 1: Text Pattern Cleaning**

**Application**: Train + Test (pas de fitting)

**Opération**:
- Correction des patterns malformés dans `prob_desc_notes`
- Exemples: `NoteIN` → `Note: In`, `NoteThe` → `Note: The`

**Fonction**: `clean_text_patterns()`

**Colonnes modifiées**: `prob_desc_notes`

---

##### **ÉTAPE 2: Translation (English Normalization)**

**Application**: Train + Test (pas de fitting)

**Colonnes traduites** (si non-anglais détecté):
- `prob_desc_description` → `prob_desc_description_translated`
- `prob_desc_input_spec` → `prob_desc_input_spec_translated`
- `prob_desc_output_spec` → `prob_desc_output_spec_translated`
- `prob_desc_notes` → `prob_desc_notes_translated`

**Résultats**:
- Description: 18/4982 traduits (0.4%)
- Input spec: 30/4982 traduits (0.6%)
- Output spec: 28/4982 traduits (0.6%)
- Notes: 21/4982 traduits (0.4%)

**Fonction**: `translate_column()` avec `googletrans`

**Conservation du LaTeX**: Les patterns LaTeX sont préservés durant la traduction

---

##### **ÉTAPE 3: Near-Duplicate Detection & Removal**

**Application**: Dataset complet (avant split)

**Méthode**:
1. Normalisation du texte (lowercase, remove punctuation, strip whitespace)
2. Hashage SHA256 de `prob_desc_description_translated`
3. Détection des groupes avec hash identique
4. Suppression des duplicates (keep='first')

**Résultats**:
- 9 groupes de near-duplicates détectés
- 9 samples supprimés (0.18%)
- Dataset final: 4,973 samples

**Fonction**: `detect_near_duplicates()`

**Colonnes créées**: `prob_desc_description_translated_hash`

---

##### **ÉTAPE 4: Train/Test Split (Iterative Stratification)**

**📍 SPLIT CRITIQUE - Emplacement dans le pipeline**

**Application**: Après la déduplication, AVANT toute opération de fitting

**Méthode**: `MultilabelStratifiedShuffleSplit`
- Garantit une distribution équilibrée des 8 labels multi-label
- Test size: 20%
- Random state: 42

**Résultats**:
- **Train**: 3,183 samples (80%)
- **Test**: 796 samples (20%)
- Différence moyenne de distribution: 0.045%

**⚠️ SÉPARATION TRAIN/TEST**:
À partir de cette étape, **toutes les opérations suivantes sont appliquées séparément** sur train et test:
- **Operations SANS fitting**: Appliquées indépendamment aux 2 datasets
- **Operations AVEC fitting**: Fit sur train UNIQUEMENT, puis apply sur train et test

---

##### **ÉTAPE 5: Numeric Variable Conversion**

**Application**: Train + Test (pas de fitting, transformation déterministe)

**5.1 Time Limit Conversion**
- Colonne source: `prob_desc_time_limit` (string: "1 second", "2 seconds")
- Colonne créée: `time_limit_seconds` (float)
- Parsing: Extraction du nombre et conversion en secondes
- Valeurs uniques: 0.5s à 15.0s (majorité: 1s et 2s)

**5.2 Difficulty Cleaning**
- Valeurs invalides (`-1`) remplacées par `NaN`
- Train: 0 valeurs invalides
- Test: 0 valeurs invalides
- Imputation réalisée plus tard (Étape 10)

**Fonctions**: 
- `convert_time_limit_column()`
- `handle_difficulty_invalid_values()`

---

##### **ÉTAPE 6: Text/LaTeX Separation**

**Application**: Train + Test (pas de fitting)

**Colonne traitée**: `prob_desc_description_translated`

**Opérations de `preprocess_text_full()`**:

1. **Extraction des symboles LaTeX**:
   - Pattern: `\\([a-zA-Z]+)` → Capture `\sum`, `\frac`, `\le`, etc.
   - Stockage de la liste des symboles

2. **Suppression des blocs LaTeX**:
   - `$$$...$$$` → `LATEXBLOCK`
   - `$$...$$` → `LATEXBLOCK`
   - `$...$` → `LATEXBLOCK`
   - `\[...\]` → `LATEXBLOCK`
   - `\(...\)` → `LATEXBLOCK`

3. **Suppression des commandes LaTeX**:
   - `\command` → ` ` (espace)

4. **Normalisation**:
   - Espaces multiples → espace unique
   - Lowercase
   - Strip

**Colonnes créées**:
- `clean_description`: Texte sans LaTeX (string)
- `nb_latex_blocks`: Nombre de blocs LaTeX (int)
- `nb_latex_symbols`: Nombre de commandes LaTeX (int)
- `latex_density`: Ratio de caractères LaTeX / total (float)
- `latex_symbols_density`: Symboles LaTeX / mots (float)
- `latex_features_desc`: Dict complet des features LaTeX

**Résultats**:
- Train: 1,782/3,183 samples avec LaTeX (56.0%)
- Test: 428/796 samples avec LaTeX (53.8%)

**Fonction**: `preprocess_text_full()`

---

##### **ÉTAPE 7: LaTeX Feature Extraction (Binary Features)**

**Application**: FIT sur train, APPLY sur train + test

**Méthode**:

1. **Extraction de tous les symboles LaTeX** (train + test séparément):
   - Train: 98 symboles uniques détectés
   - Test: 62 symboles uniques détectés

2. **Sélection des top symboles** (FIT sur train):
   - Top N: 30 symboles les plus fréquents
   - Min frequency: 10 occurrences minimum
   - Symboles sélectionnés depuis le TRAIN uniquement

3. **Création de features binaires**:
   - Format: `has_{symbol}` (ex: `has_le`, `has_frac`, `has_sum`)
   - Train: 31 features créées
   - Test: 30 features créées (symboles du train appliqués au test)

**Top 10 symboles LaTeX** (exemples):
- `has_le`: 581 train (18.3%), 140 test (17.6%)
- `has_ldots`: 286 train (9.0%), 60 test (7.5%)
- `has_dots`: 256 train (8.0%), 70 test (8.8%)
- `has_leq`: 172 train (5.4%), 47 test (5.9%)
- `has_cdot`: 161 train (5.1%), 35 test (4.4%)

**Fonction**: `extract_latex_binary_features()`

**⚠️ Note**: Les symboles sont fittés sur train, donc certains symboles présents dans test mais absents de train ne seront pas détectés

---

##### **ÉTAPE 8: Text Length Features**

**Application**: Train + Test (pas de fitting, calculs déterministes)

**Colonne traitée**: `prob_desc_description_translated`

**Features créées** (4 features):

1. `prob_desc_description_translated_char_length`:
   - Nombre total de caractères
   - Train mean: 958.57, Test mean: 936.37

2. `prob_desc_description_translated_word_count`:
   - Nombre de mots (split sur whitespace)
   - Train mean: 168.76, Test mean: 165.41

3. `prob_desc_description_translated_numeric_ratio`:
   - Ratio de chiffres (0-9) dans le texte
   - Train mean: 0.01, Test mean: 0.01

4. `prob_desc_description_translated_latex_ratio`:
   - Copie de `latex_density` (déjà calculée)
   - Train mean: 0.10, Test mean: 0.10

**Fonction**: `create_text_length_features()`

---

##### **ÉTAPE 9: Text Concatenation (Unified Document)**

**Application**: Train + Test (pas de fitting)

**But**: Créer un document unique concaténant toutes les sections du problème

**9.1 Unified Document (avec LaTeX)**

**Colonnes concaténées** (avec balises):
- `[DESC]` + `prob_desc_description_translated`
- `[INPUT]` + `prob_desc_input_spec_translated`
- `[OUTPUT]` + `prob_desc_output_spec_translated`
- `[NOTE]` + `prob_desc_notes_translated`
- `[SAMPLE_INPUT]` + `prob_desc_sample_inputs`
- `[SAMPLE_OUTPUT]` + `prob_desc_sample_outputs`

**Colonne créée**: `unified_document`

**Longueur moyenne**:
- Train: 2,046 caractères
- Test: 1,990 caractères

**Fonction**: `create_unified_document()`

---

**9.2 Unified Document (sans LaTeX)**

**Opération**:
- Remplacement de tous les blocs LaTeX par le token `[LATEX]`
- Suppression des commandes LaTeX restantes

**Colonne créée**: `unified_document_without_latex`

**Résultats**:
- Train: 2,046 → 1,964 chars (réduction 4.0%)
- Test: 1,990 → 1,917 chars (réduction 3.7%)
- Documents avec [LATEX] token: 96% (train), 96% (test)

**Fonction**: `remove_latex_from_text(replacement_token='[LATEX]')`

**💡 Usage**:
- `unified_document`: Pour modèles capables de comprendre le LaTeX (ex: SPECTER)
- `unified_document_without_latex`: Pour modèles standards (TF-IDF, BERT classique)

---

##### **ÉTAPE 10: Target Encoding (Multi-Label)**

**Application**: FIT sur train, APPLY sur train + test

**10.1 Priority Tags Column**

**Opération**:
- Filtrage des tags pour ne garder que les 8 tags prioritaires
- Colonne source: `tags` (liste de tags)
- Colonne créée: `tags_priority` (liste filtrée)

**Résultats**:
- Train: 1,715/3,183 avec au moins 1 tag prioritaire (53.9%)
- Test: 428/796 avec au moins 1 tag prioritaire (53.8%)
- Les samples sans tag prioritaire sont gardés comme exemples négatifs

**Fonction**: `create_priority_tags_column()`

---

**10.2 Multi-Label Binary Encoding**

**Opération** (FIT sur train):
1. Fit `MultiLabelBinarizer` sur `df_train['tags_priority']`
2. Transform train et test avec le même encoder
3. Création de 8 colonnes binaires: `target_{tag}`

**Classes détectées**: 
```python
['games', 'geometry', 'graphs', 'math', 'number theory', 
 'probabilities', 'strings', 'trees']
```

**Colonnes créées** (8 colonnes target):
- `target_games`
- `target_geometry`
- `target_graphs`
- `target_math`
- `target_number_theory`
- `target_probabilities`
- `target_strings`
- `target_trees`

**Distribution** (voir tableau au début du document)

**⚠️ IMPORTANT**: Le `MultiLabelBinarizer` est sauvegardé dans `mlb_encoder.pkl` pour être utilisé en inference

---

##### **ÉTAPE 11: Missing Value Imputation**

**Application**: FIT sur train, APPLY sur train + test

**Colonnes imputées**:
- `difficulty`
- `time_limit_seconds`

**Stratégie**: Médiane (calculée sur train uniquement)

**Valeurs d'imputation** (fittées sur train):
```json
{
  "difficulty": 1700.0,
  "time_limit_seconds": 2.0
}
```

**Résultats**:
- Train difficulty: 0 valeurs imputées
- Test difficulty: 0 valeurs imputées
- Train time_limit: 0 valeurs imputées
- Test time_limit: 0 valeurs imputées

**⚠️ IMPORTANT**: Les valeurs de médiane sont sauvegardées dans `imputation_values.json` pour être utilisées en inference sur de nouveaux samples

**Fonction**: `impute_missing_values()` (custom, basée sur medians)

---

#### 📊 Résumé des Features Créées (79 features au total)

**Features Originales** (21):
- `prob_desc_*` (description, input_spec, output_spec, notes, etc.)
- `difficulty`, `time_limit`, `tags`, `src_uid`, `lang`, etc.

**Features de Traduction** (4):
- `*_translated` pour les 4 colonnes textuelles principales

**Features LaTeX** (5 + 31):
- `nb_latex_blocks`, `nb_latex_symbols`, `latex_density`, `latex_symbols_density`, `latex_features_desc`
- 31 features binaires `has_{symbol}`

**Features de Longueur de Texte** (4):
- `*_char_length`, `*_word_count`, `*_numeric_ratio`, `*_latex_ratio`

**Features de Document Unifié** (2):
- `unified_document`, `unified_document_without_latex`

**Features de Target** (8 + 1):
- 8 colonnes `target_{tag}` binaires
- `tags_priority` (liste filtrée)

**Features Numériques Transformées** (1):
- `time_limit_seconds` (conversion depuis string)

**Features Techniques** (1):
- `prob_desc_description_translated_hash` (pour near-duplicate detection)

**⚠️ Note**: Train a 79 features, Test a 78 (1 feature binaire LaTeX manquante car symbole absent du train)

---

### 03_tfidf_v1.ipynb - Modèle Baseline TF-IDF

#### 🎯 Objectif
Créer un modèle baseline multi-label classification basé sur TF-IDF + Classifieurs traditionnels.

#### 📥 Input
- `data/processed/train_preprocessed.parquet`
- `data/processed/test_preprocessed.parquet`

#### 🛠️ Approche

**1. Vectorisation TF-IDF**
- Colonne utilisée: `unified_document_without_latex`
- Paramètres TF-IDF:
  - `max_features`: 5000-10000
  - `ngram_range`: (1, 2) ou (1, 3)
  - `min_df`, `max_df`: Filtrage des termes trop rares/fréquents

**2. Modèles Testés**
- Logistic Regression (One-vs-Rest)
- Random Forest
- XGBoost
- LightGBM

**3. Métriques d'Évaluation**
- Precision, Recall, F1-score par label
- Macro/Micro averages
- Hamming Loss
- Subset Accuracy

#### 📤 Output
- Modèles sauvegardés (`.pkl`)
- Matrice TF-IDF sauvegardée
- Rapport de performance par label

#### 🔑 Insights
- Baseline pour comparer avec les approches deep learning
- Identification des labels faciles vs difficiles

---

### 04_embeddings_v1.ipynb - Modèle Embeddings (Sentence-Transformers)

#### 🎯 Objectif
Tester différents embedders et identifier les colonnes textuelles les plus discriminantes pour la classification.

#### 📥 Input
- `data/processed/train_preprocessed.parquet`

#### 🛠️ Analyses

**Section 2.1: Comparaison de 3 Embedders (sur `prob_desc_description` uniquement)**

**Embedders testés**:
1. `all-MiniLM-L6-v2` (384 dim)
2. `all-mpnet-base-v2` (768 dim)
3. `allenai-specter` (768 dim)

**Métriques de comparaison**:
- **Mean Δmean**: Différence moyenne de similarité cosine (positifs vs négatifs)
- **Centroid Similarity**: Similarité moyenne entre centroids de classes
- **Performance par tag**: Δmean individuel pour chaque des 8 tags

**Résultats**:

| Modèle | Dimensions | Mean Δmean ↑ | Centroid Sim ↓ |
|--------|-----------|--------------|----------------|
| **all-MiniLM-L6-v2** ✅ | 384 | **0.1470** | **0.7255** |
| all-mpnet-base-v2 | 768 | 0.1319 | 0.7533 |
| allenai-specter ❌ | 768 | 0.0372 | 0.9654 |

**Conclusion Section 2.1**:
- **Modèle recommandé**: `all-MiniLM-L6-v2`
- Meilleure séparation positifs/négatifs
- Moins de confusion inter-classes
- Plus léger et rapide (384 dim vs 768)

---

**Section 2.2: Analyse par Colonne Textuelle** 🔄 **EN COURS D'EXÉCUTION**

**Objectif**: Identifier quelle(s) colonne(s) contiennent le plus d'information discriminante

**Modèle utilisé**: `all-MiniLM-L6-v2` (gagnant de la Section 2.1)

**Colonnes testées**:
- `prob_desc_description_translated`
- `prob_desc_input_spec_translated`
- `prob_desc_output_spec_translated`
- `prob_desc_notes_translated`

**Métriques par colonne**:
- Mean Δmean (moyenne simple)
- **Weighted Δmean** (moyenne pondérée par le nombre de positifs par label)
- Centroid Similarity
- Coverage (% de documents non-vides)
- Δmean par (Tag × Colonne) → Heatmap 8×4

**Analyses prévues**:
1. Tableau comparatif trié par Mean Δmean
2. Graphiques:
   - Mean Δmean par colonne (barres horizontales)
   - Weighted Δmean par colonne (barres horizontales)
   - Coverage vs Performance (scatter plot)
   - Centroid Similarity par colonne
   - **Heatmap Δmean par (Tag × Colonne)** → Identifier si certains tags sont mieux capturés par certaines colonnes

**Questions à répondre**:
- Quelle colonne est la plus discriminante globalement ?
- Y a-t-il des complémentarités ? (ex: "graphs" mieux dans input_spec, "math" mieux dans description)
- Trade-off coverage vs qualité ? (notes peut être sparse mais discriminant)
- Recommandation pour le document unifié: garder toutes les colonnes ou filtrer ?

#### 📤 Output (prévu)
- Embeddings par colonne: `minilm_{colonne}_embeddings_train.npy`
- Centroids par colonne
- Heatmaps de similarité
- Graphiques comparatifs: `docs/embeddings/column_comparison_complete.png`

---

## 🎯 Variables Explicatives Disponibles pour la Modélisation

### 📊 Vue d'Ensemble

**Total Features**: 79 (train) / 78 (test)

---

### 1️⃣ **Features Textuelles Brutes** (pour TF-IDF / Embeddings)

**Colonnes disponibles** (toutes traduites en anglais):

| Colonne | Description | Coverage | Usage Recommandé |
|---------|-------------|----------|------------------|
| `prob_desc_description_translated` | Description du problème | 100% | ✅ **Principale** |
| `prob_desc_input_spec_translated` | Spécification des entrées | 99.3% | ✅ Complémentaire |
| `prob_desc_output_spec_translated` | Spécification des sorties | 98.4% | ✅ Complémentaire |
| `prob_desc_notes_translated` | Notes additionnelles | 72.9% | ⚠️ Sparse mais informatif |
| `unified_document` | Concaténation avec LaTeX | 100% | ✅ Pour modèles LaTeX-aware |
| `unified_document_without_latex` | Concaténation avec token [LATEX] | 100% | ✅ Pour modèles standards |
| `clean_description` | Description sans LaTeX | 100% | ✅ Alternative sans LaTeX |

**💡 Recommandations**:
- **TF-IDF**: Utiliser `unified_document_without_latex` (concaténation + LaTeX tokenisé)
- **Embeddings génériques**: Utiliser `unified_document_without_latex`
- **Embeddings spécialisés**: Tester colonnes individuelles puis concaténation pondérée
- **SPECTER**: Utiliser `unified_document` (avec LaTeX brut)

---

### 2️⃣ **Features Numériques LaTeX** (6 features)

| Feature | Type | Description | Mean (train) |
|---------|------|-------------|--------------|
| `nb_latex_blocks` | int | Nombre de blocs LaTeX ($$$, $$, $) | Variable |
| `nb_latex_symbols` | int | Nombre de commandes LaTeX (\sum, \frac, etc.) | Variable |
| `latex_density` | float | Ratio caractères LaTeX / total | 0.10 |
| `latex_symbols_density` | float | Symboles LaTeX / mots | Variable |
| `latex_features_desc` | dict | Dict complet (pas directement utilisable) | - |

**💡 Usage**:
- Features discriminantes pour certains tags (ex: "math" corrélé avec `latex_density`)
- Peuvent être utilisées comme features additionnelles dans un modèle hybride

---

### 3️⃣ **Features Binaires LaTeX** (31 features)

**Format**: `has_{symbol}` (ex: `has_le`, `has_frac`, `has_sum`)

**Top 10 symboles les plus fréquents**:
- `has_le`: 18.3% des documents (train)
- `has_ldots`: 9.0%
- `has_dots`: 8.0%
- `has_leq`: 5.4%
- `has_cdot`: 5.1%
- `has_times`: 3.6%
- `has_ne`: 3.5%
- `has_frac`: 3.4%
- `has_ge`: 2.9%
- `has_sum`: 2.1%

**💡 Usage**:
- Features interprétables pour certains tags (ex: `has_sum` pour "math")
- Peuvent améliorer les modèles traditionnels (Logistic Regression, XGBoost)
- Moins utiles pour deep learning (déjà capturé par embeddings)

---

### 4️⃣ **Features de Longueur de Texte** (4 features)

| Feature | Mean (train) | Mean (test) |
|---------|--------------|-------------|
| `prob_desc_description_translated_char_length` | 958.57 | 936.37 |
| `prob_desc_description_translated_word_count` | 168.76 | 165.41 |
| `prob_desc_description_translated_numeric_ratio` | 0.01 | 0.01 |
| `prob_desc_description_translated_latex_ratio` | 0.10 | 0.10 |

**💡 Usage**:
- Features de complexité du problème
- Peuvent aider à détecter les outliers
- Faible pouvoir discriminant seuls, mais utiles en combinaison

---

### 5️⃣ **Features Numériques Générales** (2 features)

| Feature | Type | Imputation | Mean (train) |
|---------|------|------------|--------------|
| `difficulty` | float | Médiane (1700.0) | 1700.0 |
| `time_limit_seconds` | float | Médiane (2.0) | 2.0 |

**💡 Usage**:
- Peuvent être corrélées avec certains tags
- `difficulty` peut aider à détecter les problèmes complexes

---

### 6️⃣ **Features Target** (8 colonnes binaires)

**Format**: `target_{tag}` (0 ou 1)

**Labels**:
- `target_math`, `target_graphs`, `target_strings`, `target_number_theory`
- `target_trees`, `target_geometry`, `target_games`, `target_probabilities`

**⚠️ ATTENTION**: Ces colonnes sont les **targets** à prédire, pas des features !

---

### 7️⃣ **Features TF-IDF** (à créer)

**Source**: `unified_document_without_latex`

**Paramètres recommandés**:
```python
TfidfVectorizer(
    max_features=10000,       # À tuner (5000-20000)
    ngram_range=(1, 2),       # Unigrammes + bigrammes
    min_df=5,                 # Ignorer termes trop rares
    max_df=0.7,               # Ignorer termes trop fréquents
    sublinear_tf=True,        # log(tf)
    norm='l2'                 # Normalisation L2
)
```

**Dimensions**: 5000-20000 features (sparse)

**💡 Usage**:
- FIT sur train uniquement
- Transform train et test avec le même vectorizer
- Modèles recommandés: Logistic Regression, XGBoost, LightGBM

---

### 8️⃣ **Features Embeddings** (à créer)

**Source**: Colonnes textuelles (description, input_spec, output_spec, notes)

**Modèle recommandé**: `all-MiniLM-L6-v2` (384 dimensions)

**Stratégies possibles**:

**Stratégie 1: Document Unifié**
- Embedder `unified_document_without_latex`
- Dimensions: 384
- Simple et rapide

**Stratégie 2: Colonnes Individuelles** (recommandé selon résultats Section 2.2)
- Embedder chaque colonne séparément
- Concaténer les embeddings: 384 × 4 = 1536 dimensions
- Ou pondérer selon l'importance de chaque colonne

**Stratégie 3: Mean/Max Pooling**
- Embedder chaque colonne
- Mean/Max pooling pour obtenir 384 dimensions

**💡 Usage**:
- Calculer embeddings sur train et test
- Sauvegarder matrices d'embeddings (`.npy`)
- Modèles recommandés: Logistic Regression, XGBoost, Neural Network

---

## 🎯 Stratégie de Modélisation Recommandée

### Phase 1: Modèles Baselines (Rapides à tester)

**1.1 TF-IDF + Logistic Regression**
- Features: TF-IDF (10k features) + 42 features numériques/binaires
- Modèle: `OneVsRestClassifier(LogisticRegression())`
- Temps: ~5-10 min
- But: Baseline solide et interprétable

**1.2 TF-IDF + XGBoost**
- Features: TF-IDF + features numériques
- Modèle: XGBoost multi-label (ou One-vs-Rest)
- Temps: ~20-30 min
- But: Meilleure performance que Logistic Regression

---

### Phase 2: Modèles Embeddings (Plus longs mais meilleurs)

**2.1 Embeddings + Logistic Regression**
- Features: Embeddings (384-1536 dim) + features numériques
- Modèle: `OneVsRestClassifier(LogisticRegression())`
- Temps: ~10-15 min (après calcul des embeddings)
- But: Tester la qualité des embeddings

**2.2 Embeddings + Neural Network**
- Features: Embeddings + features numériques
- Architecture: Dense layers (384 → 256 → 128 → 8)
- Activation finale: Sigmoid (multi-label)
- Loss: Binary Cross-Entropy
- Temps: ~30-60 min
- But: Meilleure modélisation des interactions non-linéaires

---

### Phase 3: Modèles Hybrides (Combinaison TF-IDF + Embeddings)

**3.1 Stacking**
- Niveau 1: TF-IDF + Logistic, Embeddings + Logistic, XGBoost
- Niveau 2: Meta-modèle (Logistic ou XGBoost)
- Temps: ~1-2h
- But: Combiner les forces de chaque approche

**3.2 Feature Concatenation**
- Features: TF-IDF (top 5000) + Embeddings (384) + features numériques
- Modèle: XGBoost ou Neural Network
- Temps: ~1h
- But: Utiliser toute l'information disponible

---

## ⚙️ Calibrage du Modèle - Bonnes Pratiques

### 🔒 Règles Critiques (Éviter le Data Leakage)

**1. Toujours FIT sur TRAIN uniquement**
- TF-IDF Vectorizer: `fit(df_train['text'])`
- Embedder: Calculer embeddings sur train, puis test
- Scaler (si utilisé): `fit(X_train)`
- Imputation: Médiane calculée sur train

**2. Jamais toucher au TEST avant l'évaluation finale**
- Le test sert UNIQUEMENT à évaluer la performance finale
- Toutes les décisions de modélisation se font sur train/validation

**3. Cross-Validation sur TRAIN**
- Utiliser `StratifiedKFold` (ou `MultilabelStratifiedKFold` si disponible)
- 5-10 folds
- But: Sélection d'hyperparamètres, comparaison de modèles

---

### 📊 Split Train/Validation (pour tuning)

**Option 1: Holdout Validation (recommandé pour vitesse)**
```python
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in msss.split(X_train, y_train):
    X_train_split, X_val = X_train[train_idx], X_train[val_idx]
    y_train_split, y_val = y_train[train_idx], y_train[val_idx]
```

**Option 2: K-Fold Cross-Validation (recommandé pour robustesse)**
```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(mskf.split(X_train, y_train)):
    # Train sur train_idx, évaluer sur val_idx
    pass
```

---

### 🎛️ Hyperparameter Tuning

**Méthode recommandée**: Grid Search ou Random Search avec Cross-Validation

**Exemple TF-IDF + Logistic Regression**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'tfidf__max_features': [5000, 10000, 15000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__min_df': [2, 5, 10],
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l1', 'l2']
}

# GridSearchCV avec cross-validation multi-label
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=mskf,  # MultilabelStratifiedKFold
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)
```

**Scoring metrics pour multi-label**:
- `f1_macro`: Moyenne des F1 par label (équilibrée)
- `f1_micro`: F1 global (favorise les labels fréquents)
- `f1_weighted`: F1 pondéré par fréquence
- Custom scorer: `make_scorer(custom_metric)`

---

### 📈 Métriques d'Évaluation

**Métriques par Label** (8 labels):
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=label_names))
```

**Métriques Globales**:
```python
from sklearn.metrics import hamming_loss, jaccard_score

# Hamming Loss (↓ mieux): % de labels mal prédits
hamming = hamming_loss(y_test, y_pred)

# Jaccard Score (↑ mieux): Intersection / Union
jaccard = jaccard_score(y_test, y_pred, average='samples')

# Subset Accuracy (↑ mieux): % de samples parfaitement prédits
subset_acc = (y_test == y_pred).all(axis=1).mean()
```

---

## 📁 Fichiers Importants pour la Modélisation

### Input (Déjà créés)

| Fichier | Description | Taille | Usage |
|---------|-------------|--------|-------|
| `train_preprocessed.parquet` | Dataset train | 3,183 samples | Entraînement + validation |
| `test_preprocessed.parquet` | Dataset test | 796 samples | Évaluation finale |
| `imputation_values.json` | Médianes train | - | Inference sur nouveaux samples |
| `mlb_encoder.pkl` | MultiLabelBinarizer | - | Inference sur nouveaux samples |

### Output (À créer durant modélisation)

| Fichier | Description | Usage |
|---------|-------------|-------|
| `tfidf_vectorizer.pkl` | TF-IDF fitted sur train | Inference |
| `embeddings_train.npy` | Embeddings train (3183, 384) | Modélisation |
| `embeddings_test.npy` | Embeddings test (796, 384) | Évaluation |
| `model_*.pkl` | Modèles entraînés | Inference |
| `results_*.json` | Métriques de performance | Comparaison |

---

## 🚀 Prochaines Étapes

### Immédiat (En cours)
1. ✅ Attendre fin de l'analyse Section 2.2 (colonnes textuelles)
2. ✅ Analyser les résultats de la heatmap (Tag × Colonne)

### Court Terme
3. **Décider stratégie embeddings**:
   - Document unifié OU colonnes séparées ?
   - Pondération des colonnes selon Δmean ?

4. **Créer features TF-IDF**:
   - Vectorizer sur `unified_document_without_latex`
   - Sauvegarder matrice sparse + vectorizer

5. **Créer features Embeddings**:
   - all-MiniLM-L6-v2 sur colonnes sélectionnées
   - Sauvegarder matrices .npy

### Moyen Terme
6. **Entraîner modèles baselines**:
   - TF-IDF + Logistic Regression
   - TF-IDF + XGBoost
   - Embeddings + Logistic Regression

7. **Comparer performances**:
   - F1-macro, F1 par label, Hamming Loss
   - Identifier labels difficiles

8. **Tuning hyperparamètres**:
   - Grid Search avec Cross-Validation
   - Sélectionner meilleur modèle

### Long Terme
9. **Modèles avancés**:
   - Neural Networks (Dense layers)
   - Stacking / Ensemble
   - Fine-tuning d'embedders (si ressources suffisantes)

10. **Évaluation finale**:
    - Performance sur `test_preprocessed.parquet`
    - Analyse des erreurs
    - Rapport final

---

## 📝 Notes Importantes

### ⚠️ Points d'Attention

1. **Data Leakage Prevention**:
   - TOUTES les opérations de fitting doivent être sur train uniquement
   - Valeurs d'imputation, encoders, vectorizers: fit(train) puis transform(test)

2. **Stratification Multi-Label**:
   - Utiliser `iterstrat.ml_stratifiers` pour la validation
   - Éviter `StratifiedKFold` standard (ne gère pas le multi-label)

3. **Imbalance des Labels**:
   - Math: 28% vs Probabilities: 2% (ratio 14:1)
   - Envisager class weights ou oversampling si besoin

4. **Features Sparse**:
   - `prob_desc_notes`: 27% de valeurs manquantes
   - TF-IDF: Matrices très sparse (~99%)
   - Utiliser sparse matrices pour économiser mémoire

5. **Computational Cost**:
   - Embeddings: ~5-10 min pour 3,979 samples
   - TF-IDF: Quasi-instantané
   - Neural Networks: ~30-60 min d'entraînement

---

## 🎓 Récapitulatif Méthodologique

### Découpage des Données

```
Dataset Initial (4,982)
    ↓
Déduplication (-9)
    ↓
Dataset Propre (4,973)
    ↓
[SPLIT UNIQUE] Iterative Stratification (80/20)
    ↓
┌─────────────────┬──────────────────┐
│   Train (3,183) │   Test (796)     │  ← Ne JAMAIS toucher avant éval finale
│        ↓        │                  │
│   [Toutes les   │                  │
│   opérations    │                  │
│   de fitting]   │                  │
│        ↓        │                  │
│   Entraînement  │                  │
│   + Validation  │                  │
│   (avec CV)     │                  │
└─────────────────┴──────────────────┘
```

### Workflow de Modélisation

```
1. Charger train_preprocessed.parquet
2. Séparer features (X) et targets (y)
3. Créer validation split (MultilabelStratifiedShuffleSplit)
4. Fit preprocessing sur X_train (TF-IDF, Scaler, etc.)
5. Transform X_train, X_val, X_test
6. Entraîner modèle sur (X_train, y_train)
7. Tuner hyperparamètres avec Cross-Validation
8. Évaluer sur X_val
9. Itérer (steps 4-8) jusqu'à satisfaction
10. Évaluation FINALE sur test UNIQUEMENT
```

---

## 📚 Ressources et Dépendances

### Packages Principaux
```python
# Data manipulation
pandas, numpy

# NLP
sentence-transformers (embeddings)
scikit-learn (TF-IDF, models)

# Multi-label
iterative-stratification (MultilabelStratifiedKFold)

# Models
xgboost, lightgbm
tensorflow/keras (si Neural Networks)

# Metrics
sklearn.metrics
```

### Documentation Utile
- Sentence-Transformers: https://www.sbert.net/
- Iterative Stratification: https://github.com/trent-b/iterative-stratification
- Multi-label Classification: https://scikit-learn.org/stable/modules/multiclass.html

---

## ✅ Checklist Pré-Modélisation

- [x] Dataset préprocessé et sauvegardé
- [x] Train/Test split réalisé avec stratification
- [x] Imputation values sauvegardées
- [x] MultiLabelBinarizer sauvegardé
- [x] Features textuelles nettoyées et standardisées
- [x] Features numériques/binaires créées
- [ ] TF-IDF vectorizer créé et sauvegardé
- [ ] Embeddings calculés et sauvegardés
- [ ] Stratégie de validation définie
- [ ] Métriques d'évaluation choisies
- [ ] Pipeline de modélisation testé sur petit échantillon

---

**Document généré le**: 14 Janvier 2026  
**Version**: 1.0  
**Auteur**: Assistant IA  
**Statut**: Section 2.2 (Embeddings) en cours d'exécution

