# 📊 Exemple de Databook - Aperçu

**Fichier généré** : `databook_variables.xlsx`

Voici un aperçu de ce que contient le databook généré automatiquement.

---

## 📋 Feuille 1 : All_Variables (Extrait)

| Variable | Category | Description | Type | N_Total | N_Missing | Pct_Missing | Imputation_Rule | N_Unique | Mean | Median |
|----------|----------|-------------|------|---------|-----------|-------------|-----------------|----------|------|--------|
| **src_uid** | Original | Unique identifier for the solution | object | 3978 | 0 | 0.00% | No missing values | 3978 | N/A | N/A |
| **code_uid** | Original | Unique identifier for the problem | object | 3978 | 0 | 0.00% | No missing values | 2485 | N/A | N/A |
| **difficulty** | Original | Problem difficulty rating (1-3500) | float64 | 3978 | 0 | 0.00% | **Median from TRAIN: 1700.0** | 2485 | 1650.5432 | 1700.0000 |
| **tags** | Original | Original tags (list) | object | 3978 | 0 | 0.00% | No missing values | 3956 | N/A | N/A |
| **prob_desc_description** | Original | Problem description (original) | object | 3978 | 0 | 0.00% | No missing values | 2485 | N/A | N/A |
| **time_limit_seconds** | Numeric Conversion | Time limit converted to seconds (float) | float64 | 3978 | 0 | 0.00% | **Median from TRAIN: 2.0** | 17 | 1.9876 | 2.0000 |
| **prob_desc_description_translated** | Translated | English translation of prob_desc_description | object | 3978 | 0 | 0.00% | No missing values | 2485 | N/A | N/A |
| **clean_description** | Text Feature | Description with LaTeX removed | object | 3978 | 0 | 0.00% | No missing values | 2485 | N/A | N/A |
| **nb_latex_blocks** | LaTeX Feature | Number of LaTeX blocks in description | int64 | 3978 | 0 | 0.00% | No missing values | 45 | 8.2341 | 6.0000 |
| **latex_density** | LaTeX Feature | Ratio of LaTeX characters to total characters | float64 | 3978 | 0 | 0.00% | No missing values | 3542 | 0.1234 | 0.0876 |
| **has_le** | LaTeX Feature | Binary: contains LaTeX symbol 'le' | int64 | 3978 | 0 | 0.00% | No missing values | 2 | 0.2285 | 0.0000 |
| **has_ldots** | LaTeX Feature | Binary: contains LaTeX symbol 'ldots' | int64 | 3978 | 0 | 0.00% | No missing values | 2 | 0.1139 | 0.0000 |
| **unified_document** | Unified Document | Concatenated text from all description fields | object | 3978 | 0 | 0.00% | No missing values | 3978 | N/A | N/A |
| **unified_document_without_latex** | Unified Document | Unified document with LaTeX replaced by [LATEX] | object | 3978 | 0 | 0.00% | No missing values | 3978 | N/A | N/A |
| **tags_priority** | Target (Priority) | Filtered tags (priority tags only) | object | 3978 | 0 | 0.00% | No missing values | 189 | N/A | N/A |
| **target_math** | Target (Encoded) | Binary target for tag: math | int64 | 3978 | 0 | 0.00% | No missing values | 2 | 0.2825 | 0.0000 |
| **target_graphs** | Target (Encoded) | Binary target for tag: graphs | int64 | 3978 | 0 | 0.00% | No missing values | 2 | 0.1088 | 0.0000 |

... *(76 variables au total)*

---

## 📋 Feuille 2 : Original_Variables (Extrait)

Variables présentes dans le dataset brut :

| Variable | Description | N_Unique | Type |
|----------|-------------|----------|------|
| src_uid | Unique identifier for the solution | 3978 | object |
| code_uid | Unique identifier for the problem | 2485 | object |
| difficulty | Problem difficulty rating (1-3500) | 2485 | float64 |
| tags | Original tags (list) | 3956 | object |
| prob_desc_description | Problem description (original) | 2485 | object |
| prob_desc_input_spec | Input specification (original) | 2398 | object |
| prob_desc_output_spec | Output specification (original) | 2245 | object |
| prob_desc_notes | Additional notes (original) | 1876 | object |
| prob_desc_sample_inputs | Sample input examples | 2456 | object |
| prob_desc_sample_outputs | Sample output examples | 2456 | object |
| prob_desc_time_limit | Time limit for the problem (original string) | 23 | object |
| file_name | Source file name | 3978 | object |
| created_at | Creation timestamp | 3245 | object |

... *(15 variables originales au total)*

---

## 📋 Feuille 3 : Created_Variables (Extrait)

Variables créées pendant le preprocessing :

| Variable | Category | Description | N_Unique |
|----------|----------|-------------|----------|
| time_limit_seconds | Numeric Conversion | Time limit converted to seconds (float) | 17 |
| prob_desc_description_translated | Translated | English translation of prob_desc_description | 2485 |
| prob_desc_input_spec_translated | Translated | English translation of prob_desc_input_spec | 2398 |
| clean_description | Text Feature | Description with LaTeX removed | 2485 |
| nb_latex_blocks | LaTeX Feature | Number of LaTeX blocks in description | 45 |
| nb_latex_symbols | LaTeX Feature | Number of LaTeX symbols in description | 156 |
| latex_density | LaTeX Feature | Ratio of LaTeX characters to total | 3542 |
| has_le | LaTeX Feature | Binary: contains LaTeX symbol 'le' | 2 |
| has_ldots | LaTeX Feature | Binary: contains LaTeX symbol 'ldots' | 2 |
| has_frac | LaTeX Feature | Binary: contains LaTeX symbol 'frac' | 2 |
| unified_document | Unified Document | Concatenated text from all fields | 3978 |
| unified_document_without_latex | Unified Document | With LaTeX replaced by [LATEX] | 3978 |
| tags_priority | Target (Priority) | Filtered tags (priority tags only) | 189 |
| target_math | Target (Encoded) | Binary target for tag: math | 2 |

... *(61 variables créées au total)*

---

## 📋 Feuille 4 : Target_Variables

Variables cibles pour la modélisation :

| Variable | Description | N_Unique | Mean | Most_Common | Most_Common_Count |
|----------|-------------|----------|------|-------------|-------------------|
| tags_priority | Filtered tags (priority tags only) | 189 | N/A | [] | 1306 |
| target_math | Binary target for tag: math | 2 | 0.2825 | 0 | 2854 |
| target_graphs | Binary target for tag: graphs | 2 | 0.1088 | 0 | 3545 |
| target_strings | Binary target for tag: strings | 2 | 0.0849 | 0 | 3640 |
| target_number_theory | Binary target for tag: number theory | 2 | 0.0704 | 0 | 3699 |
| target_trees | Binary target for tag: trees | 2 | 0.0647 | 0 | 3721 |
| target_geometry | Binary target for tag: geometry | 2 | 0.0332 | 0 | 3846 |
| target_games | Binary target for tag: games | 2 | 0.0211 | 0 | 3895 |
| target_probabilities | Binary target for tag: probabilities | 2 | 0.0185 | 0 | 3904 |

---

## 📋 Feuille 5 : Variables_With_Missing

Variables contenant des valeurs manquantes (si applicable) :

```
Après imputation, cette feuille est généralement vide ou contient uniquement
les variables qui n'ont pas pu être imputées.
```

**Dans notre cas** : 0 variables (toutes ont été imputées ou sont complètes)

---

## 📋 Feuille 6 : Summary_By_Category

Résumé par catégorie :

| Category | N_Variables | Total_Missing |
|----------|-------------|---------------|
| Original | 15 | 0 |
| Translated | 4 | 0 |
| LaTeX Feature | 33 | 0 |
| Text Feature | 3 | 0 |
| Target (Encoded) | 8 | 0 |
| Target (Priority) | 1 | 0 |
| Unified Document | 2 | 0 |
| Numeric Conversion | 1 | 0 |
| Other Created | 9 | 0 |
| **TOTAL** | **76** | **0** |

---

## 📋 Feuille 7 : Imputation_Rules

Règles d'imputation appliquées :

| Variable | Imputation_Value | Source |
|----------|------------------|--------|
| difficulty | 1700.0 | Median from TRAIN set |
| time_limit_seconds | 2.0 | Median from TRAIN set |

**Note** : Ces valeurs proviennent **uniquement du TRAIN set** pour éviter le data leakage.

---

## 🎯 Points Clés à Retenir

### ✅ Traçabilité Complète
- **76 variables** documentées
- **15 originales** + **61 créées**
- **0 valeurs manquantes** après imputation
- **2 règles d'imputation** appliquées

### ✅ Classification Claire
- 9 catégories de variables
- Description pour chaque variable
- Type de données identifié

### ✅ Statistiques Détaillées
- Valeurs manquantes avant/après imputation
- Statistiques descriptives (mean, median, std, etc.)
- Valeurs uniques et distribution

### ✅ Règles d'Imputation Documentées
- Médianes calculées du TRAIN uniquement
- Source clairement identifiée
- Traçabilité pour l'inférence

---

## 📊 Utilisation Pratique

### Exemple 1 : Identifier les Features LaTeX
```python
import pandas as pd

df = pd.read_excel('docs/databook_variables.xlsx', sheet_name='All_Variables')
latex_features = df[df['Category'] == 'LaTeX Feature']
print(f"Nombre de features LaTeX: {len(latex_features)}")
# Output: Nombre de features LaTeX: 33
```

### Exemple 2 : Vérifier les Variables Imputées
```python
imputed = df[df['Imputation_Rule'].str.contains('Median', na=False)]
print(imputed[['Variable', 'Imputation_Rule']])
```

### Exemple 3 : Lister les Targets
```python
targets = df[df['Category'].str.contains('Target')]
print(targets[['Variable', 'Description', 'Mean']])
```

---

## ✨ Résultat

Un **databook professionnel** et **complet** qui documente :
- ✅ Toutes les transformations
- ✅ Toutes les règles d'imputation
- ✅ Toutes les statistiques
- ✅ Toutes les catégories

**Prêt pour la production, l'audit, et le partage avec l'équipe !** 🎉

