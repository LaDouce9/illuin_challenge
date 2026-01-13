# 📚 Databook Generation - Guide d'Utilisation

## 🎯 Objectif

Générer automatiquement un **dictionnaire de données complet** au format Excel avec :
- ✅ Liste exhaustive de toutes les variables
- ✅ Classification (origine vs créées)
- ✅ Statistiques descriptives
- ✅ Taux de valeurs manquantes
- ✅ Règles d'imputation documentées

---

## 🚀 Comment Générer le Databook

### Option 1 : Script Python (Recommandé)

```bash
# Depuis le répertoire notebooks/
python generate_databook.py
```

**Sortie** : `docs/databook_variables.xlsx`

---

### Option 2 : Depuis Jupyter Lab

```bash
# 1. Ouvrir Jupyter Lab
jupyter lab

# 2. Créer un nouveau notebook
# 3. Copier-coller le contenu de generate_databook.py
# 4. Exécuter toutes les cellules
```

---

## 📊 Contenu du Fichier Excel

Le fichier `databook_variables.xlsx` contient **7 feuilles** :

### **1. All_Variables** (Principale)
Dictionnaire complet de toutes les variables avec :
- Nom de la variable
- Catégorie (Original / Translated / LaTeX Feature / etc.)
- Description détaillée
- Type de données (int64, float64, object)
- Nombre total de valeurs
- **Nombre de valeurs manquantes**
- **Pourcentage de valeurs manquantes**
- **Règle d'imputation appliquée**
- Nombre de valeurs uniques
- Statistiques (Mean, Std, Min, Median, Max)
- Valeur la plus fréquente

### **2. Original_Variables**
Variables présentes dans le dataset brut (avant preprocessing)

### **3. Created_Variables**
Variables créées pendant le preprocessing :
- Features LaTeX
- Features de texte
- Variables traduites
- Unified documents
- etc.

### **4. Target_Variables**
Variables cibles pour la modélisation :
- `tags_priority` : Tags prioritaires filtrés
- `target_xxx` : Encodage binaire multi-label

### **5. Variables_With_Missing**
Variables contenant des valeurs manquantes (si applicable)

### **6. Summary_By_Category**
Résumé statistique par catégorie de variables

### **7. Imputation_Rules**
Règles d'imputation appliquées avec :
- Variable concernée
- Valeur d'imputation (médiane)
- Source (TRAIN set uniquement)

---

## 📋 Exemple de Sortie

### Extrait de All_Variables :

| Variable | Category | Description | Type | N_Total | N_Missing | Pct_Missing | Imputation_Rule | N_Unique | Mean | Median |
|----------|----------|-------------|------|---------|-----------|-------------|-----------------|----------|------|--------|
| difficulty | Original | Problem difficulty rating (1-3500) | int64 | 3978 | 0 | 0.00% | Median from TRAIN: 1700.0 | 2500 | 1650.5 | 1700.0 |
| has_le | LaTeX Feature | Binary: contains LaTeX symbol 'le' | int64 | 3978 | 0 | 0.00% | No missing values | 2 | 0.23 | 0.0 |
| target_math | Target (Encoded) | Binary target for tag: math | int64 | 3978 | 0 | 0.00% | No missing values | 2 | 0.28 | 0.0 |
| unified_document | Unified Document | Concatenated text from all fields | object | 3978 | 0 | 0.00% | No missing values | 3978 | N/A | N/A |

---

## 🔍 Catégories de Variables

Le databook classe les variables en **9 catégories** :

| Catégorie | Description | Exemple |
|-----------|-------------|---------|
| **Original** | Variables du dataset brut | `difficulty`, `src_uid`, `tags` |
| **Translated** | Variables traduites en anglais | `prob_desc_description_translated` |
| **LaTeX Feature** | Features extraites du LaTeX | `has_le`, `latex_density`, `nb_latex_blocks` |
| **Text Feature** | Features de texte | `char_length`, `word_count`, `latex_ratio` |
| **Target (Encoded)** | Target encodé en binaire | `target_math`, `target_graphs` |
| **Target (Priority)** | Tags prioritaires | `tags_priority` |
| **Unified Document** | Texte concaténé | `unified_document`, `unified_document_without_latex` |
| **Numeric Conversion** | Conversions numériques | `time_limit_seconds` |
| **Other Created** | Autres variables créées | Divers |

---

## 📊 Statistiques Résumées

### Exemple de sortie console :

```
==================================================
SUMMARY
==================================================

Total variables:              76
Original variables:           15
Created variables:            61

Variables with missing data:  0
Variables imputed:            2

==================================================
```

---

## 🎯 Cas d'Usage

### 1. **Documentation Projet**
- Partager avec l'équipe
- Documentation technique
- Onboarding de nouveaux membres

### 2. **Référence pour Modeling**
- Identifier les features disponibles
- Comprendre les types de données
- Vérifier les valeurs manquantes

### 3. **Audit Data Quality**
- Vérifier les taux de complétude
- Valider les règles d'imputation
- Identifier les problèmes potentiels

### 4. **Traçabilité**
- Documenter les transformations
- Justifier les choix d'imputation
- Historique des features créées

---

## ⚠️ Prérequis

### Packages Python

```bash
pip install pandas openpyxl
```

ou (si vous utilisez l'environnement Docker du projet) :
```bash
# Les packages sont déjà installés
```

### Fichiers Requis

Le script nécessite :
- ✅ `data/processed/train_preprocessed.parquet`
- ✅ `data/processed/test_preprocessed.parquet`
- ✅ `data/processed/imputation_values.json`

Ces fichiers sont générés par le notebook `04_preprocessing_pipeline.ipynb`.

---

## 🔧 Personnalisation

### Ajouter des Descriptions Personnalisées

Éditez le dictionnaire `descriptions` dans la fonction `get_variable_description()` :

```python
descriptions = {
    'my_variable': 'Description personnalisée de ma variable',
    # Ajouter d'autres variables ici
}
```

### Modifier les Catégories

Éditez la fonction `categorize_variable()` pour ajuster les règles de classification.

### Ajouter des Statistiques

Modifiez la fonction `get_variable_stats()` pour inclure d'autres métriques.

---

## 📝 Exemple d'Utilisation dans un Notebook

```python
# Charger le databook
import pandas as pd

databook = pd.read_excel('docs/databook_variables.xlsx', sheet_name='All_Variables')

# Explorer les variables LaTeX
latex_vars = databook[databook['Category'] == 'LaTeX Feature']
print(f"Nombre de features LaTeX: {len(latex_vars)}")

# Voir les variables avec valeurs manquantes
missing_vars = databook[databook['N_Missing'] > 0]
print(missing_vars[['Variable', 'Pct_Missing', 'Imputation_Rule']])

# Compter les variables par catégorie
print(databook['Category'].value_counts())
```

---

## ✅ Checklist de Validation

Après génération, vérifiez :

- [ ] Le fichier Excel a bien été créé dans `docs/`
- [ ] Il contient 7 feuilles
- [ ] La feuille "All_Variables" contient toutes les variables (76 dans notre cas)
- [ ] Les règles d'imputation sont documentées
- [ ] Les catégories sont correctement attribuées
- [ ] Les statistiques sont cohérentes

---

## 🎉 Résultat

Vous obtenez un **dictionnaire de données professionnel** et **complet** :

```
docs/databook_variables.xlsx
├── All_Variables (76 variables)
├── Original_Variables (15 variables)
├── Created_Variables (61 variables)
├── Target_Variables (9 variables)
├── Variables_With_Missing (0 variables après imputation)
├── Summary_By_Category
└── Imputation_Rules (2 règles)
```

**Prêt à être partagé, archivé, ou utilisé comme référence !** 📊✨

