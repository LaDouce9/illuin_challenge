# 🚀 Guide d'Exécution - Notebook 04

**Notebook** : `04_preprocessing_pipeline.ipynb`  
**Status** : ✅ Prêt à l'emploi

---

## ⚡ Démarrage Rapide

### 1. Lancer l'Environnement Jupyter

#### Option A : Docker (Recommandé)
```bash
# Construire l'image (première fois uniquement)
make build

# Lancer le container
make run-jupyter

# URL d'accès
http://localhost:8888/lab?token=illuin2024
```

#### Option B : Local
```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter
jupyter lab

# Naviguer vers notebooks/04_preprocessing_pipeline.ipynb
```

### 2. Exécuter le Notebook

**Dans Jupyter Lab** :
1. Ouvrir `04_preprocessing_pipeline.ipynb`
2. Menu : `Kernel` → `Restart Kernel and Run All Cells...`
3. Attendre la fin (environ 5-10 minutes selon la machine)
4. Vérifier qu'il n'y a pas d'erreurs

---

## 📊 Ce Que Fait le Notebook

### Étapes Principales

1. **Chargement des données** (4,982 samples)
2. **Nettoyage** (patterns, traduction)
3. **Déduplication** (→ 4,973 samples)
4. **🎯 Train/Test Split** (80/20 sur `src_uid`)
   - Train : ~3,978 samples
   - Test : ~995 samples
5. **Feature Engineering**
   - Séparation text/LaTeX
   - Extraction features LaTeX (top 30 symboles **du train**)
   - Longueurs de texte
   - Unified documents
6. **Target Encoding** (fit sur train)
7. **Imputation** (fit sur train)
8. **Sauvegarde** des datasets et artifacts

---

## 📁 Fichiers Générés

Après exécution, vous trouverez dans `data/processed/` :

```
data/processed/
├── train_preprocessed.parquet      # Dataset d'entraînement
├── test_preprocessed.parquet       # Dataset de test
├── imputation_values.json          # Médianes pour inférence
└── mlb_encoder.pkl                 # Encoder multi-label pour inférence
```

### Détails des Fichiers

#### **train_preprocessed.parquet**
- ~3,978 samples
- ~76 features
- Prêt pour le training

#### **test_preprocessed.parquet**
- ~995 samples
- ~76 features (mêmes que train)
- Prêt pour l'évaluation

#### **imputation_values.json**
```json
{
  "difficulty": 1700.0,
  "time_limit_seconds": 2.0
}
```
**Usage** : Imputer les valeurs manquantes en production

#### **mlb_encoder.pkl**
- `MultiLabelBinarizer` fitted sur train
- 8 classes prioritaires
**Usage** : Encoder les tags en production

---

## ✅ Vérifications

### Après Exécution

Vérifiez que :

```bash
# 1. Les fichiers existent
ls -lh data/processed/

# 2. Les tailles sont correctes
# train_preprocessed.parquet : ~1-2 MB
# test_preprocessed.parquet  : ~300-500 KB
# imputation_values.json     : <1 KB
# mlb_encoder.pkl            : <10 KB

# 3. Vérifier le contenu (optionnel)
python -c "
import pandas as pd
train = pd.read_parquet('data/processed/train_preprocessed.parquet')
test = pd.read_parquet('data/processed/test_preprocessed.parquet')
print(f'Train: {train.shape}')
print(f'Test: {test.shape}')
print(f'Columns match: {set(train.columns) == set(test.columns)}')
"
```

**Sortie attendue** :
```
Train: (3978, 76)
Test: (995, 76)
Columns match: True
```

---

## 🔧 Dépannage

### Erreur : Module not found

```bash
# Vérifier que l'environnement est actif
# Dans le notebook, première cellule :
import sys
print(sys.executable)

# Doit pointer vers l'environnement du projet
```

**Solution** : Relancer le kernel ou reinstaller les dépendances

---

### Erreur : Memory Error

Le dataset est assez petit, mais si vous rencontrez des problèmes :

```python
# Dans la cellule de chargement, réduire la taille
df = df.sample(n=1000, random_state=42)  # Pour tester
```

---

### Erreur : fill_values not defined

**Cause** : Vous avez exécuté les cellules dans le désordre

**Solution** :
```
Kernel → Restart Kernel and Run All Cells...
```

---

### Erreur : mlb not defined (Section 13)

**Cause** : La variable `mlb` n'est pas définie (cellule 33 non exécutée)

**Solution** :
```
Kernel → Restart Kernel and Run All Cells...
```

---

## 📈 Temps d'Exécution

Temps indicatifs par section (machine standard) :

| Section | Temps Estimé |
|---------|--------------|
| 1-2. Setup + Cleaning | ~10s |
| 3. Translation | ~30-60s |
| 4. Deduplication + Split | ~5s |
| 5-6. Numeric + Text/LaTeX | ~30s |
| 7. LaTeX Features | ~20s |
| 8-9. Text Length + Unified | ~30s |
| 10-11. Target + Imputation | ~5s |
| 12-13. Validation + Save | ~10s |
| **TOTAL** | **~3-5 minutes** |

---

## 🎯 Prochaines Étapes

Après avoir exécuté ce notebook avec succès :

1. ✅ **Vérifier les fichiers** dans `data/processed/`
2. ✅ **Vérifier les shapes** train/test
3. ✅ **Passer au Notebook 05** : Modeling
   - TF-IDF / Embeddings
   - Model training
   - Evaluation

---

## 📚 Documentation

- **Status détaillé** : `notebooks/NOTEBOOK_04_STATUS.md`
- **Modifications** : `docs/Notebook_04_Modifications_Summary.md`
- **Stratégie split** : `docs/Train_Test_Split_Strategy.md`
- **Pipeline technique** : `docs/Preprocessing_Pipeline_Documentation.md`

---

## ❓ Questions Fréquentes

### Q : Puis-je modifier les paramètres du split ?

**Oui**, dans la Section 4.1 :
```python
df_train, df_test = train_test_split_grouped(
    df,
    group_column='src_uid',
    test_size=0.2,        # Modifier ici (0.2 = 20% test)
    random_state=42       # Modifier pour changer le découpage
)
```

---

### Q : Puis-je changer le nombre de symboles LaTeX ?

**Oui**, dans la Section 7 :
```python
df_train = extract_latex_binary_features(
    df_train,
    latex_stats_train,
    top_n=30,              # Modifier ici
    min_frequency=10,      # Fréquence minimum
    prefix='has_'
)
```

---

### Q : Où sont les valeurs d'imputation ?

**Dans le fichier** : `data/processed/imputation_values.json`

**Pour les lire** :
```python
import json
with open('data/processed/imputation_values.json', 'r') as f:
    fill_values = json.load(f)
print(fill_values)
```

---

## ✅ Checklist de Validation

Avant de passer à la modélisation :

- [ ] Le notebook s'exécute sans erreur
- [ ] 4 fichiers créés dans `data/processed/`
- [ ] `train.shape[0]` ≈ 3,978
- [ ] `test.shape[0]` ≈ 995
- [ ] `train.shape[1]` == `test.shape[1]` ≈ 76
- [ ] `imputation_values.json` contient 2 valeurs
- [ ] `mlb_encoder.pkl` existe

---

## 🎉 Félicitations !

Si toutes les étapes sont ✅, votre pipeline de preprocessing est **opérationnel** !

**Vous êtes prêt pour le Notebook 05 : Modeling** 🚀

