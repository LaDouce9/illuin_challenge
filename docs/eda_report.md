# Rapport d'Analyse Exploratoire des Données (EDA)
## Challenge Illuin - Classification de Code

**Date** : 9 Janvier 2026  
**Analyste** : Antigravity AI  
**Dataset** : xCodeEval (sous-ensemble Codeforces)

---

## 1. Vue d'Ensemble du Dataset

### 1.1 Caractéristiques Générales
- **Nombre total d'échantillons** : 4982
- **Nombre de colonnes** : 21
- **Format** : JSON (un fichier par échantillon)

### 1.2 Colonnes Principales
| Colonne | Type | Description |
|---------|------|-------------|
| `prob_desc_description` | object | Description du problème algorithmique |
| `source_code` | object | Code Python solution |
| `tags` | object | Liste des tags algorithmiques (cible) |
| `difficulty` | float64 | Niveau de difficulté (800-3500) |
| `lang` | object | Langage (Python 2/3, PyPy 3) |
| `exec_outcome` | object | Résultat d'exécution (PASSED/FAILED) |

---

## 2. Qualité des Données

### 2.1 Valeurs Manquantes

**À compléter après exécution du notebook**

| Colonne | Nombre | Pourcentage |
|---------|--------|-------------|
| ... | ... | ... |

### 2.2 Décisions de Traitement

**À documenter :**
- Colonnes avec valeurs manquantes significatives
- Stratégie de traitement (suppression, imputation, conservation)
- Justification des choix

---

## 3. Analyse des Tags

### 3.1 Tags Prioritaires

Les 8 tags prioritaires du challenge :
```
['math', 'graphs', 'strings', 'number theory', 
 'trees', 'geometry', 'games', 'probabilities']
```

### 3.2 Distribution des Tags Prioritaires

**À compléter avec les résultats :**

| Tag | Count | Fréquence (%) | Rang Global |
|-----|-------|---------------|-------------|
| math | ... | ... | ... |
| graphs | ... | ... | ... |
| strings | ... | ... | ... |
| number theory | ... | ... | ... |
| trees | ... | ... | ... |
| geometry | ... | ... | ... |
| games | ... | ... | ... |
| probabilities | ... | ... | ... |

### 3.3 Couverture des Tags Prioritaires

- **Échantillons avec au moins un tag prioritaire** : ... (... %)
- **Échantillons sans tag prioritaire** : ... (... %)

**Décision Stratégique** :
> Faut-il filtrer le dataset pour ne garder que les exemples avec tags prioritaires ?
> 
> **Recommandation** : [À compléter]

### 3.4 Distribution Multi-Label

- **Nombre moyen de tags par échantillon** : ...
- **Nombre moyen de tags prioritaires** : ...
- **Distribution** : [Histogramme à insérer]

---

## 4. Co-occurrence des Tags

### 4.1 Matrice de Co-occurrence

**Insights clés** :
- Paires de tags fréquemment associées
- Tags mutuellement exclusifs
- Implications pour la modélisation

### 4.2 Top 10 Paires de Tags

**À compléter :**

1. `tag1` + `tag2` : ... occurrences
2. ...

---

## 5. Analyse par Tag Prioritaire

### 5.1 Caractéristiques Moyennes

**À compléter avec tableau comparatif :**

| Tag | Difficulté Moy. | Longueur Code | Longueur Desc. | Langage Dominant |
|-----|-----------------|---------------|----------------|------------------|
| ... | ... | ... | ... | ... |

### 5.2 Observations Clés

**Par tag** :
- **math** : [Observations]
- **graphs** : [Observations]
- **strings** : [Observations]
- **number theory** : [Observations]
- **trees** : [Observations]
- **geometry** : [Observations]
- **games** : [Observations]
- **probabilities** : [Observations]

---

## 6. Analyse NLP des Descriptions

### 6.1 Mots-Clés Discriminants

**Top mots-clés par tag** (extraits des nuages de mots) :

- **math** : [Liste des mots-clés]
- **graphs** : [Liste des mots-clés]
- **strings** : [Liste des mots-clés]
- ...

### 6.2 Insights pour le Feature Engineering

**Mots-clés à utiliser comme features** :
- Présence de "graph", "node", "edge" → indicateur pour `graphs`
- Présence de "tree", "parent", "child" → indicateur pour `trees`
- Présence de "prime", "divisor", "gcd" → indicateur pour `number theory`
- ...

---

## 7. Analyse du Code Source

### 7.1 Statistiques de Longueur

| Métrique | Moyenne | Médiane | Min | Max |
|----------|---------|---------|-----|-----|
| Caractères | ... | ... | ... | ... |
| Mots | ... | ... | ... | ... |
| Lignes | ... | ... | ... | ... |

### 7.2 Complexité du Code

**Métriques AST (échantillon)** :

| Métrique | Moyenne | Médiane |
|----------|---------|---------|
| Nombre de fonctions | ... | ... |
| Nombre de boucles | ... | ... |
| Nombre de conditions | ... | ... |
| Profondeur AST | ... | ... |

### 7.3 Distribution des Langages

| Langage | Count | Pourcentage |
|---------|-------|-------------|
| Python 3 | ... | ... |
| PyPy 3 | ... | ... |
| Python 2 | ... | ... |

**Décision** : Faut-il normaliser les différences Python 2/3 ?

---

## 8. Déséquilibre des Classes

### 8.1 Analyse du Déséquilibre

**Ratio max/min** : ...  
**Tags sous-représentés** : ...  
**Tags sur-représentés** : ...

### 8.2 Stratégies Proposées

1. **Weighted Loss** : Pondérer la loss en fonction de la fréquence
2. **Oversampling** : SMOTE ou duplication pour tags rares
3. **Undersampling** : Réduire les tags fréquents (non recommandé)
4. **Focal Loss** : Mettre l'accent sur les exemples difficiles

**Recommandation** : [À compléter]

---

## 9. Exemples Représentatifs

### 9.1 Exemples par Tag

**Voir notebook section 9 pour exemples détaillés**

### 9.2 Cas Limites Identifiés

- Exemples avec tags ambigus
- Exemples multi-tags complexes
- Exemples avec code très court/long

---

## 10. Recommandations pour le Feature Engineering

### 10.1 Features Textuelles (Description)

1. **TF-IDF** :
   - Uni/bi/tri-grammes
   - Max features : 5000-10000
   - Min_df : 2-5

2. **Features Binaires** :
   - Présence de mots-clés algorithmiques
   - Dictionnaire par tag

3. **Features Numériques** :
   - Longueur de la description
   - Nombre de formules LaTeX

### 10.2 Features Code Source

1. **TF-IDF sur le code** :
   - Tokenisation spécifique au code
   - Identifier les patterns récurrents

2. **Features AST** :
   - Nombre de fonctions/classes
   - Nombre de boucles/conditions
   - Profondeur de l'arbre
   - Complexité cyclomatique

3. **Features Simples** :
   - Longueur du code (lignes, caractères)
   - Imports utilisés (collections, itertools, math, etc.)

### 10.3 Features Structurées

- `difficulty` (numérique, normalisé)
- `lang` (one-hot encoding)
- `exec_outcome` (binaire)

---

## 11. Recommandations pour la Modélisation

### 11.1 Approche Multi-Label

**Recommandation** : **Classifier Chain** avec XGBoost/LightGBM

**Justification** :
- Capture les dépendances entre tags
- Performant sur CPU
- Interprétable

**Alternatives** :
- Binary Relevance (baseline simple)
- Label Powerset (si peu de combinaisons)

### 11.2 Gestion du Déséquilibre

- Utiliser `scale_pos_weight` dans XGBoost
- Ou weighted loss dans sklearn

### 11.3 Split Train/Val/Test

**Stratégie** : Stratified split pour multi-label

```python
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# 70/15/15 split
train: 3487 samples
val:   747 samples
test:  748 samples
```

### 11.4 Métriques d'Évaluation

**Métriques principales** :
1. **Micro F1-Score** (pondéré par fréquence)
2. **Macro F1-Score** (moyenne non pondérée)
3. **Hamming Loss**
4. **F1-Score par tag** (pour identifier les tags bien/mal prédits)

---

## 12. Prochaines Étapes

### Jour 2 (Vendredi)
1. **Preprocessing** :
   - Nettoyage des descriptions (LaTeX, caractères spéciaux)
   - Normalisation du code (Python 2 vs 3)
   - Gestion des valeurs manquantes

2. **Feature Engineering** :
   - Implémentation des features textuelles
   - Extraction des features AST
   - Création du dataset final

3. **Baseline Models** :
   - Logistic Regression (One-vs-Rest)
   - Random Forest
   - Évaluation initiale

### Jour 3 (Samedi)
- Modèles avancés (XGBoost/LightGBM)
- Hyperparameter tuning
- Ensembling

### Jour 4 (Dimanche)
- CLI
- Slides

---

## 13. Conclusion

**Résumé des insights clés** :
1. [À compléter]
2. [À compléter]
3. [À compléter]

**Risques identifiés** :
- Déséquilibre des classes
- Tags rares difficiles à prédire
- Différences Python 2/3

**Opportunités** :
- Mots-clés discriminants identifiés
- Features AST prometteuses
- Co-occurrences exploitables

---

**Annexes** :
- Notebook complet : `notebooks/01_eda_analysis.ipynb`
- Visualisations : `docs/wordclouds_by_tag.png`
- Dataset enrichi : `data/processed/dataset_with_eda_features.parquet`
