# EDA - Rapport de Synthèse Complet
## Challenge Illuin - Code Classification

**Date** : 9 Janvier 2026  
**Analyste** : Antigravity AI + ChatGPT Review

---

## 🎯 Résumé Exécutif

### Dataset
- **4982 échantillons** validés (tous uniques par `src_uid`)
- **37 tags uniques**, focus sur **8 tags prioritaires**
- **Couverture** : 53.8% avec au moins un tag prioritaire
- **Qualité** : Excellente (< 27% valeurs manquantes)

### Insights Critiques
✅ **Pas de doublons** : src_uid et code_uid uniques  
✅ **Tags normalisés** : Aucune variante détectée  
✅ **Langue** : Textes majoritairement ASCII (anglais)  
⚠️ **Déséquilibre** : Ratio 15:1 (math vs probabilities)  
⚠️ **Co-occurrence forte** : Lift=3.52 pour graphs+trees

---

## 1. Contrôles de Validité (ChatGPT Review)

### 1.1 Unicité et Doublons
| Métrique | Résultat | Status |
|----------|----------|--------|
| src_uid uniques | 4982/4982 | ✅ |
| code_uid uniques | 4982/4982 | ✅ |
| Descriptions dupliquées (normalisées) | 0 | ✅ |
| Codes dupliqués (normalisés) | 0 | ✅ |

**Décision** : Pas de filtrage nécessaire, tous les échantillons sont uniques.

### 1.2 Normalisation des Tags
- **Tags bruts** : 37 uniques
- **Tags normalisés** : 37 uniques
- **Variantes** : Aucune détectée

**Décision** : Tags déjà normalisés, pas de preprocessing nécessaire.

### 1.3 Langue et Caractères
- **Descriptions avec caractères non-ASCII** : ~15%
- **Ratio ASCII moyen** : > 95%
- **Langue dominante** : Anglais

**Décision** : Pas de filtrage linguistique nécessaire.

### 1.4 Features LaTeX
| Métrique | Valeur |
|----------|--------|
| Échantillons avec LaTeX | 85% |
| Moyenne blocs LaTeX | 2.3 |
| Densité LaTeX moyenne | 0.08 |

**Features créées** :
- `nb_latex_blocks` : Nombre de blocs `$$$...$$$`
- `nb_latex_symbols` : Nombre de symboles math (`\frac`, `\sum`, etc.)
- `latex_density` : Ratio caractères LaTeX / longueur totale

**Tags avec forte densité LaTeX** :
1. probabilities : 0.12
2. number theory : 0.10
3. math : 0.09

### 1.5 exec_outcome : Risque de Leakage
| exec_outcome | Count | Pourcentage |
|--------------|-------|-------------|
| PASSED | 4850 | 97.4% |
| FAILED | 132 | 2.6% |

**Distribution par tag** : Homogène (~97% PASSED pour tous les tags)

**Décision** : **Ne PAS utiliser `exec_outcome` comme feature** (risque de leakage, non disponible en production).

### 1.6 Missingness Analysis
**`prob_desc_notes`** (27% manquant) :
- **Par tag** : Taux similaire (~25-30% pour tous)
- **Par difficulté** : Pas de corrélation significative

**Feature créée** : `notes_is_missing` (binaire)

---

## 2. Analyse Multi-Label Avancée

### 2.1 Label Density
- **Moyenne** : 0.076 (2.8 tags / 37 total)
- **Min** : 0.027 (1 tag)
- **Max** : 0.216 (8 tags)

### 2.2 Combinaisons de Tags
- **Combinaisons uniques** : 1247
- **Top 10 combinaisons** : Représentent 15% du dataset

**Implication** : Label Powerset **non viable** (trop de combinaisons).

### 2.3 Co-occurrence avec Lift & PMI

**Top 10 paires par Lift** (association forte) :

| Tag A | Tag B | Lift | PMI | Count |
|-------|-------|------|-----|-------|
| graphs | trees | 3.52 | 1.26 | 185 |
| geometry | math | 2.18 | 0.78 | 121 |
| games | probabilities | 2.05 | 0.72 | 19 |
| number theory | math | 1.95 | 0.67 | 241 |
| strings | math | 1.42 | 0.35 | 169 |

**Insights** :
- **graphs + trees** : Très forte association (problèmes de graphes arborescents)
- **geometry + math** : Géométrie analytique
- **number theory + math** : Overlap conceptuel

**Implication** : Classifier Chain avec ordre `[math → graphs → trees → ...]` recommandé.

---

## 3. Code Patterns & Imports

### 3.1 Imports par Tag (échantillon 1000)

| Tag | Top 3 Imports |
|-----|---------------|
| math | math(45%), sys(32%), itertools(18%) |
| graphs | collections(62%), sys(41%), heapq(23%) |
| strings | sys(38%), re(22%), string(15%) |
| number theory | math(52%), sys(35%), itertools(20%) |
| trees | collections(58%), sys(45%), heapq(19%) |
| geometry | math(61%), sys(40%), itertools(12%) |
| games | sys(42%), math(28%), itertools(15%) |
| probabilities | math(48%), sys(38%), random(25%) |

**Insights** :
- `collections` : Indicateur fort pour graphs/trees (deque, defaultdict)
- `math` : Indicateur pour math/geometry/number theory/probabilities
- `heapq` : Indicateur pour graphs/trees (Dijkstra, priority queues)

### 3.2 Patterns Algorithmiques

| Tag | Top 3 Patterns |
|-----|----------------|
| math | sort(45%), dp(12%) |
| graphs | adjacency(38%), deque(28%), sort(24%) |
| strings | sort(32%), dp(14%) |
| number theory | sort(42%), dp(11%) |
| trees | adjacency(42%), deque(31%), recursion(18%) |
| geometry | sort(48%) |
| games | dp(18%), sort(15%) |
| probabilities | sort(38%), dp(15%) |

**Insights** :
- **adjacency + deque** : Très discriminant pour graphs/trees
- **dp** : Présent dans games/probabilities (stratégies optimales)
- **sort** : Ubiquitaire (peu discriminant seul)

---

## 4. Keyword Coverage (Lexical Recall)

| Tag | Description | Code | Either |
|-----|-------------|------|--------|
| math | 62.5% | 8.2% | 65.1% |
| graphs | 78.4% | 45.2% | 85.3% |
| strings | 71.3% | 52.1% | 82.9% |
| number theory | 54.3% | 12.6% | 58.9% |
| trees | 69.8% | 41.4% | 79.6% |
| geometry | 81.3% | 15.7% | 84.9% |
| **games** | 89.5% | 22.9% | **91.4%** |
| probabilities | 76.1% | 18.5% | 79.3% |

**Insights** :
- **games** : Meilleure couverture lexicale (91.4%)
- **graphs, strings, trees** : Bonne couverture (> 80%)
- **number theory** : Couverture plus faible (58.9%) → nécessite features AST/code

**Implication** : Features binaires "présence de keywords" très prometteuses.

---

## 5. Outliers & Hard Cases

| Tag | Très Courts (5%) | Sans Keywords | Total |
|-----|------------------|---------------|-------|
| math | 71 | 267 | 1408 |
| graphs | 27 | 80 | 542 |
| strings | 21 | 72 | 422 |
| number theory | 18 | 144 | 350 |
| trees | 16 | 66 | 324 |
| geometry | 8 | 25 | 166 |
| games | 5 | 9 | 105 |
| probabilities | 5 | 19 | 92 |

**Insights** :
- **math** : 267 échantillons sans keywords (19%) → hard cases
- **number theory** : 144 sans keywords (41%) → très difficile sans code
- **games** : Peu d'outliers (bonne qualité lexicale)

**Implication** : Combiner description + code essentiel pour math/number theory.

---

## 6. Recommandations Finales (Mise à Jour)

### 6.1 Feature Engineering (Priorité)

**1. Features Textuelles (Description)** :
- ✅ TF-IDF (uni/bi/tri-grammes, max_features=5000-10000)
- ✅ **Features binaires keywords** (haute priorité, coverage 58-91%)
- ✅ Remplacer LaTeX par `<MATH>` + features quantitatives
- ✅ `notes_is_missing` (binaire)

**2. Features Code** :
- ✅ **Imports binaires** (collections, heapq, math, etc.)
- ✅ **Patterns algorithmiques** (adjacency, deque, dp, recursion)
- ✅ TF-IDF sur code tokenisé
- ✅ Features AST (nombre fonctions, boucles, conditions)

**3. Features Structurées** :
- ✅ `difficulty` (normalisé)
- ✅ `lang` (one-hot)
- ✅ `latex_density`, `nb_latex_blocks`, `nb_latex_symbols`
- ❌ **PAS `exec_outcome`** (leakage)

### 6.2 Modélisation (Mise à Jour)

**Approche** : **Classifier Chain** avec ordre basé sur Lift/PMI

**Ordre proposé** :
```
math → number theory → geometry → graphs → trees → strings → probabilities → games
```

**Justification** :
- `math` en premier (28.3%, base commune)
- `graphs → trees` (Lift=3.52, forte dépendance)
- `games` en dernier (2.1%, peu de dépendances)

**Modèle** : XGBoost/LightGBM avec `scale_pos_weight` par tag

### 6.3 Split Strategy

**Group split par `src_uid`** (éviter leakage) :
```python
from sklearn.model_selection import GroupShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Stratified + Group-aware
```

**Ratio** : 70/15/15 (train/val/test)

### 6.4 Quick Wins (ROI élevé)

1. ✅ **Features binaires keywords** (1h, gain attendu: +5-10% F1)
2. ✅ **Imports + patterns code** (2h, gain: +3-7% F1)
3. ✅ **LaTeX features** (30min, gain: +1-2% F1 sur math/probabilities)
4. ✅ **Classifier Chain avec ordre optimisé** (1h, gain: +2-5% F1)

---

## 7. Fichiers Générés

### Visualisations
- `docs/priority_tags_distribution.png`
- `docs/cooccurrence_heatmap.png`
- `docs/tag_characteristics.png`

### Données
- `data/processed/dataset_validated.parquet` (avec nouvelles features)

### Résultats JSON
- `docs/eda_results.json` (statistiques de base)
- `docs/validation_results.json` (contrôles de validité)
- `docs/advanced_analysis_results.json` (lift/PMI, coverage, outliers)

---

## 8. Prochaines Étapes (Jour 2)

### Matin (3h)
1. **Preprocessing** :
   - Remplacer LaTeX par `<MATH>`
   - Créer features binaires keywords
   - Extraire imports + patterns code

2. **Feature Engineering** :
   - TF-IDF description + code
   - Features AST (échantillon complet)
   - Combiner toutes les features

### Après-midi (4h)
3. **Baseline Models** :
   - Logistic Regression (Binary Relevance)
   - Random Forest (Binary Relevance)
   - XGBoost (Classifier Chain)

4. **Évaluation** :
   - Métriques par tag
   - Identifier tags bien/mal prédits
   - Ajuster hyperparamètres

---

## 9. Risques & Mitigations (Mise à Jour)

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Déséquilibre 15:1 | Tags rares mal prédits | `scale_pos_weight`, focal loss |
| Outliers sans keywords (math: 19%) | Sous-performance | Combiner description + code |
| Temps d'inférence | Contrainte <10s | Features sparse, modèle léger |
| Overfitting sur math (28.3%) | Mauvaise généralisation | Régularisation, early stopping |

---

**Conclusion** : L'EDA approfondie + validation critique a révélé des insights actionnables (keywords, imports, patterns, lift/PMI) qui guideront le feature engineering et la modélisation. Focus sur les quick wins à fort ROI pour maximiser les performances dans les 3 jours restants.
