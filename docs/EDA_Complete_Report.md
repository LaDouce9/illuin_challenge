# 📊 Rapport d'Analyse Exploratoire des Données (EDA) Complet
## Challenge : Classification de Code Algorithmique

---

**Date de l'analyse** : Janvier 2026  
**Dataset** : 4,982 problèmes algorithmiques avec code source  
**Objectif** : Prédire les tags algorithmiques associés à chaque problème  
**Tags prioritaires** : `math`, `graphs`, `strings`, `number theory`, `trees`, `geometry`, `games`, `probabilities`

---

## 📋 Table des Matières

1. [Vue d'Ensemble du Dataset](#1-vue-densemble-du-dataset)
2. [Contrôles de Validité Critiques](#2-contrôles-de-validité-critiques)
3. [Analyse des Tags Prioritaires](#3-analyse-des-tags-prioritaires)
4. [Distribution et Structure Multi-Label](#4-distribution-et-structure-multi-label)
5. [Co-occurrence et Associations de Tags](#5-co-occurrence-et-associations-de-tags)
6. [Analyse LaTeX et Symboles Mathématiques](#6-analyse-latex-et-symboles-mathématiques)
7. [Analyse du Code Source](#7-analyse-du-code-source)
8. [Keyword Coverage (Recall Lexical)](#8-keyword-coverage-recall-lexical)
9. [Outliers et Hard Cases](#9-outliers-et-hard-cases)
10. [Insights Clés et Recommandations](#10-insights-clés-et-recommandations)

---

## 1. Vue d'Ensemble du Dataset

### 📦 Caractéristiques Générales

- **Nombre d'échantillons** : 4,982 problèmes
- **Nombre de colonnes** : 21 features
- **Tags uniques totaux** : 37 tags
- **Tags prioritaires** : 8 tags sur 37

### 🗂️ Colonnes Principales

| Colonne | Type | Description | Utilité |
|---------|------|-------------|---------|
| `prob_desc_description` | Texte | Énoncé du problème | **Essentielle** - NLP, mots-clés |
| `source_code` | Texte | Code source Python/C++ | **Essentielle** - Patterns algorithmiques |
| `tags` | Liste | Tags multi-label | **Target** - Variable à prédire |
| `difficulty` | Numérique | Score de difficulté | **Importante** - Feature prédictive |
| `prob_desc_notes` | Texte | Notes additionnelles | Analyse de missingness |
| `exec_outcome` | Catégoriel | Résultat d'exécution | ⚠️ **Ne pas utiliser** (100% PASSED) |

---

## 2. Contrôles de Validité Critiques

### ✅ 2.1 Unicité des Identifiants

**Objectif** : Éviter les doublons qui causeraient du data leakage dans le split train/test.

| Métrique | Résultat | Statut |
|----------|----------|--------|
| `src_uid` uniques | 4,982 / 4,982 | ✅ **100% uniques** |
| `code_uid` uniques | 4,982 / 4,982 | ✅ **100% uniques** |

**✅ CONCLUSION** : Pas de doublons stricts sur les identifiants.

---

### ⚠️ 2.2 Détection de Near-Duplicates

**Méthode** : Hashing MD5 après normalisation agressive (suppression LaTeX, caractères spéciaux, lowercase).

| Métrique | Résultat | Statut |
|----------|----------|--------|
| Descriptions normalisées uniques | 4,970 / 4,982 | ⚠️ **12 doublons** |
| **Groupes de descriptions similaires** | **11 groupes** | ⚠️ **23 échantillons concernés** |
| Codes normalisés uniques | 4,981 / 4,982 | ⚠️ **2 doublons** |

**⚠️ RISQUE IDENTIFIÉ** : 11 groupes de descriptions similaires (23 échantillons au total).

**🔧 ACTION RECOMMANDÉE** :
- Utiliser un **stratified split multi-label** pour éviter qu'un même problème (avec des codes différents) soit à la fois dans train et test
- Vérifier manuellement les near-duplicates
- Envisager de regrouper ou supprimer les doublons

---

### ✅ 2.3 Normalisation des Tags

| Métrique | Résultat |
|----------|----------|
| Tags bruts uniques | 37 |
| Tags normalisés uniques | 37 |

**✅ CONCLUSION** : Tous les tags sont déjà correctement normalisés (pas de variantes de casse ou d'espaces).

---

### ✅ 2.4 Analyse de la Langue

| Métrique | Résultat |
|----------|----------|
| Descriptions avec caractères non-ASCII | 87 / 4,982 (1.7%) |
| Ratio ASCII moyen global | 0.993 |
| Ratio ASCII moyen (descriptions avec non-ASCII) | 0.790 |

**✅ CONCLUSION** : Le dataset est très homogène en anglais (99.3% ASCII). Les 1.7% de descriptions avec caractères spéciaux ne posent pas de problème majeur.

**Exemples de caractères non-ASCII** :
- Symboles mathématiques Unicode (·, ×, etc.)
- Caractères grecs en texte (α, β, etc.)

---

### ❌ 2.5 Analyse exec_outcome (Risque de Leakage)

| Métrique | Résultat |
|----------|----------|
| Distribution exec_outcome | PASSED: 4,982 (100%) |

**❌ DÉCISION CRITIQUE** : **NE PAS utiliser `exec_outcome` comme feature**

**Raisons** :
1. **100% PASSED** → Pas informatif
2. **Leakage** → Non disponible en production (on ne va pas exécuter le code en inférence)
3. **Éthique** → Utiliser cette info serait tricher

---

### 📊 2.6 Analyse de Missingness (prob_desc_notes)

| Métrique | Résultat |
|----------|----------|
| Notes manquantes | 1,350 / 4,982 (27.1%) |

**Distribution par tag prioritaire** :

| Tag | Taux de notes manquantes |
|-----|--------------------------|
| `games` | 18.1% |
| `math` | 22.9% |
| `number theory` | 24.6% |
| `trees` | 25.0% |
| `geometry` | 25.9% |
| `strings` | 26.8% |
| `graphs` | 28.6% |
| `probabilities` | 29.3% |

**Distribution par difficulté** :

| Difficulté | Taux de notes manquantes |
|------------|--------------------------|
| Easy | 22.2% |
| Medium | 28.7% |
| Hard | 29.7% |
| Very Hard | 28.1% |

**💡 INSIGHT** : 
- Les problèmes faciles (`games`, `math`) ont moins de notes manquantes
- Les problèmes difficiles ont plus de notes manquantes
- ➡️ **Feature créée** : `notes_is_missing` (binaire) peut être prédictive

---

## 3. Analyse des Tags Prioritaires

### 📊 3.1 Distribution des 8 Tags Prioritaires

| Rang | Tag | Count | Fréquence | Couverture Dataset |
|------|-----|-------|-----------|-------------------|
| 1 | `math` | 1,409 | 28.3% | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 |
| 2 | `graphs` | 542 | 10.9% | 🟦🟦🟦🟦🟦 |
| 3 | `strings` | 422 | 8.5% | 🟦🟦🟦🟦 |
| 4 | `number theory` | 350 | 7.0% | 🟦🟦🟦 |
| 5 | `trees` | 324 | 6.5% | 🟦🟦🟦 |
| 6 | `geometry` | 166 | 3.3% | 🟦🟦 |
| 7 | `games` | 105 | 2.1% | 🟦 |
| 8 | `probabilities` | 92 | 1.8% | 🟦 |

**Couverture totale** : 2,678 / 4,982 (53.8%) des problèmes ont au moins 1 tag prioritaire.

**⚠️ DÉSÉQUILIBRE CRITIQUE** :
- `math` est **15.3× plus fréquent** que `probabilities` (1409 vs 92)
- Ratio le plus élevé : `math` / `probabilities` = **15.3:1**
- ➡️ **Risque** : Le modèle va sur-prédire `math` et sous-prédire `probabilities`

**🔧 STRATÉGIES POUR GÉRER LE DÉSÉQUILIBRE** :
1. **Class weighting** dans la loss function
2. **Stratified sampling** obligatoire
3. **SMOTE** ou sur-échantillonnage pour `probabilities`, `games`, `geometry`
4. **Focal Loss** pour pénaliser les erreurs sur classes rares
5. **Métriques** : Privilégier F1-macro (vs accuracy) pour valoriser les classes rares

---

## 4. Distribution et Structure Multi-Label

### 📊 4.1 Statistiques Globales sur TOUS les Tags

| Métrique | Valeur |
|----------|--------|
| Nombre total de tags uniques | 37 |
| Tags prioritaires | 8 (21.6%) |
| Tags non-prioritaires | 29 (78.4%) |

**Distribution du nombre de tags par problème** :

| Statistique | Valeur |
|-------------|--------|
| Minimum | 0 tags |
| Maximum | 11 tags |
| **Moyenne** | **2.80 tags** |
| **Médiane** | **3 tags** |
| Écart-type | 1.43 |
| Q1 (25%) | 2 tags |
| Q3 (75%) | 4 tags |

**💡 INSIGHT** : En moyenne, chaque problème a **~3 tags**. C'est clairement un **problème multi-label**.

---

### 📊 4.2 Focus sur les Tags Prioritaires

| Métrique | Avant Filtrage | Après Filtrage (≥1 tag prioritaire) |
|----------|----------------|-------------------------------------|
| Nombre de problèmes | 4,982 | 2,678 (53.8%) |
| Nb moyen tags (tous) | 2.80 | 3.27 |
| Nb moyen tags prioritaires | 0.68 | 1.27 |
| Nb médian tags (tous) | 3.0 | 3.0 |
| Nb médian tags prioritaires | 1.0 | 1.0 |

**⚠️ DÉCISION STRATÉGIQUE** :
- **46.2% des problèmes n'ont AUCUN tag prioritaire**
- Faut-il :
  - **Option A** : Se concentrer uniquement sur les 2,678 problèmes avec tags prioritaires (filtrer le reste)
  - **Option B** : Prédire aussi les tags non-prioritaires
  - **Option C** : Créer une classe "other" pour les problèmes sans tags prioritaires

➡️ **RECOMMANDATION** : **Option A** (focus sur les 2,678 avec tags prioritaires) pour un MVP, puis étendre à Option B.

---

### 🎯 4.3 Analyse Mono-Tag vs Multi-Tags (CRITIQUE)

**Sur les 2,678 problèmes avec au moins 1 tag prioritaire** :

| Catégorie | Nombre | Pourcentage | Visualisation |
|-----------|--------|-------------|---------------|
| **Mono-tag** (1 tag) | **2,018** | **75.4%** | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 |
| **Multi-tags** (2 tags) | **597** | **22.3%** | 🟧🟧🟧🟧 |
| **Multi-tags** (3 tags) | **57** | **2.1%** | 🟥 |
| **Multi-tags** (4+ tags) | **6** | **0.2%** | ⬜ |

**💡 INSIGHT MAJEUR** : **75.4% des problèmes n'ont qu'UN SEUL tag prioritaire**.

**🎯 IMPLICATION POUR LA MODÉLISATION** :
- **Stratégie recommandée** : **Approche hybride**
  - **Baseline** : Multi-class classification (75% des cas)
  - **Avancé** : Multi-label classification (25% des cas)
- **Alternative** : One-vs-Rest avec 8 classifieurs binaires
- **Métriques** : 
  - F1-score par classe
  - Hamming Loss
  - Exact Match Ratio (% de problèmes avec tous les tags corrects)

---

## 5. Co-occurrence et Associations de Tags

### 📊 5.1 Matrice de Co-occurrence (Top 10 Paires par Lift)

**Lift** : Mesure l'association entre deux tags. **Lift > 1** signifie que les tags apparaissent ensemble **plus souvent que par hasard**.

**Formule** : Lift(A,B) = P(A,B) / (P(A) × P(B))

| Rang | Tag A | Tag B | Lift | PMI | Count | Interprétation |
|------|-------|-------|------|-----|-------|----------------|
| 1 | `graphs` | `trees` | **3.52** | 1.26 | 123 | ⭐⭐⭐ **Très forte association** |
| 2 | `math` | `number theory` | **2.89** | 1.06 | 284 | ⭐⭐⭐ **Forte association** |
| 3 | `math` | `games` | **2.14** | 0.76 | 67 | ⭐⭐ **Association modérée** |
| 4 | `graphs` | `geometry` | **1.98** | 0.68 | 36 | ⭐⭐ **Association modérée** |
| 5 | `strings` | `games` | **1.87** | 0.63 | 17 | ⭐⭐ **Association modérée** |
| 6 | `math` | `geometry` | **1.76** | 0.57 | 83 | ⭐ **Légère association** |
| 7 | `math` | `probabilities` | **1.65** | 0.50 | 43 | ⭐ **Légère association** |
| 8 | `number theory` | `probabilities` | **1.42** | 0.35 | 9 | **Indépendants** |
| 9 | `graphs` | `strings` | **0.97** | -0.03 | 44 | **Indépendants** |
| 10 | `strings` | `math` | **0.89** | -0.12 | 106 | **Légère répulsion** |

**💡 INSIGHTS CLÉS** :

1. **`graphs` + `trees`** : Lift = 3.52 → Les problèmes de graphes sont **3.5× plus susceptibles** d'être aussi des problèmes d'arbres. C'est logique car **les arbres sont un cas particulier de graphes**.

2. **`math` + `number theory`** : Lift = 2.89 → Forte association. Les problèmes de théorie des nombres sont presque toujours aussi taggés `math`.

3. **`math` + `games`** : Lift = 2.14 → Les problèmes de jeux ont souvent une composante mathématique (théorie des jeux, probabilités, etc.).

4. **`strings` + `math`** : Lift = 0.89 < 1 → **Légère répulsion**. Les problèmes de strings sont généralement purement algorithmiques, sans forte composante mathématique.

**🔧 UTILITÉ POUR LA MODÉLISATION** :
- **Features de co-occurrence** : Créer des features comme `has_graphs_AND_trees`, `has_math_AND_number_theory`
- **Contraintes logiques** : Post-processing pour forcer cohérence (si `trees` prédit avec haute confiance, augmenter la probabilité de `graphs`)
- **Stratified split intelligent** : Éviter de séparer les paires fortement associées

---

### 📊 5.2 Combinaisons de Tags Rares (Risque de Leakage)

**⚠️ PROBLÈME** : Certaines combinaisons de tags n'apparaissent que **1-2 fois** dans le dataset.

**Exemple** : [`graphs`, `geometry`, `probabilities`] n'apparaît qu'une seule fois.

**🚨 RISQUE** :
- Si cette combinaison unique est dans **test**, le modèle ne l'aura **jamais vue** en train → Performance nulle
- Si cette combinaison unique est dans **train**, le modèle risque d'**overfitter** sur ce seul exemple

**🔧 SOLUTION** :
- Utiliser **`iterstrat.ml_stratifiers.MultilabelStratifiedKFold`** pour garantir que chaque combinaison est représentée dans train ET val/test
- Alternative : Regrouper les combinaisons rares sous une catégorie "other"

---

## 6. Analyse LaTeX et Symboles Mathématiques

### 📊 6.1 Présence de LaTeX dans les Descriptions

| Métrique | Résultat |
|----------|----------|
| Échantillons avec LaTeX | 2,782 / 4,982 (**55.8%**) |
| Moyenne de blocs LaTeX par description | 7.94 |
| Moyenne de symboles LaTeX par description | 0.11 |
| Densité LaTeX moyenne | 0.0991 (9.9% du texte) |

**💡 INSIGHT** : Plus de la moitié des problèmes contiennent du LaTeX, ce qui est attendu pour des problèmes algorithmiques avec des formules mathématiques.

---

### 📊 6.2 Densité LaTeX par Tag Prioritaire

| Tag | Densité Blocs LaTeX | Densité Symboles LaTeX | Niveau Mathématique |
|-----|---------------------|------------------------|---------------------|
| `number theory` | **0.1568** | **0.0016** | 🟦🟦🟦🟦🟦 **Très élevé** |
| `math` | **0.1395** | **0.0011** | 🟦🟦🟦🟦 **Élevé** |
| `probabilities` | 0.0892 | 0.0010 | 🟦🟦🟦 **Modéré** |
| `trees` | 0.0898 | 0.0003 | 🟦🟦🟦 **Modéré** |
| `strings` | 0.0860 | 0.0002 | 🟦🟦 **Faible** |
| `graphs` | 0.0835 | 0.0002 | 🟦🟦 **Faible** |
| `geometry` | 0.0718 | 0.0003 | 🟦🟦 **Faible** |
| `games` | **0.0645** | 0.0005 | 🟦 **Très faible** |

**💡 INSIGHTS** :
1. **`number theory`** et **`math`** ont la **densité LaTeX la plus élevée** → Formules mathématiques complexes
2. **`games`** a la **densité LaTeX la plus faible** → Descriptions plus narratives
3. ➡️ **Feature prédictive** : La densité LaTeX peut aider à discriminer `number theory` vs `games`

---

### 📊 6.3 Top 20 Symboles LaTeX les Plus Fréquents

| Rang | Symbole | Documents | Occurrences Totales | % Documents | Description |
|------|---------|-----------|---------------------|-------------|-------------|
| 1 | `\le` | 914 | 2,865 | 18.3% | Inférieur ou égal (≤) |
| 2 | `\ldots` | 455 | 947 | 9.1% | Points de suspension (...) |
| 3 | `\dot` | 419 | 848 | 8.4% | Point au-dessus (ȧ) |
| 4 | `\cdot` | 241 | 443 | 4.8% | Multiplication (·) |
| 5 | `\frac` | 178 | 300 | 3.6% | Fraction |
| 6 | `\times` | 178 | 270 | 3.6% | Multiplication (×) |
| 7 | `\,` | 78 | 253 | 1.6% | Espace fine |
| 8 | `\ne` | 177 | 234 | 3.6% | Différent de (≠) |
| 9 | `\oplus` | 65 | 220 | 1.3% | XOR (⊕) |
| 10 | `\rightarrow` | 25 | 176 | 0.5% | Flèche droite (→) |
| 11 | `\ge` | 138 | 172 | 2.8% | Supérieur ou égal (≥) |
| 12 | `\bmod` | 122 | 158 | 2.4% | Modulo |
| 13 | `\gcd` | 89 | 143 | 1.8% | PGCD |
| 14 | `\sum` | 86 | 139 | 1.7% | Somme (Σ) |
| 15 | `\sqrt` | 62 | 113 | 1.2% | Racine carrée (√) |
| 16 | `\in` | 79 | 109 | 1.6% | Appartient à (∈) |
| 17 | `\lfloor` | 57 | 107 | 1.1% | Partie entière inf (⌊) |
| 18 | `\rfloor` | 57 | 107 | 1.1% | Partie entière inf (⌋) |
| 19 | `\lceil` | 51 | 99 | 1.0% | Partie entière sup (⌈) |
| 20 | `\rceil` | 51 | 99 | 1.0% | Partie entière sup (⌉) |

**💡 INSIGHTS** :
- **Comparaisons** (`\le`, `\ge`, `\ne`) sont très fréquentes → Problèmes d'optimisation, inégalités
- **Modulo** (`\bmod`) et **PGCD** (`\gcd`) → Signature de `number theory`
- **XOR** (`\oplus`) → Problèmes de bits, cryptographie
- **Somme** (`\sum`) → Problèmes de séries, combinatoire

---

### 📊 6.4 Symboles LaTeX Caractéristiques par Tag (Lift Analysis)

**Analyse d'enrichissement** : Quels symboles sont **surreprésentés** dans chaque tag par rapport à la moyenne ?

#### **Tag : `number theory`**

| Symbole | Lift | Interprétation |
|---------|------|----------------|
| `\gcd` | **5.42** | ⭐⭐⭐ **Signature forte** |
| `\bmod` | **4.87** | ⭐⭐⭐ **Signature forte** |
| `\lcm` | **3.91** | ⭐⭐⭐ **Signature forte** |
| `\phi` | **3.24** | ⭐⭐ **Indicatrice d'Euler** |
| `\equiv` | **2.76** | ⭐⭐ **Congruences** |

**💡 INSIGHT** : Les symboles `\gcd`, `\bmod`, `\lcm` sont **fortement caractéristiques** de `number theory`. Un modèle peut utiliser ces features pour détecter ce tag.

#### **Tag : `geometry`**

| Symbole | Lift | Interprétation |
|---------|------|----------------|
| `\angle` | **8.12** | ⭐⭐⭐ **Signature très forte** |
| `\perp` | **6.43** | ⭐⭐⭐ **Perpendiculaire** |
| `\parallel` | **5.21** | ⭐⭐⭐ **Parallèle** |
| `\sqrt` | **2.89** | ⭐⭐ **Distances** |
| `\pi` | **2.34** | ⭐⭐ **Cercles, aires** |

#### **Tag : `probabilities`**

| Symbole | Lift | Interprétation |
|---------|------|----------------|
| `\frac` | **4.21** | ⭐⭐⭐ **Fractions (probabilités)** |
| `\sum` | **3.87** | ⭐⭐⭐ **Espérances** |
| `\binom` | **3.54** | ⭐⭐⭐ **Coefficients binomiaux** |
| `\Pr` | **3.12** | ⭐⭐⭐ **Notation probabilité** |

**🔧 FEATURES À CRÉER** :
- `latex_gcd_count` → Prédictif pour `number theory`
- `latex_angle_count` → Prédictif pour `geometry`
- `latex_binom_count` → Prédictif pour `probabilities`
- `latex_enrichment_score_per_tag` → Score de lift agrégé par tag

---

## 7. Analyse du Code Source

### 📊 7.1 Patterns Algorithmiques Détectés

**Méthode** : Détection de patterns dans le code source par regex et heuristiques.

| Pattern | Description | Fréquence Globale |
|---------|-------------|-------------------|
| `has_sort` | `.sort()` ou `sorted()` | 63.2% |
| `has_deque` | `from collections import deque` | 12.7% |
| `has_queue` | `Queue` ou `deque` | 11.4% |
| `has_adjacency` | `adj`, `graph[`, `edges` | 9.8% |
| `has_dp` | `dp[`, `memo` | 8.3% |
| `has_bisect` | Module `bisect` | 7.1% |
| `has_dsu` | Union-Find (`parent`, `find`, `union`) | 4.2% |
| `has_recursion` | `setrecursionlimit` | 2.8% |

**💡 INSIGHT** : **63% des codes** utilisent le tri → Pattern algorithmique très commun.

---

### 📊 7.2 Patterns Algorithmiques par Tag Prioritaire

| Tag | Top 3 Patterns (>10% d'utilisation) |
|-----|--------------------------------------|
| `graphs` | `adjacency` (23%), `deque` (17%), `queue` (17%) |
| `trees` | `adjacency` (19%), `sort` (19%), `queue` (18%) |
| `geometry` | `sort` (26%) |
| `probabilities` | `sort` (26%), `dp` (11%) |
| `math` | `sort` (16%) |
| `strings` | `sort` (20%) |
| `number theory` | `sort` (15%), `bisect` (14%), `queue` (11%) |
| `games` | `sort` (14%) |

**💡 INSIGHTS** :
1. **`graphs`** et **`trees`** sont caractérisés par **`adjacency`** (listes d'adjacence) et **`deque`** (BFS)
2. **`geometry`** et **`probabilities`** utilisent beaucoup le **`sort`** (tri de points, événements)
3. **`number theory`** utilise **`bisect`** (recherche binaire sur des valeurs triées)
4. **`probabilities`** utilise **`dp`** (programmation dynamique pour espérances)

**🔧 FEATURES À CRÉER** :
- `has_adjacency` → Prédictif pour `graphs`, `trees`
- `has_deque` → Prédictif pour `graphs`
- `has_dp` → Prédictif pour `probabilities`, `games`
- `has_bisect` → Prédictif pour `number theory`

---

### 📊 7.3 Analyse des Imports Python

**Top 10 imports les plus fréquents** :

| Import | Fréquence | Tags Associés |
|--------|-----------|---------------|
| `collections` | 18.3% | `graphs`, `trees`, `strings` |
| `heapq` | 8.7% | `graphs` (Dijkstra), `trees` |
| `bisect` | 7.1% | `number theory`, `math` |
| `math` | 5.4% | `math`, `geometry`, `number theory` |
| `itertools` | 4.9% | `math`, `strings` (permutations) |
| `functools` | 2.1% | `probabilities` (memoization) |
| `operator` | 1.8% | Générique |
| `sys` | 12.6% | Générique (recursion limit) |

**💡 INSIGHT** :
- `collections` (Counter, defaultdict, deque) → Tag `graphs` fort
- `heapq` → Tag `graphs` (Dijkstra, Prim)
- `bisect` → Tag `number theory` fort

---

## 8. Keyword Coverage (Recall Lexical)

### 🎯 8.1 Objectif

Mesurer **si un modèle basé sur des mots-clés simples peut détecter chaque tag**. En d'autres termes : est-ce que les descriptions et le code contiennent des mots caractéristiques qui "trahissent" la présence d'un tag ?

### 📊 8.2 Dictionnaires de Mots-Clés Définis

| Tag | Mots-clés |
|-----|-----------|
| `math` | number, sum, product, divide, multiply, calculate, formula, equation |
| `graphs` | graph, node, edge, vertex, path, connected, component, cycle |
| `strings` | string, substring, character, prefix, suffix, palindrome, pattern |
| `number theory` | prime, divisor, gcd, lcm, modulo, factor, coprime, remainder |
| `trees` | tree, root, parent, child, leaf, ancestor, descendant, subtree |
| `geometry` | point, line, angle, distance, coordinate, polygon, circle, area |
| `games` | game, player, win, lose, strategy, move, turn, optimal |
| `probabilities` | probability, expected, random, distribution, chance, likelihood |

---

### 📊 8.3 Résultats de la Couverture Lexicale

| Tag | Description | Code | **Either (Desc OU Code)** | Difficulté de Détection |
|-----|-------------|------|---------------------------|-------------------------|
| `games` | **96.2%** | 41.0% | **98.1%** | ✅ **Très facile** |
| `geometry` | **90.4%** | 45.2% | **96.4%** | ✅ **Très facile** |
| `probabilities` | 69.6% | 21.7% | 82.6% | ⚠️ **Moyen** |
| `math` | 81.0% | 19.8% | 83.5% | ⚠️ **Moyen** |
| `strings` | 80.1% | 14.9% | 81.3% | ⚠️ **Moyen** |
| `trees` | 64.8% | 43.2% | 79.3% | ⚠️ **Moyen** |
| `graphs` | 57.7% | 38.6% | 70.3% | 🚨 **Difficile** |
| `number theory` | **53.4%** | 41.7% | **69.1%** | 🚨 **Difficile** |

---

### 💡 8.4 Insights Clés

#### ✅ **Tags Faciles à Détecter (Coverage > 90%)**

**`games` (98.1%)** :
- Les mots "game", "player", "win" apparaissent presque systématiquement
- Un modèle basé sur TF-IDF devrait avoir **F1 > 95%**
- Stratégie : **Simple Logistic Regression** suffit

**`geometry` (96.4%)** :
- Les mots "point", "line", "distance" sont très présents
- Descriptions très explicites
- Stratégie : **TF-IDF + Logistic Regression**

---

#### ⚠️ **Tags Moyennement Difficiles (Coverage 80-85%)**

**`math` (83.5%)**, **`strings` (81.3%)**, **`probabilities` (82.6%)** :
- Bonne couverture mais pas parfaite
- Nécessite des features additionnelles :
  - **LaTeX** pour `math` et `probabilities`
  - **Patterns code** pour `strings` (manipulations de chaînes)
- Stratégie : **TF-IDF + Features LaTeX + XGBoost**

---

#### 🚨 **Tags Difficiles (Coverage < 75%)**

**`number theory` (69.1%)** :
- Seulement **53.4% des descriptions** contiennent "prime", "gcd", "divisor"
- **Pourquoi ?** Les problèmes peuvent être implicites ("find count of coprime pairs" sans mentionner "coprime")
- **Solution** : **Features du code source** essentielles :
  - `has_gcd` dans le code
  - `has_bmod` (modulo)
  - Symboles LaTeX `\gcd`, `\bmod`
- Stratégie : **Modèle hybride** BERT (description) + XGBoost (features code)

**`graphs` (70.3%)** :
- **57.7% des descriptions** ne mentionnent pas explicitement "graph"
- **Exemple** : "You are given n cities and m roads..." → Pas le mot "graph"
- **Solution** : **Features code** (adjacency, deque, BFS/DFS)
- Stratégie : **Modèle hybride**

**`trees` (79.3%)** :
- Situation intermédiaire
- Bénéficie aussi des features code (adjacency, recursion)

---

### 🔧 8.5 Implications pour la Modélisation

| Tag | Stratégie Recommandée | Features Clés | F1-Score Attendu |
|-----|----------------------|---------------|------------------|
| `games`, `geometry` | TF-IDF + LogReg | Description seule | **> 90%** |
| `math`, `strings`, `probabilities` | TF-IDF + XGBoost | Description + LaTeX | **85-90%** |
| `number theory`, `graphs`, `trees` | Hybrid BERT + XGBoost | Description + Code + LaTeX | **75-85%** |

---

## 9. Outliers et Hard Cases

### 📊 9.1 Détection d'Outliers par Tag

**Méthode** : Identifier les problèmes "atypiques" qui seront difficiles à classifier.

| Tag | Très courts (5%) | Sans keywords | Total | % Hard Cases |
|-----|------------------|---------------|-------|--------------|
| `math` | 70 | 267 | 1,409 | **23.9%** |
| `graphs` | 27 | 229 | 542 | **47.2%** |
| `strings` | 21 | 84 | 422 | **24.9%** |
| `number theory` | 17 | 163 | 350 | **51.4%** |
| `trees` | 16 | 114 | 324 | **40.1%** |
| `geometry` | 8 | 16 | 166 | **14.5%** |
| `games` | 5 | 4 | 105 | **8.6%** |
| `probabilities` | 5 | 28 | 92 | **35.9%** |

**💡 INSIGHTS** :
1. **`number theory`** a le **plus de hard cases** (51.4%) → 163 problèmes sans keywords
2. **`graphs`** : 47.2% de hard cases → Descriptions implicites
3. **`games`** et **`geometry`** : Moins de 15% de hard cases → Descriptions explicites

**🔧 ACTION** :
- Pour `number theory` et `graphs` : **Nécessité absolue d'embeddings sémantiques (BERT)**
- Les approches lexicales (TF-IDF) vont échouer sur ces 30-50% de hard cases
- Envisager un **modèle à deux étages** :
  1. **Règles simples** (keywords) pour les cas faciles (70%)
  2. **BERT** pour les hard cases (30%)

---

## 10. Insights Clés et Recommandations

### 🎯 10.1 Insights Principaux

#### **1. Structure Multi-Label avec Dominance Mono-Tag**
- ✅ **75.4% des problèmes** n'ont qu'**UN SEUL tag prioritaire**
- ⚠️ Mais **24.6%** ont **plusieurs tags** → Vrai problème multi-label
- 💡 Approche hybride recommandée

#### **2. Déséquilibre Extrême des Classes**
- ⚠️ `math` (1409) vs `probabilities` (92) → Ratio **15:1**
- 🔧 Class weighting + SMOTE obligatoires
- 📊 Métriques : F1-macro > accuracy

#### **3. Near-Duplicates Présents**
- ⚠️ **11 groupes** de descriptions similaires (23 échantillons)
- 🔧 Stratified split multi-label obligatoire
- 📦 Utiliser `iterstrat.ml_stratifiers.MultilabelStratifiedKFold`

#### **4. Features LaTeX Très Discriminantes**
- ✅ `number theory` : Densité LaTeX **2.4× supérieure** à `games`
- ✅ Symboles spécifiques : `\gcd` → `number theory`, `\angle` → `geometry`
- 🔧 Créer features de lift par symbole

#### **5. Code Source Essentiel pour 3 Tags**
- 🚨 `number theory` (69%), `graphs` (70%), `trees` (79%) → **<80% coverage keywords**
- ✅ Features code **indispensables** :
  - `has_adjacency` → `graphs`, `trees`
  - `has_bisect` → `number theory`
  - `has_dp` → `probabilities`

#### **6. Associations Fortes entre Tags**
- ⭐ `graphs` + `trees` : Lift = **3.52**
- ⭐ `math` + `number theory` : Lift = **2.89**
- 🔧 Exploiter ces co-occurrences en features

---

### 🚀 10.2 Roadmap de Modélisation

#### **Phase 1 : Baseline Simple (Semaine 1)**

**Modèle** : XGBoost One-vs-Rest (8 classifieurs binaires)

**Features** :
1. **TF-IDF** sur `prob_desc_description` (500 dimensions)
2. **Densité LaTeX** : `latex_density`, `nb_latex_blocks`
3. **Symboles LaTeX** : `latex_gcd_count`, `latex_angle_count`, etc. (20 features)
4. **Patterns code** : `has_adjacency`, `has_deque`, `has_sort`, etc. (10 features)
5. **Longueur** : `desc_length_chars`, `code_length_chars`
6. **Difficulté** : `difficulty`
7. **Missingness** : `notes_is_missing`

**Target** : F1-macro **> 70%**

**Avantages** :
- Rapide à implémenter
- Interprétable
- Permet d'établir un **benchmark solide**

---

#### **Phase 2 : Modèle Hybride (Semaine 2-3)**

**Architecture** : BERT (description) + XGBoost (code features)

**Branche 1 : BERT Fine-tuné**
- Input : `prob_desc_description`
- Output : Embeddings 768-dim
- Fine-tuning sur classification multi-label

**Branche 2 : XGBoost**
- Input : Features engineered (LaTeX, code patterns, etc.)
- Output : Scores par tag

**Fusion** : Moyenne pondérée ou Meta-learner

**Target** : F1-macro **> 80%**

**Avantages** :
- Capture la sémantique profonde (BERT)
- Exploite les patterns algorithmiques (XGBoost)

---

#### **Phase 3 : Modèle Avancé (Semaine 4)**

**Architecture** : CodeBERT ou GraphCodeBERT

**Input** : Description + Code source (tokenisé ensemble)

**Avantages** :
- Pré-entraîné sur du code source
- Comprend la syntaxe et sémantique du code
- Capture les relations entre description et implémentation

**Target** : F1-macro **> 85%**

---

### 📊 10.3 Métriques d'Évaluation

| Métrique | Utilité | Seuil de Succès |
|----------|---------|-----------------|
| **F1-macro** | Moyenne des F1 par classe (valorise classes rares) | **> 80%** |
| **F1-micro** | F1 global (biaisé par classes fréquentes) | Informatif |
| **Hamming Loss** | Proportion d'erreurs de prédiction | **< 0.15** |
| **Exact Match Ratio** | % de problèmes avec TOUS les tags corrects | **> 60%** |
| **F1 par classe** | Performance individuelle par tag | Toutes > 70% |
| **Precision/Recall par classe** | Trade-off par tag | Équilibrés |

**⚠️ À ÉVITER** : Accuracy globale (biaisée par `math`)

---

### 🔧 10.4 Features à Créer (Top 30 Prioritaires)

#### **A. Features Textuelles (Description)**

1. `desc_length_chars` - Longueur en caractères
2. `desc_length_words` - Longueur en mots
3. `desc_length_sentences` - Nombre de phrases
4. `desc_avg_word_length` - Longueur moyenne des mots
5. `desc_has_non_ascii` - Présence de caractères non-ASCII
6. `desc_ascii_ratio` - Ratio de caractères ASCII
7. TF-IDF vecteurs (500 dimensions)

#### **B. Features LaTeX**

8. `latex_density` - Densité de LaTeX (% du texte)
9. `nb_latex_blocks` - Nombre de blocs LaTeX
10. `nb_latex_symbols` - Nombre de symboles LaTeX
11. `latex_gcd_count` - Nombre de `\gcd`
12. `latex_bmod_count` - Nombre de `\bmod`
13. `latex_angle_count` - Nombre de `\angle`
14. `latex_frac_count` - Nombre de `\frac`
15. `latex_sum_count` - Nombre de `\sum`
16. `latex_binom_count` - Nombre de `\binom`
17. `latex_le_count` - Nombre de `\le` (inégalités)
18. `latex_enrichment_score_[tag]` - Score de lift agrégé (8 features)

#### **C. Features Code Source**

26. `code_length_chars` - Longueur en caractères
27. `code_length_lines` - Longueur en lignes
28. `has_adjacency` - Présence de liste d'adjacence
29. `has_deque` - Utilise deque
30. `has_heapq` - Utilise heapq
31. `has_bisect` - Utilise bisect
32. `has_dp` - Programmation dynamique
33. `has_sort` - Utilise tri
34. `has_recursion` - Limite de récursion
35. `nb_loops` - Nombre de boucles (for, while)
36. `nb_conditions` - Nombre de conditions (if)
37. `nb_functions` - Nombre de fonctions définies
38. `has_import_collections` - Import collections
39. `has_import_heapq` - Import heapq
40. `has_import_math` - Import math
41. `has_import_bisect` - Import bisect

#### **D. Features Méta**

42. `difficulty` - Score de difficulté
43. `notes_is_missing` - Notes manquantes (binaire)
44. `nb_tags_total` - Nombre total de tags
45. `nb_priority_tags` - Nombre de tags prioritaires

#### **E. Features de Co-occurrence**

46. `has_graphs_and_trees` - Présence simultanée
47. `has_math_and_number_theory` - Présence simultanée
48. `tag_cooccurrence_score` - Score agrégé

---

### ✅ 10.5 Checklist Avant Modélisation

- [ ] **Stratified Split Multi-Label** avec `iterstrat`
- [ ] **Vérifier les near-duplicates** (11 groupes)
- [ ] **Class weighting** pour gérer déséquilibre
- [ ] **Créer features LaTeX** (densité, symboles, lift)
- [ ] **Créer features Code** (patterns, imports, complexité)
- [ ] **TF-IDF** sur description (500-1000 dim)
- [ ] **Valider sur validation set** avec F1-macro
- [ ] **Analyser les erreurs** par tag (Precision/Recall)
- [ ] **Post-processing** : Exploiter co-occurrences (si `trees` → augmenter prob `graphs`)
- [ ] **Calibration** : Ajuster les seuils de décision par tag

---

## 📚 Conclusion

Ce dataset présente des **défis intéressants** pour la classification multi-label :

### ✅ **Points Forts du Dataset**
- Qualité des données (pas de doublons stricts)
- Tags bien normalisés
- Features riches (description + code + LaTeX)
- Taille raisonnable (4,982 échantillons)

### ⚠️ **Défis Majeurs**
1. **Déséquilibre extrême** (15:1 ratio)
2. **Near-duplicates** à gérer (23 échantillons)
3. **Hard cases** fréquents pour `number theory` (51%) et `graphs` (47%)
4. **Dominance mono-tag** (75%) mais 25% multi-tags
5. **Nécessité de features code** pour 3 tags

### 🚀 **Stratégie Gagnante**
1. **Baseline XGBoost** avec features engineered → F1 ~70%
2. **Modèle hybride BERT + XGBoost** → F1 ~80%
3. **CodeBERT** pour aller au-delà de 85%
4. **Post-processing** avec co-occurrences pour gain final de 2-3%

### 🎯 **Objectif Réaliste**
**F1-macro > 80%** est un objectif **ambitieux mais atteignable** avec une approche hybride et un bon feature engineering.

---

**📊 Rapport généré à partir de l'EDA complète du notebook `02_eda_complete_validated.ipynb`**

