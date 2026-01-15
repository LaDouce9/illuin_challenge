# Analyse de l'Overfitting - Modèles d'Arbre avec sample_weight

## Problème identifié

Les modèles d'arbre (XGBoost et RandomForest) présentent un **fort overfitting** :

- **XGBoost (Embeddings + Denses)** : Train Macro-F1: **0.9940**, Test Macro-F1: **0.4931** → Gap de **0.50**
- **RandomForest (Toutes)** : Train Macro-F1: **0.9675**, Test Macro-F1: **0.3817** → Gap de **0.59**

Alors que **LogisticRegression** a un overfitting modéré :
- **LogReg (Toutes)** : Train Macro-F1: **0.7403**, Test Macro-F1: **0.5442** → Gap de **0.20**

## Cause du problème : Double Entraînement

### Code actuel (PROBLÉMATIQUE)

```python
# Étape 1 : Premier fit() pour créer les estimateurs
model_xgb_emb_dense.fit(X_train_full, y_train)

# Étape 2 : Refit() de chaque estimateur avec sample_weight
for i, label_col in enumerate(target_columns):
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    model_xgb_emb_dense.estimators_[i].fit(
        X_train_full, 
        y_train[:, i],
        sample_weight=sample_weights
    )
```

### Pourquoi c'est problématique ?

1. **Premier `fit()`** : Entraîne complètement le modèle **sans** `sample_weight`
   - Le modèle apprend déjà les patterns du train set
   - Pour XGBoost/RandomForest, cela construit déjà les arbres

2. **Deuxième `fit()`** : **Réinitialise et réentraîne** le modèle avec `sample_weight`
   - Pour XGBoost et RandomForest, `fit()` sur un modèle déjà entraîné **réinitialise complètement** le modèle
   - Le modèle est donc entraîné **deux fois** sur les mêmes données
   - Cela peut causer un **mémorisation excessive** des données d'entraînement

3. **Résultat** : Le modèle "voit" les données d'entraînement deux fois, ce qui favorise l'overfitting

## Comment fonctionne `sample_weight` ?

### Principe

`sample_weight` permet de donner **plus d'importance** à certains échantillons lors de l'entraînement :

- **Échantillons de classe minoritaire** → Poids élevé
- **Échantillons de classe majoritaire** → Poids faible

### Calcul avec `compute_sample_weight('balanced')`

```python
from sklearn.utils.class_weight import compute_sample_weight

# Pour un label binaire avec déséquilibre
y = [0, 0, 0, 0, 0, 1, 1]  # 5 négatifs, 2 positifs

weights = compute_sample_weight('balanced', y)
# Résultat approximatif :
# - Échantillons de classe 0 (majoritaire) : poids ~0.7
# - Échantillons de classe 1 (minoritaire) : poids ~1.75
```

**Formule** : `n_samples / (n_classes * np.bincount(y))`

### Impact sur l'entraînement

Pour **XGBoost** :
- Les échantillons avec poids élevé contribuent **plus** au calcul du gradient
- Les splits dans les arbres favorisent les échantillons pondérés
- Les feuilles finales sont influencées par les poids

Pour **RandomForest** :
- Les échantillons avec poids élevé ont **plus de chances** d'être sélectionnés dans les bootstraps
- Les splits favorisent les échantillons pondérés
- Les votes finaux sont pondérés

## Solution : Entraîner directement avec sample_weight

### Approche correcte

**Ne PAS faire de premier fit()**, mais créer directement les estimateurs avec `sample_weight` :

```python
# Créer le modèle
model_xgb = OneVsRestClassifier(
    xgb.XGBClassifier(...),
    n_jobs=-1
)

# Entraîner directement avec sample_weight (sans premier fit)
for i, label_col in enumerate(target_columns):
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    
    # Créer un nouvel estimateur pour ce label
    estimator = xgb.XGBClassifier(...)
    estimator.fit(
        X_train_full,
        y_train[:, i],
        sample_weight=sample_weights
    )
    
    # Stocker l'estimateur
    model_xgb.estimators_.append(estimator)
    model_xgb.classes_ = np.arange(y_train.shape[1])
```

**OU** utiliser un wrapper personnalisé (comme dans le GridSearch) qui gère `sample_weight` dès le premier fit.

## Comparaison des approches

### Approche actuelle (PROBLÉMATIQUE)
```
Fit 1 (sans sample_weight) → Modèle entraîné
Fit 2 (avec sample_weight) → Modèle réinitialisé et réentraîné
Résultat : Double entraînement → Overfitting
```

### Approche correcte
```
Fit unique (avec sample_weight dès le début)
Résultat : Entraînement unique avec pondération → Moins d'overfitting
```

## Recommandations

1. **Supprimer le premier `fit()`** pour XGBoost et RandomForest
2. **Créer directement les estimateurs** avec `sample_weight` dès le début
3. **Utiliser le wrapper personnalisé** (comme dans GridSearch) pour une approche propre
4. **Ajouter de la régularisation** :
   - XGBoost : `reg_alpha`, `reg_lambda`, `min_child_weight`
   - RandomForest : `max_depth` plus faible, `min_samples_split` plus élevé

## Code corrigé proposé

Voir la section suivante pour le code corrigé à intégrer dans le notebook.


## Problème identifié

Les modèles d'arbre (XGBoost et RandomForest) présentent un **fort overfitting** :

- **XGBoost (Embeddings + Denses)** : Train Macro-F1: **0.9940**, Test Macro-F1: **0.4931** → Gap de **0.50**
- **RandomForest (Toutes)** : Train Macro-F1: **0.9675**, Test Macro-F1: **0.3817** → Gap de **0.59**

Alors que **LogisticRegression** a un overfitting modéré :
- **LogReg (Toutes)** : Train Macro-F1: **0.7403**, Test Macro-F1: **0.5442** → Gap de **0.20**

## Cause du problème : Double Entraînement

### Code actuel (PROBLÉMATIQUE)

```python
# Étape 1 : Premier fit() pour créer les estimateurs
model_xgb_emb_dense.fit(X_train_full, y_train)

# Étape 2 : Refit() de chaque estimateur avec sample_weight
for i, label_col in enumerate(target_columns):
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    model_xgb_emb_dense.estimators_[i].fit(
        X_train_full, 
        y_train[:, i],
        sample_weight=sample_weights
    )
```

### Pourquoi c'est problématique ?

1. **Premier `fit()`** : Entraîne complètement le modèle **sans** `sample_weight`
   - Le modèle apprend déjà les patterns du train set
   - Pour XGBoost/RandomForest, cela construit déjà les arbres

2. **Deuxième `fit()`** : **Réinitialise et réentraîne** le modèle avec `sample_weight`
   - Pour XGBoost et RandomForest, `fit()` sur un modèle déjà entraîné **réinitialise complètement** le modèle
   - Le modèle est donc entraîné **deux fois** sur les mêmes données
   - Cela peut causer un **mémorisation excessive** des données d'entraînement

3. **Résultat** : Le modèle "voit" les données d'entraînement deux fois, ce qui favorise l'overfitting

## Comment fonctionne `sample_weight` ?

### Principe

`sample_weight` permet de donner **plus d'importance** à certains échantillons lors de l'entraînement :

- **Échantillons de classe minoritaire** → Poids élevé
- **Échantillons de classe majoritaire** → Poids faible

### Calcul avec `compute_sample_weight('balanced')`

```python
from sklearn.utils.class_weight import compute_sample_weight

# Pour un label binaire avec déséquilibre
y = [0, 0, 0, 0, 0, 1, 1]  # 5 négatifs, 2 positifs

weights = compute_sample_weight('balanced', y)
# Résultat approximatif :
# - Échantillons de classe 0 (majoritaire) : poids ~0.7
# - Échantillons de classe 1 (minoritaire) : poids ~1.75
```

**Formule** : `n_samples / (n_classes * np.bincount(y))`

### Impact sur l'entraînement

Pour **XGBoost** :
- Les échantillons avec poids élevé contribuent **plus** au calcul du gradient
- Les splits dans les arbres favorisent les échantillons pondérés
- Les feuilles finales sont influencées par les poids

Pour **RandomForest** :
- Les échantillons avec poids élevé ont **plus de chances** d'être sélectionnés dans les bootstraps
- Les splits favorisent les échantillons pondérés
- Les votes finaux sont pondérés

## Solution : Entraîner directement avec sample_weight

### Approche correcte

**Ne PAS faire de premier fit()**, mais créer directement les estimateurs avec `sample_weight` :

```python
# Créer le modèle
model_xgb = OneVsRestClassifier(
    xgb.XGBClassifier(...),
    n_jobs=-1
)

# Entraîner directement avec sample_weight (sans premier fit)
for i, label_col in enumerate(target_columns):
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    
    # Créer un nouvel estimateur pour ce label
    estimator = xgb.XGBClassifier(...)
    estimator.fit(
        X_train_full,
        y_train[:, i],
        sample_weight=sample_weights
    )
    
    # Stocker l'estimateur
    model_xgb.estimators_.append(estimator)
    model_xgb.classes_ = np.arange(y_train.shape[1])
```

**OU** utiliser un wrapper personnalisé (comme dans le GridSearch) qui gère `sample_weight` dès le premier fit.

## Comparaison des approches

### Approche actuelle (PROBLÉMATIQUE)
```
Fit 1 (sans sample_weight) → Modèle entraîné
Fit 2 (avec sample_weight) → Modèle réinitialisé et réentraîné
Résultat : Double entraînement → Overfitting
```

### Approche correcte
```
Fit unique (avec sample_weight dès le début)
Résultat : Entraînement unique avec pondération → Moins d'overfitting
```

## Recommandations

1. **Supprimer le premier `fit()`** pour XGBoost et RandomForest
2. **Créer directement les estimateurs** avec `sample_weight` dès le début
3. **Utiliser le wrapper personnalisé** (comme dans GridSearch) pour une approche propre
4. **Ajouter de la régularisation** :
   - XGBoost : `reg_alpha`, `reg_lambda`, `min_child_weight`
   - RandomForest : `max_depth` plus faible, `min_samples_split` plus élevé

## Code corrigé proposé

Voir la section suivante pour le code corrigé à intégrer dans le notebook.

