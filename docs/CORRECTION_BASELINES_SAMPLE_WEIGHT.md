# Correction des Baselines - Élimination du Double Entraînement

## Code corrigé pour XGBoost Baseline

### Avant (PROBLÉMATIQUE)
```python
# D'abord fitter le modèle pour créer les estimateurs
model_xgb_emb_dense.fit(X_train_emb_dense, y_train)

# Puis refitter chaque estimateur avec sample_weight
for i, label_col in enumerate(target_columns):
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    model_xgb_emb_dense.estimators_[i].fit(
        X_train_emb_dense, 
        y_train[:, i],
        sample_weight=sample_weights
    )
```

### Après (CORRIGÉ)
```python
# Créer directement les estimateurs avec sample_weight (pas de premier fit)
model_xgb_emb_dense.estimators_ = []
model_xgb_emb_dense.classes_ = np.arange(y_train.shape[1])

for i, label_col in enumerate(target_columns):
    label_name = label_col.replace('target_', '')
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    
    # Créer un nouvel estimateur pour ce label
    estimator = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # Entraîner directement avec sample_weight
    estimator.fit(
        X_train_emb_dense,
        y_train[:, i],
        sample_weight=sample_weights
    )
    
    model_xgb_emb_dense.estimators_.append(estimator)
    
    if (i + 1) % 2 == 0 or (i + 1) == len(target_columns):
        print(f"  Label {i+1}/{len(target_columns)}: {label_name}")
```

## Code corrigé pour RandomForest Baseline

### Avant (PROBLÉMATIQUE)
```python
# D'abord fitter le modèle pour créer les estimateurs
model_rf_baseline.fit(X_train_full, y_train)

# Puis refitter chaque estimateur avec sample_weight
for i, label_col in enumerate(target_columns):
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    model_rf_baseline.estimators_[i].fit(
        X_train_full, 
        y_train[:, i],
        sample_weight=sample_weights
    )
```

### Après (CORRIGÉ)
```python
# Créer directement les estimateurs avec sample_weight (pas de premier fit)
model_rf_baseline.estimators_ = []
model_rf_baseline.classes_ = np.arange(y_train.shape[1])

for i, label_col in enumerate(target_columns):
    label_name = label_col.replace('target_', '')
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    
    # Créer un nouvel estimateur pour ce label
    estimator = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    # Entraîner directement avec sample_weight
    estimator.fit(
        X_train_full,
        y_train[:, i],
        sample_weight=sample_weights
    )
    
    model_rf_baseline.estimators_.append(estimator)
    
    if (i + 1) % 2 == 0 or (i + 1) == len(target_columns):
        print(f"  Label {i+1}/{len(target_columns)}: {label_name}")
```

## Avantages de cette approche

1. **Pas de double entraînement** : Chaque estimateur est entraîné une seule fois
2. **sample_weight appliqué dès le début** : Pas de réinitialisation
3. **Moins d'overfitting** : Le modèle ne "voit" les données qu'une fois
4. **Cohérence avec GridSearch** : Même approche que les wrappers personnalisés

## Résultats attendus

Après correction, on devrait observer :
- **Réduction de l'overfitting** : Gap train/test plus faible
- **Meilleure généralisation** : Test Macro-F1 potentiellement amélioré
- **Performance plus réaliste** : Train Macro-F1 plus proche du test

## Recommandations supplémentaires

Pour réduire encore l'overfitting, considérer :

### Pour XGBoost
- Augmenter `reg_alpha` et `reg_lambda` (régularisation L1/L2)
- Augmenter `min_child_weight` (minimum d'échantillons par feuille)
- Réduire `max_depth` (profondeur des arbres)
- Réduire `learning_rate` avec plus d'`n_estimators`

### Pour RandomForest
- Réduire `max_depth` (ex: 8 au lieu de 10)
- Augmenter `min_samples_split` (ex: 10 au lieu de 5)
- Augmenter `min_samples_leaf` (ex: 4 au lieu de 2)
- Réduire `n_estimators` si nécessaire pour la vitesse


## Code corrigé pour XGBoost Baseline

### Avant (PROBLÉMATIQUE)
```python
# D'abord fitter le modèle pour créer les estimateurs
model_xgb_emb_dense.fit(X_train_emb_dense, y_train)

# Puis refitter chaque estimateur avec sample_weight
for i, label_col in enumerate(target_columns):
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    model_xgb_emb_dense.estimators_[i].fit(
        X_train_emb_dense, 
        y_train[:, i],
        sample_weight=sample_weights
    )
```

### Après (CORRIGÉ)
```python
# Créer directement les estimateurs avec sample_weight (pas de premier fit)
model_xgb_emb_dense.estimators_ = []
model_xgb_emb_dense.classes_ = np.arange(y_train.shape[1])

for i, label_col in enumerate(target_columns):
    label_name = label_col.replace('target_', '')
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    
    # Créer un nouvel estimateur pour ce label
    estimator = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # Entraîner directement avec sample_weight
    estimator.fit(
        X_train_emb_dense,
        y_train[:, i],
        sample_weight=sample_weights
    )
    
    model_xgb_emb_dense.estimators_.append(estimator)
    
    if (i + 1) % 2 == 0 or (i + 1) == len(target_columns):
        print(f"  Label {i+1}/{len(target_columns)}: {label_name}")
```

## Code corrigé pour RandomForest Baseline

### Avant (PROBLÉMATIQUE)
```python
# D'abord fitter le modèle pour créer les estimateurs
model_rf_baseline.fit(X_train_full, y_train)

# Puis refitter chaque estimateur avec sample_weight
for i, label_col in enumerate(target_columns):
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    model_rf_baseline.estimators_[i].fit(
        X_train_full, 
        y_train[:, i],
        sample_weight=sample_weights
    )
```

### Après (CORRIGÉ)
```python
# Créer directement les estimateurs avec sample_weight (pas de premier fit)
model_rf_baseline.estimators_ = []
model_rf_baseline.classes_ = np.arange(y_train.shape[1])

for i, label_col in enumerate(target_columns):
    label_name = label_col.replace('target_', '')
    sample_weights = compute_sample_weight('balanced', y_train[:, i])
    
    # Créer un nouvel estimateur pour ce label
    estimator = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    # Entraîner directement avec sample_weight
    estimator.fit(
        X_train_full,
        y_train[:, i],
        sample_weight=sample_weights
    )
    
    model_rf_baseline.estimators_.append(estimator)
    
    if (i + 1) % 2 == 0 or (i + 1) == len(target_columns):
        print(f"  Label {i+1}/{len(target_columns)}: {label_name}")
```

## Avantages de cette approche

1. **Pas de double entraînement** : Chaque estimateur est entraîné une seule fois
2. **sample_weight appliqué dès le début** : Pas de réinitialisation
3. **Moins d'overfitting** : Le modèle ne "voit" les données qu'une fois
4. **Cohérence avec GridSearch** : Même approche que les wrappers personnalisés

## Résultats attendus

Après correction, on devrait observer :
- **Réduction de l'overfitting** : Gap train/test plus faible
- **Meilleure généralisation** : Test Macro-F1 potentiellement amélioré
- **Performance plus réaliste** : Train Macro-F1 plus proche du test

## Recommandations supplémentaires

Pour réduire encore l'overfitting, considérer :

### Pour XGBoost
- Augmenter `reg_alpha` et `reg_lambda` (régularisation L1/L2)
- Augmenter `min_child_weight` (minimum d'échantillons par feuille)
- Réduire `max_depth` (profondeur des arbres)
- Réduire `learning_rate` avec plus d'`n_estimators`

### Pour RandomForest
- Réduire `max_depth` (ex: 8 au lieu de 10)
- Augmenter `min_samples_split` (ex: 10 au lieu de 5)
- Augmenter `min_samples_leaf` (ex: 4 au lieu de 2)
- Réduire `n_estimators` si nécessaire pour la vitesse

