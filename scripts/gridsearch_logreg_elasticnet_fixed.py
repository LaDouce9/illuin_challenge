# ============================================================================
# GRIDSEARCH - LOGISTIC REGRESSION (TOUTES LES FEATURES) - AVEC ELASTICNET
# ============================================================================

print("="*80)
print("GRIDSEARCH - LOGISTIC REGRESSION (TOUTES LES FEATURES)")
print("="*80)

# Créer le scorer Macro-F1
scorer = make_scorer(f1_score, average='macro')

# Créer le CV stratifié multi-label (3 folds pour rapidité)
cv = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

# Définir la grille de paramètres (L2 + ElasticNet)
print(f"\n[1/4] Définition de la grille de paramètres...")
param_grid_logreg = [
    {   # L2
        'estimator__penalty': ['l2'],
        'estimator__C': [0.1, 0.5, 1, 2],
        'estimator__class_weight': [None, 'balanced'],
        'estimator__solver': ['saga'],
    },
    {   # ElasticNet
        'estimator__penalty': ['elasticnet'],
        'estimator__l1_ratio': [0.1, 0.3, 0.5],
        'estimator__C': [0.1, 0.5, 1, 2],
        'estimator__class_weight': [None, 'balanced'],
        'estimator__solver': ['saga'],
    },
]

# Afficher les paramètres à tester
print(f"  Paramètres à tester:")
print(f"  --- L2 ---")
for param, values in param_grid_logreg[0].items():
    print(f"    {param}: {values}")
print(f"  --- ElasticNet ---")
for param, values in param_grid_logreg[1].items():
    print(f"    {param}: {values}")

# Calculer le nombre total de combinaisons
total_combinations_l2 = np.prod([len(v) for v in param_grid_logreg[0].values()])
total_combinations_elasticnet = np.prod([len(v) for v in param_grid_logreg[1].values()])
total_combinations = total_combinations_l2 + total_combinations_elasticnet

print(f"\n  Total combinaisons:")
print(f"    L2:        {total_combinations_l2}")
print(f"    ElasticNet: {total_combinations_elasticnet}")
print(f"    TOTAL:     {total_combinations}")
print(f"  CV folds: {cv.n_splits}")
print(f"  Total fits: {total_combinations * cv.n_splits}")

# Créer le modèle de base
base_model_logreg = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        solver='saga',
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    n_jobs=-1
)

# GridSearchCV
print(f"\n[2/4] Création du GridSearchCV...")
grid_search_logreg = GridSearchCV(
    estimator=base_model_logreg,
    param_grid=param_grid_logreg,
    scoring=scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

# Lancer le gridsearch
print(f"\n[3/4] Lancement du GridSearchCV...")
print(f"  ⏱️  Temps estimé: ~20-30 min")
print(f"  📊 Score: Macro-F1")
print(f"  Features: TOUTES ({X_train_logreg.shape[1]:,} features)")
start_time = time.time()

grid_search_logreg.fit(X_train_logreg, y_train)

search_time = time.time() - start_time
print(f"\n  ✅ Terminé en {search_time:.2f}s ({search_time/60:.2f} min)")

# Meilleurs paramètres
print(f"\n[4/4] Résultats...")
print(f"\n🏆 MEILLEURS PARAMÈTRES:")
best_params_logreg = grid_search_logreg.best_params_
for param, value in best_params_logreg.items():
    print(f"  {param}: {value}")

print(f"\n📊 MEILLEURE SCORE (CV Macro-F1): {grid_search_logreg.best_score_:.4f}")

# Évaluer sur test avec le meilleur modèle
best_model_logreg = grid_search_logreg.best_estimator_
y_test_pred_logreg_best = best_model_logreg.predict(X_test_logreg)

macro_f1_test_logreg_best = f1_score(y_test, y_test_pred_logreg_best, average='macro')
micro_f1_test_logreg_best = f1_score(y_test, y_test_pred_logreg_best, average='micro')
hamming_test_logreg_best = hamming_loss(y_test, y_test_pred_logreg_best)

print(f"\n📊 PERFORMANCE SUR TEST (meilleur modèle):")
print(f"  Macro-F1: {macro_f1_test_logreg_best:.4f}")
print(f"  Micro-F1: {micro_f1_test_logreg_best:.4f}")
print(f"  Hamming:  {hamming_test_logreg_best:.4f}")

# Comparaison avec baseline
print(f"\n📊 COMPARAISON AVEC BASELINE:")
print(f"  Macro-F1: {macro_f1_test_logreg:.4f} → {macro_f1_test_logreg_best:.4f} ({macro_f1_test_logreg_best - macro_f1_test_logreg:+.4f})")
print(f"  Micro-F1: {micro_f1_test_logreg:.4f} → {micro_f1_test_logreg_best:.4f} ({micro_f1_test_logreg_best - micro_f1_test_logreg:+.4f})")
print(f"  Hamming:  {hamming_test_logreg:.4f} → {hamming_test_logreg_best:.4f} ({hamming_test_logreg_best - hamming_test_logreg:+.4f})")

# Sauvegarder le meilleur modèle
print(f"\n💾 Sauvegarde du meilleur modèle...")
best_model_path_logreg = MODELS_DIR / 'logreg_all_features_best.pkl'
joblib.dump(best_model_logreg, best_model_path_logreg)
print(f"  ✅ Modèle sauvegardé: {best_model_path_logreg}")

print("\n✅ GridSearch LogisticRegression terminé")
print("="*80)

