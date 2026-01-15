"""
Script de ré-entrainement du modèle LogisticRegression optimal pour la CLI.

Objectif :
- Repartir des artefacts de préprocessing déjà présents dans data/processed
  (parquet préprocessé, TF-IDF, embeddings, scaler dense).
- Recréer la matrice de features complète X_train_logreg (TF-IDF + embeddings + denses normalisées).
- Entraîner une LogisticRegression multi-label avec les meilleurs hyperparamètres
  trouvés dans le notebook 05_modeling_v1.ipynb.
- Sauvegarder le modèle sous un nouveau nom pour un usage par la CLI, sans
  écraser les modèles sauvegardés dans le notebook.

Ce script NE modifie pas la logique ML utilisée en production, il la
rejoue simplement en script avec les paramètres optimaux figés.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack, save_npz, load_npz
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def main() -> None:
    # Chemins de base (version "production" du notebook)
    data_dir = Path("data/processed")
    models_dir = Path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RE-ENTRAINEMENT LOGREG OPTIMAL POUR LA CLI")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Chargement des données préprocessées
    # ------------------------------------------------------------------
    train_path = data_dir / "train_preprocessed.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Fichier train préprocessé introuvable: {train_path}")

    print("\n[1/6] Chargement du train preprocesse...")
    df_train = pd.read_parquet(train_path)
    print(f"  Train: {len(df_train):,} echantillons, {len(df_train.columns)} colonnes")

    # Colonnes cibles (mêmes règles que dans le notebook)
    target_columns = sorted(
        [col for col in df_train.columns if col.startswith("target_")]
    )
    if not target_columns:
        raise ValueError("Aucune colonne cible 'target_*' trouvée dans df_train.")

    print(f"  Targets: {len(target_columns)} colonnes")

    # ------------------------------------------------------------------
    # 2. Construction des features denses (comme dans le notebook)
    # ------------------------------------------------------------------
    print("\n[2/6] Construction des features denses...")

    # Features LaTeX numériques (4 features)
    features_latex_numeriques = [
        "nb_latex_blocks",
        "nb_latex_symbols",
        "latex_density",
        "latex_symbols_density",
    ]

    # Features LaTeX binaires (toutes les colonnes commençant par 'has_')
    features_latex_binaires = [
        col for col in df_train.columns if col.startswith("has_")
    ]

    # Features LaTeX complètes
    features_latex = features_latex_numeriques + features_latex_binaires

    # Features numériques générales (2 features)
    features_numeriques_base = [
        "difficulty",
        "time_limit_seconds",
    ]

    # Features de longueur de texte (pour prob_desc_description_translated)
    features_longueur_texte = [
        "prob_desc_description_translated_char_length",
        "prob_desc_description_translated_word_count",
        "prob_desc_description_translated_numeric_ratio",
        "prob_desc_description_translated_latex_ratio",
    ]

    # Features numériques complètes
    features_numeriques = features_numeriques_base + features_longueur_texte

    # Filtrer pour ne garder que les colonnes présentes
    features_latex = [f for f in features_latex if f in df_train.columns]
    features_numeriques = [f for f in features_numeriques if f in df_train.columns]

    print(f"  Features LaTeX:      {len(features_latex)}")
    print(f"  Features numériques: {len(features_numeriques)}")
    print(f"  TOTAL denses:        {len(features_latex) + len(features_numeriques)}")

    X_dense_train = df_train[features_numeriques + features_latex].values

    # ------------------------------------------------------------------
    # 3. Embeddings (all-MiniLM-L6-v2) - chargement ou calcul
    # ------------------------------------------------------------------
    print("\n[3/6] Embeddings all-MiniLM-L6-v2 (train)...")

    embeddings_train_path = data_dir / "embeddings_train_minilm.npy"
    if embeddings_train_path.exists():
        embeddings_train = np.load(embeddings_train_path)
        print(f"  Embeddings train chargés: {embeddings_train.shape}")
    else:
        print("  Fichier embeddings manquant, calcul en cours...")
        text_column = "prob_desc_description_translated"
        if text_column not in df_train.columns:
            raise ValueError(
                f"Colonne texte pour embeddings manquante: {text_column}"
            )

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts_train = df_train[text_column].fillna("").astype(str).tolist()

        start_time = time.time()
        embeddings_train = model.encode(
            texts_train,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        dur = time.time() - start_time
        print(f"  Embeddings calculés en {dur:.2f}s, shape={embeddings_train.shape}")

        np.save(embeddings_train_path, embeddings_train)
        print(f"  Embeddings sauvegardés: {embeddings_train_path}")

    # ------------------------------------------------------------------
    # 4. TF-IDF (fit sur train uniquement) - chargement ou calcul
    # ------------------------------------------------------------------
    print("\n[4/6] TF-IDF (train, fit sur train uniquement)...")

    vectorizer_path = data_dir / "tfidf_vectorizer.pkl"
    tfidf_train_path = data_dir / "X_tfidf_train.npz"

    vectorizer = None
    X_tfidf_train = None

    need_recompute_tfidf = False
    if vectorizer_path.exists() and tfidf_train_path.exists():
        try:
            vectorizer = joblib.load(vectorizer_path)
            X_tfidf_train = load_npz(tfidf_train_path)
            print(
                f"  TF-IDF chargé: vocab={len(vectorizer.vocabulary_):,}, "
                f"shape={X_tfidf_train.shape}"
            )
        except Exception as e:
            print(
                f"  Impossible de charger TF-IDF existant ({e}), "
                "recalcul complet..."
            )
            need_recompute_tfidf = True
    else:
        need_recompute_tfidf = True

    if need_recompute_tfidf:
        text_column_tfidf = "unified_document_without_latex"
        if text_column_tfidf not in df_train.columns:
            print(
                f"  ⚠️  Colonne {text_column_tfidf} manquante, utilisation de "
                "'unified_document'"
            )
            text_column_tfidf = "unified_document"

        if text_column_tfidf not in df_train.columns:
            raise ValueError(
                f"Aucune colonne texte disponible pour TF-IDF "
                f"({text_column_tfidf} ou unified_document)."
            )

        texts_train_tfidf = df_train[text_column_tfidf].fillna("").astype(str).tolist()

        print("  Création et fit du TfidfVectorizer sur le train...")
        vectorizer = TfidfVectorizer(
            analyzer="word",
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=5000,
            sublinear_tf=True,
            norm="l2",
            smooth_idf=True,
            stop_words=None,
        )

        start_time = time.time()
        X_tfidf_train = vectorizer.fit_transform(texts_train_tfidf)
        dur = time.time() - start_time
        print(
            f"  TF-IDF fitté en {dur:.2f}s, vocab={len(vectorizer.vocabulary_):,}, "
            f"shape={X_tfidf_train.shape}"
        )

        joblib.dump(vectorizer, vectorizer_path)
        save_npz(tfidf_train_path, X_tfidf_train)
        print(f"  Vectorizer sauvegardé: {vectorizer_path}")
        print(f"  X_tfidf_train sauvegardé: {tfidf_train_path}")

    # ------------------------------------------------------------------
    # 5. Normalisation des features denses & construction de X_train_logreg
    # ------------------------------------------------------------------
    print("\n[5/6] Normalisation des features denses & construction de X_train_logreg...")

    scaler_path = data_dir / "scaler_dense_features.pkl"
    scaler: StandardScaler

    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            print("  Scaler dense chargé depuis disque")
        except Exception as e:
            print(
                f"  Impossible de charger le scaler existant ({e}), "
                "re-fit sur train..."
            )
            scaler = StandardScaler()
            X_dense_train_scaled = scaler.fit_transform(X_dense_train)
            joblib.dump(scaler, scaler_path)
            print(f"  Nouveau scaler sauvegardé: {scaler_path}")
    else:
        print("  Scaler non trouvé, fit sur train...")
        scaler = StandardScaler()
        X_dense_train_scaled = scaler.fit_transform(X_dense_train)
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler sauvegardé: {scaler_path}")

    # Si on n'a pas déjà normalisé juste au-dessus
    if "X_dense_train_scaled" not in locals():
        X_dense_train_scaled = scaler.transform(X_dense_train)

    print(f"  Denses normalisées: shape={X_dense_train_scaled.shape}")

    # Concaténation complète (même ordre que dans le notebook)
    X_train_logreg = hstack([X_tfidf_train, embeddings_train, X_dense_train_scaled])
    print(f"  X_train_logreg: shape={X_train_logreg.shape}")

    # ------------------------------------------------------------------
    # 6. Entraînement de la LogisticRegression optimale & sauvegarde
    # ------------------------------------------------------------------
    print("\n[6/6] Entraînement du modèle LogisticRegression optimal...")

    y_train = df_train[target_columns].values
    print(f"  y_train: shape={y_train.shape}")

    # Hyperparamètres optimaux (du notebook 05_modeling_v1)
    logreg = OneVsRestClassifier(
        LogisticRegression(
            C=2,
            penalty="l2",
            solver="saga",
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        n_jobs=-1,
    )

    start_time = time.time()
    logreg.fit(X_train_logreg, y_train)
    dur = time.time() - start_time
    print(f"  Entraînement terminé en {dur:.2f}s ({dur/60:.2f} min)")

    # Sauvegarde du modèle sous un nom dédié CLI
    model_filename = "logreg_all_features_cli.pkl"
    model_path = models_dir / model_filename
    joblib.dump(logreg, model_path)
    size_mb = model_path.stat().st_size / (1024 * 1024)

    print(f"\nModèle ré-entrainé sauvegardé: {model_path}")
    print(f"   Taille: {size_mb:.2f} MB")
    print("\nRé-entrainement CLI terminé avec succès")
    print("=" * 80)


if __name__ == "__main__":
    main()


