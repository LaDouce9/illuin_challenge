# 📄 TF-IDF Feature Extraction V1 - Guide d'Utilisation

## 🎯 Objectif

Le notebook `05_tfidf_v1.ipynb` construit une première version de features TF-IDF sur la colonne `unified_documents` du dataset preprocessé.

**Ce notebook ne fait PAS de modélisation**, uniquement :
- ✅ Preprocessing minimal du texte
- ✅ Statistiques sur le vocabulaire
- ✅ Construction du TfidfVectorizer
- ✅ Sauvegarde des artefacts

---

## 📋 Prérequis

### Fichiers Requis

Le notebook nécessite :
```
data/processed/train_preprocessed.parquet
```

Ce fichier est généré par le notebook `04_preprocessing_pipeline.ipynb`.

### Packages Python

Tous les packages sont déjà installés dans l'environnement du projet :
- `pandas`, `numpy`, `scikit-learn`
- `scipy` (pour matrices sparse)
- `joblib` (pour sauvegarde du vectorizer)

---

## 🚀 Exécution

### Option 1 : Jupyter Lab (Docker)

```bash
# Lancer le container Docker
docker start illuin-jupyter

# Ouvrir Jupyter Lab dans le navigateur
http://localhost:8888/?token=illuin2024

# Naviguer vers notebooks/05_tfidf_v1.ipynb
# Executer toutes les cellules (Kernel > Restart & Run All)
```

### Option 2 : Local avec uv

```bash
cd notebooks
uv run jupyter lab 05_tfidf_v1.ipynb
```

---

## 📊 Structure du Notebook

### 1. Imports et Configuration
- Chargement des librairies
- **Configuration des tokens LaTeX** (modifiable facilement)

```python
LATEX_COMMANDS_TO_TOKEN = {
    r'\\gcd': 'LATEX_GCD',
    r'\\lcm': 'LATEX_LCM',
    # ... 11 commandes au total
}
```

### 2. Chargement des Données
- Charge uniquement le **TRAIN set** (évite le leakage)
- Vérifie la présence de `unified_document`

### 3. Preprocessing du Texte
Preprocessing **minimal mais robuste** :
- ✅ Remplacement des blocs math LaTeX (`$...$`, `$$...$$`) par `MATHBLOCK`
- ✅ Remplacement des commandes LaTeX spécifiques (gcd, lcm, etc.) par tokens
- ✅ Normalisation des espaces
- ✅ Lowercase
- ✅ **Conservation** des chiffres et expressions techniques (ex: `1e9+7`)
- ❌ **PAS de stop-words removal** en V1
- ❌ **PAS de lemmatization** en V1

### 4. Statistiques sur le Texte
- Distribution des longueurs (caractères)
- Statistiques sur `MATHBLOCK` (nombre de documents, distribution)
- **5 exemples avant/après** preprocessing (pris aléatoirement)

### 5. Statistiques sur le Vocabulaire
- **Unigrams** :
  - Nombre total de tokens
  - Nombre de tokens uniques
  - Top 30 tokens les plus fréquents
- **Bigrams** :
  - Top 30 bigrams les plus fréquents (avec `CountVectorizer`)

### 6. Construction du TfidfVectorizer

**Paramètres V1 EXACTS** :
```python
TfidfVectorizer(
    analyzer='word',
    lowercase=False,        # Déjà fait en preprocessing
    ngram_range=(1, 2),     # Unigrams + bigrams
    min_df=2,               # Minimum 2 documents
    max_df=0.95,            # Maximum 95% des documents
    max_features=100000,    # Max 100k features
    sublinear_tf=True,      # Log-scaling des fréquences
    norm='l2',              # Normalisation L2
    smooth_idf=True,
    stop_words=None         # Pas de stop-words
)
```

**Fit** sur `unified_document_clean` (TRAIN uniquement).

### 7. Analyse du Vocabulaire TF-IDF
- **Taille du vocabulaire final**
- **Top 30 tokens avec plus forte IDF** (rares, informatifs)
- **Top 30 tokens avec plus faible IDF** (très fréquents)
- Shape de la matrice TF-IDF
- Densité (% de valeurs non-nulles)
- Mémoire utilisée

### 8. Sauvegarde des Artefacts

Fichiers sauvegardés dans `data/processed/` :

| Fichier | Description | Format |
|---------|-------------|--------|
| `tfidf_vectorizer_v1.pkl` | Vectorizer fit (réutilisable pour transform) | Pickle (joblib) |
| `X_tfidf_train_v1.npz` | Matrice TF-IDF sparse (TRAIN) | Scipy sparse (.npz) |
| `train_with_tfidf_clean.parquet` | DataFrame avec `unified_document_clean` | Parquet |

### 9. Résumé Final
Affichage récapitulatif de toutes les métriques.

---

## 📦 Outputs

### Variables en Mémoire (fin de notebook)

| Variable | Type | Description |
|----------|------|-------------|
| `df` | DataFrame | Dataset avec colonne `unified_document_clean` |
| `vectorizer` | TfidfVectorizer | Vectorizer fit sur TRAIN |
| `X_tfidf` | csr_matrix | Matrice sparse TF-IDF (shape: n_docs × vocab_size) |

### Fichiers Sauvegardés

```
data/processed/
├── tfidf_vectorizer_v1.pkl           # Vectorizer
├── X_tfidf_train_v1.npz              # Matrice TF-IDF
└── train_with_tfidf_clean.parquet    # DataFrame avec texte clean
```

---

## 🔧 Modification des Tokens LaTeX

Pour ajouter/retirer des commandes LaTeX à tokeniser, **modifier la cellule 3** :

```python
LATEX_COMMANDS_TO_TOKEN = {
    r'\\gcd': 'LATEX_GCD',
    r'\\lcm': 'LATEX_LCM',
    # Ajouter d'autres commandes ici:
    r'\\alpha': 'LATEX_ALPHA',
    r'\\beta': 'LATEX_BETA',
}
```

Puis **relancer toutes les cellules** du notebook.

---

## 📊 Exemple de Sortie

```
================================================================================
RESUME FINAL - TF-IDF V1
================================================================================

DATA:
  Nombre de documents (TRAIN):            3,979
  Documents vides/courts (<50 chars):         12
  Documents avec MATHBLOCK:                2,847

PREPROCESSING:
  Commandes LaTeX tokenisees:                 11
  Exemple tokens: LATEX_GCD, LATEX_LCM, LATEX_BINOM, LATEX_PR, LATEX_SQRT

VOCABULAIRE BRUT:
  Nombre total de tokens:               1,245,678
  Tokens uniques:                          45,892

TF-IDF:
  Shape matrice:                      (3979, 45892)
  Vocabulaire final:                       45,892 tokens
  Densite:                              0.012345 (1.2345%)
  Memoire:                                 12.34 MB

ARTEFACTS SAUVEGARDES:
  - tfidf_vectorizer_v1.pkl
  - X_tfidf_train_v1.npz
  - train_with_tfidf_clean.parquet

VARIABLES EN MEMOIRE:
  - df: DataFrame avec colonne 'unified_document_clean'
  - vectorizer: TfidfVectorizer fit
  - X_tfidf: Matrice sparse TF-IDF (shape (3979, 45892))

================================================================================
NOTEBOOK TERMINE - Pret pour modelisation
================================================================================
```

---

## 🎯 Utilisation pour Modélisation

### Charger les Artefacts Sauvegardés

```python
import joblib
from scipy.sparse import load_npz

# Charger le vectorizer
vectorizer = joblib.load('data/processed/tfidf_vectorizer_v1.pkl')

# Charger la matrice TF-IDF train
X_train_tfidf = load_npz('data/processed/X_tfidf_train_v1.npz')

# Transformer le TEST set (sans refitter)
df_test = pd.read_parquet('data/processed/test_preprocessed.parquet')
df_test['unified_document_clean'] = df_test['unified_document'].apply(clean_text)
X_test_tfidf = vectorizer.transform(df_test['unified_document_clean'])
```

### Utiliser dans un Pipeline Sklearn

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Le vectorizer est déjà fit, mais on peut l'intégrer dans un pipeline
pipeline = Pipeline([
    # ('tfidf', vectorizer),  # Déjà fit séparément
    ('clf', LogisticRegression())
])

# Fit sur les features TF-IDF pré-calculées
pipeline.fit(X_train_tfidf, y_train)
```

---

## ⚠️ Points d'Attention

### 1. **Train/Test Split**
Le notebook travaille **uniquement sur TRAIN** pour éviter le data leakage. Pour le TEST :
- Charger `test_preprocessed.parquet`
- Appliquer la **même fonction `clean_text`**
- Utiliser `vectorizer.transform()` (PAS `fit_transform()`)

### 2. **Preprocessing Cohérent**
Pour l'inférence, toujours appliquer le **même preprocessing** :
```python
text_clean = clean_text(text_raw, latex_commands=LATEX_COMMANDS_TO_TOKEN)
X_new = vectorizer.transform([text_clean])
```

### 3. **Vocabulaire Fixé**
Le vocabulaire est fixé après le fit. Les nouveaux tokens en production seront ignorés.

---

## 🔄 Prochaines Étapes (V2)

Améliorations possibles pour une V2 :
- [ ] Ajouter des stop-words spécifiques au domaine
- [ ] Expérimenter avec `max_features` (50k, 150k)
- [ ] Tester `ngram_range=(1,3)` (trigrams)
- [ ] Lemmatization ou stemming
- [ ] Ajuster `min_df` et `max_df`
- [ ] Combiner TF-IDF avec d'autres features (LaTeX, numériques)

---

## 📝 Notes Techniques

### Fonction `clean_text()`

La fonction est **pure** (sans effet de bord) et peut être réutilisée :
```python
def clean_text(text: str, latex_commands: dict = None) -> str:
    """Nettoie un texte avec preprocessing minimal pour TF-IDF"""
    # Voir cellule 7 du notebook pour l'implémentation complète
```

### Gestion des NaN
Les valeurs `NaN` dans `unified_document` sont automatiquement remplacées par `""` dans `clean_text()`.

---

## ✅ Checklist d'Exécution

Avant d'exécuter le notebook :
- [ ] Le fichier `train_preprocessed.parquet` existe
- [ ] L'environnement Python a tous les packages requis
- [ ] Le répertoire `data/processed/` est accessible en écriture

Après exécution :
- [ ] Vérifier que 3 fichiers sont créés dans `data/processed/`
- [ ] Vérifier que le vocabulaire final > 0
- [ ] Vérifier que la densité est < 5% (matrice sparse)
- [ ] Examiner les exemples avant/après pour valider le preprocessing

---

## 🎉 Résultat

Vous disposez maintenant de :
- ✅ Un vectorizer TF-IDF **fit et prêt** pour l'inférence
- ✅ Une matrice sparse **efficace** en mémoire
- ✅ Un preprocessing **reproductible** et **configurable**
- ✅ Des statistiques complètes sur le vocabulaire

**Prêt pour la modélisation !** 🚀

