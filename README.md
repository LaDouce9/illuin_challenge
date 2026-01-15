# Illuin Challenge

This project contains the code, data and notebooks for the Illuin code classification challenge, plus a reusable prediction CLI.

## Prerequisites

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) for dependency management

## Installation

### 1. Install `uv`

You need to install `uv` to manage the project dependencies.

**Windows:**

You can install `uv` using pip:
```bash
pip install uv
```

Or via PowerShell (if allowed by your security policy):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**

Use the following command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup environment

Once `uv` is installed, synchronize the project dependencies:

```bash
uv sync
```

This will create a virtual environment in `.venv` with all the required packages.

### 3. (Optional but recommended) Configure Hugging Face caches

Configure the Hugging Face caches outside OneDrive or synced folders. Example on Windows PowerShell:

```powershell
$env:HF_HOME="C:\hf_cache"
$env:HF_HUB_CACHE="C:\hf_cache\hub"
$env:HF_DATASETS_CACHE="C:\hf_cache\datasets"
$env:TRANSFORMERS_CACHE="C:\hf_cache\transformers"
```

On macOS / Linux, export the same variables in your shell.

---

## Running Jupyter (local)

From the project root:

```bash
uv run jupyter lab
```

Then open the URL printed in the terminal (typically `http://localhost:8888/...`) and select the `.venv` kernel if needed.

### Notebooks overview

All notebooks are in the `notebooks/` directory:

- `01_eda.ipynb`  
  Exploratory data analysis on the raw dataset.

- `02_preprocessing_pipeline.ipynb`  
  Definition and validation of the preprocessing pipeline; saves artifacts to `data/processed/`.

- `03_tfidf_v1.ipynb`  
  Builds and saves the TF‑IDF vectorizer and matrices (`tfidf_vectorizer.pkl`, `X_tfidf_train.npz`, `X_tfidf_test.npz`).

- `04_embeddings_v1.ipynb`  
  Computes and saves sentence-transformer embeddings for train/test.

- `05_modeling_v1.ipynb`  
  Builds complete feature sets, trains and evaluates several models, runs GridSearch, and saves models to `models/`.

All notebooks assume that:

- Raw data is available under `data/raw/code_classification_dataset/`
- Preprocessed artifacts live in `data/processed/`

---

## Jupyter with Docker (optional)

If you prefer running everything inside Docker, you can use the existing `Dockerfile`.

### Build the Docker image

```bash
docker build -t illuin-challenge .
```

### Run Jupyter Lab

**Linux / macOS:**
```bash
docker run -d \
  --name illuin-jupyter \
  -p 8888:8888 \
  -v "$(pwd)":/app \
  illuin-challenge
```

**Windows (PowerShell):**
```powershell
docker run -d `
  --name illuin-jupyter `
  -p 8888:8888 `
  -v "${PWD}:/app" `
  illuin-challenge
```

Then open the Jupyter URL printed by Docker logs for the container.

---

## Prediction CLI

The CLI applies the full preprocessing + model pipeline to one or more JSON files.

### Basic usage

From the project root, to predict on a single file:

```bash
uv run python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json
```

To predict on all JSON files in a directory:

```bash
uv run python scripts/predict.py --input-dir data/raw/code_classification_dataset/
```

The CLI uses `scripts/prediction_config.json` for configuration, which by default points to:

- Model: `models/logreg_all_features_cli.pkl`
- Artifacts in `data/processed/` (imputation values, TF‑IDF vectorizer, scaler, etc.).

### Input JSON format

Each JSON file must contain at least:

- `prob_desc_description` – main problem statement (required)

Recommended (but not strictly required) fields:

- `src_uid` – unique ID for the problem
- `prob_desc_title`
- `prob_desc_input_spec`
- `prob_desc_output_spec`
- `prob_desc_notes`
- `prob_desc_time_limit` (e.g. `"3 seconds"`)
- `difficulty` (e.g. `1400`)

If `prob_desc_description` is missing or empty for all samples, the CLI will raise a clear error.

### Using the manual input template

A template is provided in `src/prediction/input_template.json`. Typical workflow:

1. Copy the template:

   ```bash
   cp src/prediction/input_template.json data/raw/code_classification_dataset/my_sample.json
   ```

2. Edit `my_sample.json` and fill at least `prob_desc_description` (and optionally the other fields).
3. Run the prediction:

   ```bash
   uv run python scripts/predict.py --input data/raw/code_classification_dataset/my_sample.json
   ```

Example of output:

```json
[
  {
    "src_uid": "test_1",
    "predicted_tags": ["trees"],
    "num_tags": 1
  }
]
```

---

## Retraining the optimal Logistic Regression model (script)

The script `scripts/train_logreg_best_cli.py` re-trains the optimal Logistic Regression model used by the CLI, based on the artifacts saved under `data/processed/`.

### What the script does

- Loads `data/processed/train_preprocessed.parquet`
- Reconstructs the dense features as in `05_modeling_v1.ipynb`
- Loads or (if needed) recomputes:
  - `data/processed/embeddings_train_minilm.npy`
  - `data/processed/tfidf_vectorizer.pkl` and `X_tfidf_train.npz`
  - `data/processed/scaler_dense_features.pkl`
- Builds `X_train_logreg` = TF‑IDF + embeddings + normalized dense features
- Trains a `OneVsRestClassifier(LogisticRegression)` with the best hyperparameters:
  - `C=2`, `penalty='l2'`, `solver='saga'`, `class_weight='balanced'`, `max_iter=1000`
- Saves the model as `models/logreg_all_features_cli.pkl`

### Running the training script

```bash
uv run python scripts/train_logreg_best_cli.py
```

Use this if you regenerate preprocessing artifacts and want the CLI model to be perfectly aligned with them.

---

## Quick recap

- **Setup:** `uv sync` (+ HF cache variables if needed)
- **Run notebooks:** `uv run jupyter lab` and open `notebooks/01_eda.ipynb` → `05_modeling_v1.ipynb`
- **Predict:** `uv run python scripts/predict.py --input <path-to-json>`
- **Retrain CLI model (optional):** `uv run python scripts/train_logreg_best_cli.py`