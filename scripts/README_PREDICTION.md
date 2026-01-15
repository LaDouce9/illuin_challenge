# Prediction Scripts

This directory contains scripts for making predictions on new data using trained models.

## Structure

- `predict.py`: Main CLI script for making predictions
- `prediction_config.json`: Configuration file specifying model and preprocessing artifacts

## Usage

### Basic Usage

Predict on a single file:
```bash
python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json
```

Predict on all files in a directory:
```bash
python scripts/predict.py --input-dir data/raw/predictions/
```

Save predictions to a file:
```bash
python scripts/predict.py --input sample.json --output predictions.json
```

### Advanced Usage

Use a custom configuration file:
```bash
python scripts/predict.py --input sample.json --config custom_config.json
```

Specify model path directly:
```bash
python scripts/predict.py --input sample.json --model models/best_model.pkl
```

Custom artifacts directory:
```bash
python scripts/predict.py --input sample.json --artifacts-dir custom/processed/
```

## Configuration

The `prediction_config.json` file specifies:
- Model path and type
- Paths to preprocessing artifacts (imputation values, TF-IDF vectorizer, scaler, etc.)
- Preprocessing parameters

## Input Format

Input files should be JSON files with the same structure as the training data:
- `prob_desc_description`: Problem description
- `prob_desc_input_spec`: Input specification
- `prob_desc_output_spec`: Output specification
- `prob_desc_notes`: Additional notes
- `difficulty`: Problem difficulty
- `prob_desc_time_limit`: Time limit
- Other fields as in the original dataset

## Output Format

The output is a JSON file (or stdout) containing predictions:
```json
[
  {
    "src_uid": "sample_1",
    "predicted_tags": ["math", "number theory"],
    "num_tags": 2
  }
]
```

## Requirements

All preprocessing artifacts must be available in `data/processed/`:
- `imputation_values.json`: Values for missing value imputation
- `train_preprocessed.parquet`: For extracting LaTeX feature names
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer fitted on training data
- `scaler_dense_features.pkl`: StandardScaler for dense features

The trained model must be available in `models/` or specified via `--model`.


This directory contains scripts for making predictions on new data using trained models.

## Structure

- `predict.py`: Main CLI script for making predictions
- `prediction_config.json`: Configuration file specifying model and preprocessing artifacts

## Usage

### Basic Usage

Predict on a single file:
```bash
python scripts/predict.py --input data/raw/code_classification_dataset/sample_1.json
```

Predict on all files in a directory:
```bash
python scripts/predict.py --input-dir data/raw/predictions/
```

Save predictions to a file:
```bash
python scripts/predict.py --input sample.json --output predictions.json
```

### Advanced Usage

Use a custom configuration file:
```bash
python scripts/predict.py --input sample.json --config custom_config.json
```

Specify model path directly:
```bash
python scripts/predict.py --input sample.json --model models/best_model.pkl
```

Custom artifacts directory:
```bash
python scripts/predict.py --input sample.json --artifacts-dir custom/processed/
```

## Configuration

The `prediction_config.json` file specifies:
- Model path and type
- Paths to preprocessing artifacts (imputation values, TF-IDF vectorizer, scaler, etc.)
- Preprocessing parameters

## Input Format

Input files should be JSON files with the same structure as the training data:
- `prob_desc_description`: Problem description
- `prob_desc_input_spec`: Input specification
- `prob_desc_output_spec`: Output specification
- `prob_desc_notes`: Additional notes
- `difficulty`: Problem difficulty
- `prob_desc_time_limit`: Time limit
- Other fields as in the original dataset

## Output Format

The output is a JSON file (or stdout) containing predictions:
```json
[
  {
    "src_uid": "sample_1",
    "predicted_tags": ["math", "number theory"],
    "num_tags": 2
  }
]
```

## Requirements

All preprocessing artifacts must be available in `data/processed/`:
- `imputation_values.json`: Values for missing value imputation
- `train_preprocessed.parquet`: For extracting LaTeX feature names
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer fitted on training data
- `scaler_dense_features.pkl`: StandardScaler for dense features

The trained model must be available in `models/` or specified via `--model`.

