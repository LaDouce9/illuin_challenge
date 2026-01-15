"""
Preprocessing pipeline for inference.

This module provides the InferencePreprocessor class that applies
all preprocessing steps to raw input data, ensuring no data leakage
by using artifacts saved from the training phase.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

from src.utils.preprocessing import (
    clean_text_patterns,
    handle_difficulty_invalid_values,
)
from src.utils.translation_helpers import translate_column
from src.utils.numeric_analysis import convert_time_limit_column
from src.utils.text_analysis import (
    preprocess_text_full,
    create_unified_document,
    remove_latex_from_text
)
from src.utils.latex_analysis import extract_all_latex_symbols
from src.utils.preprocessing import (
    create_text_length_features,
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class InferencePreprocessor:
    """
    Preprocessing pipeline for inference data.
    
    This class applies all preprocessing steps to raw input data,
    using artifacts (imputation values, LaTeX symbols, etc.) saved
    from the training phase to ensure no data leakage.
    """
    
    def __init__(
        self,
        artifacts_dir: str = "data/processed",
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the inference preprocessor.
        
        Parameters
        ----------
        artifacts_dir : str
            Directory containing preprocessing artifacts
        embeddings_model_name : str
            Name of the SentenceTransformer model for embeddings
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.embeddings_model_name = embeddings_model_name
        
        # Artifacts to load
        self.imputation_values: Optional[Dict[str, float]] = None
        self.latex_symbols_train: Optional[pd.DataFrame] = None
        self.has_features_list: Optional[List[str]] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.scaler_dense: Optional[StandardScaler] = None
        self.embeddings_model: Optional[SentenceTransformer] = None
        
        # Columns to translate
        self.columns_to_translate = [
            'prob_desc_description',
            'prob_desc_input_spec',
            'prob_desc_output_spec',
            'prob_desc_notes'
        ]
    
    def load_artifacts(self) -> None:
        """
        Load all preprocessing artifacts from disk.
        
        Raises
        ------
        FileNotFoundError
            If required artifacts are not found
        """
        print("Loading preprocessing artifacts...")
        
        # Load imputation values
        imputation_path = self.artifacts_dir / "imputation_values.json"
        if not imputation_path.exists():
            raise FileNotFoundError(f"Imputation values not found: {imputation_path}")
        with open(imputation_path, 'r') as f:
            self.imputation_values = json.load(f)
        print(f"  Loaded imputation values: {list(self.imputation_values.keys())}")
        
        # Load LaTeX symbols from train (to identify which has_* features to create)
        # We need to load the train preprocessed data to extract the has_* features
        train_path = self.artifacts_dir / "train_preprocessed.parquet"
        if train_path.exists():
            df_train_sample = pd.read_parquet(train_path, nrows=1000)
            self.has_features_list = [
                col for col in df_train_sample.columns 
                if col.startswith('has_')
            ]
            print(f"  Identified {len(self.has_features_list)} LaTeX binary features from train")
        else:
            print("  WARNING: train_preprocessed.parquet not found, LaTeX features may be incomplete")
            self.has_features_list = []
        
        # Load TF-IDF vectorizer
        tfidf_path = self.artifacts_dir / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            print(f"  Loaded TF-IDF vectorizer (vocab size: {len(self.tfidf_vectorizer.vocabulary_)})")
        else:
            print("  WARNING: TF-IDF vectorizer not found, TF-IDF features will not be computed")
        
        # Load scaler for dense features
        scaler_path = self.artifacts_dir / "scaler_dense_features.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler_dense = pickle.load(f)
            print(f"  Loaded StandardScaler for dense features")
        else:
            print("  WARNING: Dense features scaler not found, normalization will be skipped")
        
        # Load embeddings model
        try:
            self.embeddings_model = SentenceTransformer(self.embeddings_model_name)
            print(f"  Loaded embeddings model: {self.embeddings_model_name}")
        except Exception as e:
            print(f"  ERROR: Failed to load embeddings model: {e}")
            raise
        
        print("Artifacts loaded successfully")
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply full preprocessing pipeline to input data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe with same structure as training data
        
        Returns
        -------
        df_processed : pd.DataFrame
            Preprocessed dataframe with all features
        X_tfidf : np.ndarray or scipy.sparse matrix
            TF-IDF features
        X_embeddings : np.ndarray
            Embeddings features
        X_dense : np.ndarray
            Dense features (numeric + LaTeX)
        """
        if self.imputation_values is None:
            raise ValueError("Artifacts must be loaded before preprocessing. Call load_artifacts() first.")
        
        df = df.copy()
        print(f"\nPreprocessing {len(df)} samples...")
        
        # Step 1: Text pattern cleaning
        print("\n[1/10] Text pattern cleaning...")
        df = clean_text_patterns(df, column='prob_desc_notes')
        
        # Step 2: Translation
        print("\n[2/10] Translation to English...")
        for col in self.columns_to_translate:
            df = translate_column(
                df,
                column=col,
                target_lang='en',
                new_column='_translated'
            )
        
        # Step 3: Numeric variable conversion
        print("\n[3/10] Numeric variable conversion...")
        df = convert_time_limit_column(df, column='prob_desc_time_limit')
        df = handle_difficulty_invalid_values(df, column='difficulty', invalid_value=-1)
        
        # Step 4: Text/LaTeX separation
        print("\n[4/10] Text/LaTeX separation...")
        latex_analysis = df['prob_desc_description_translated'].apply(preprocess_text_full)
        df['clean_description'] = latex_analysis.apply(lambda x: x[0])
        df['latex_features_desc'] = latex_analysis.apply(lambda x: x[1])
        df['nb_latex_blocks'] = df['latex_features_desc'].apply(lambda x: x['nb_latex_blocks'])
        df['nb_latex_symbols'] = df['latex_features_desc'].apply(lambda x: x['nb_latex_symbols'])
        df['latex_density'] = df['latex_features_desc'].apply(lambda x: x['latex_density'])
        df['latex_symbols_density'] = df['latex_features_desc'].apply(lambda x: x['latex_symbols_density'])
        
        # Step 5: LaTeX binary features (using same symbols as train)
        print("\n[5/10] LaTeX binary features...")
        if self.has_features_list:
            # Extract LaTeX symbols from current data
            latex_stats = extract_all_latex_symbols(df, column='prob_desc_description_translated')
            
            # Create only the has_* features that were selected during training
            # Map feature names to LaTeX symbols
            for feat_name in self.has_features_list:
                symbol_name = feat_name.replace('has_', '')
                # Try to find the symbol in latex_stats columns
                found = False
                for symbol in latex_stats.columns:
                    # Normalize symbol name for comparison
                    symbol_safe = symbol.replace('\\', '').replace('{', '').replace('}', '')
                    if symbol_safe == symbol_name:
                        df[feat_name] = latex_stats[symbol].values
                        found = True
                        break
                
                if not found:
                    # Symbol not present in this data, set to 0
                    df[feat_name] = 0
            
            print(f"  Created {len(self.has_features_list)} LaTeX binary features")
        else:
            print("  WARNING: No LaTeX binary features to create")
        
        # Step 6: Text length features
        print("\n[6/10] Text length features...")
        # Note: In the training pipeline, text length features are created ONLY for prob_desc_description_translated
        # This matches the preprocessing notebook (Section 8)
        df = create_text_length_features(
            df,
            text_columns=['prob_desc_description_translated'],
            compute_latex_ratio=True,
            latex_density_columns={'prob_desc_description_translated': 'latex_density'}
        )
        
        # Step 7: Unified document creation
        print("\n[7/10] Unified document creation...")
        # For unified document, use all translated columns (matching notebook Section 9)
        text_columns_unified = [
            'prob_desc_description_translated',
            'prob_desc_input_spec_translated',
            'prob_desc_output_spec_translated',
            'prob_desc_notes_translated'
        ]
        df['unified_document'] = df.apply(
            lambda row: create_unified_document(row, text_columns_unified), axis=1
        )
        df['unified_document_without_latex'] = df['unified_document'].apply(
            lambda x: remove_latex_from_text(x, replacement_token="[LATEX]")
        )
        
        # Step 8: Missing value imputation
        print("\n[8/10] Missing value imputation...")
        for col, value in self.imputation_values.items():
            if col in df.columns:
                df[col].fillna(value, inplace=True)
                print(f"  Imputed {col}: {value}")
        
        # Step 9: Extract dense features
        print("\n[9/10] Extracting dense features...")
        # Features LaTeX numériques (4 features - matching notebook Section 5)
        features_latex_numeriques = [
            'nb_latex_blocks', 'nb_latex_symbols', 'latex_density',
            'latex_symbols_density'
        ]
        # Note: latex_features_desc is NOT included in dense features (it's a dict column)
        
        # Features LaTeX binaires (has_* columns)
        features_latex_binaires = self.has_features_list if self.has_features_list else []
        
        # Features numériques de base (2 features)
        features_numeriques_base = [
            'difficulty', 'time_limit_seconds'
        ]
        
        # Features de longueur de texte (uniquement pour prob_desc_description_translated - 4 features)
        # Matching notebook Section 8
        features_longueur_texte = [
            'prob_desc_description_translated_char_length',
            'prob_desc_description_translated_word_count',
            'prob_desc_description_translated_numeric_ratio',
            'prob_desc_description_translated_latex_ratio'
        ]
        
        # Filter to only include columns that exist
        all_dense_features = (
            features_latex_numeriques + 
            features_latex_binaires + 
            features_numeriques_base + 
            features_longueur_texte
        )
        all_dense_features = [f for f in all_dense_features if f in df.columns]
        
        X_dense = df[all_dense_features].values
        
        # Normalize dense features if scaler is available
        if self.scaler_dense is not None:
            X_dense = self.scaler_dense.transform(X_dense)
            print(f"  Normalized {len(all_dense_features)} dense features")
        else:
            print(f"  Using {len(all_dense_features)} dense features (not normalized)")
        
        # Step 10: Compute embeddings
        print("\n[10/10] Computing embeddings...")
        if self.embeddings_model is not None:
            texts_for_embeddings = df['prob_desc_description_translated'].apply(
                lambda x: self._clean_text_for_embeddings(x)
            ).tolist()
            X_embeddings = self.embeddings_model.encode(
                texts_for_embeddings,
                batch_size=32,
                show_progress_bar=False
            )
            print(f"  Computed embeddings: {X_embeddings.shape}")
        else:
            raise ValueError("Embeddings model not loaded")
        
        # Step 11: Compute TF-IDF
        print("\n[11/11] Computing TF-IDF...")
        if self.tfidf_vectorizer is not None:
            X_tfidf = self.tfidf_vectorizer.transform(df['unified_document_without_latex'].tolist())
            print(f"  Computed TF-IDF: {X_tfidf.shape}")
        else:
            print("  WARNING: TF-IDF not computed (vectorizer not available)")
            X_tfidf = None
        
        print("\nPreprocessing completed successfully")
        
        return df, X_tfidf, X_embeddings, X_dense
    
    def _clean_text_for_embeddings(self, text: str) -> str:
        """
        Clean text for embeddings (same logic as in training).
        
        Parameters
        ----------
        text : str
            Raw text
        
        Returns
        -------
        str
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove LaTeX blocks and replace with token
        text = remove_latex_from_text(text, replacement_token="[LATEX]")
        
        return text



This module provides the InferencePreprocessor class that applies
all preprocessing steps to raw input data, ensuring no data leakage
by using artifacts saved from the training phase.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

from src.utils.preprocessing import (
    clean_text_patterns,
    handle_difficulty_invalid_values,
)
from src.utils.translation_helpers import translate_column
from src.utils.numeric_analysis import convert_time_limit_column
from src.utils.text_analysis import (
    preprocess_text_full,
    create_unified_document,
    remove_latex_from_text
)
from src.utils.latex_analysis import extract_all_latex_symbols
from src.utils.preprocessing import (
    create_text_length_features,
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class InferencePreprocessor:
    """
    Preprocessing pipeline for inference data.
    
    This class applies all preprocessing steps to raw input data,
    using artifacts (imputation values, LaTeX symbols, etc.) saved
    from the training phase to ensure no data leakage.
    """
    
    def __init__(
        self,
        artifacts_dir: str = "data/processed",
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the inference preprocessor.
        
        Parameters
        ----------
        artifacts_dir : str
            Directory containing preprocessing artifacts
        embeddings_model_name : str
            Name of the SentenceTransformer model for embeddings
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.embeddings_model_name = embeddings_model_name
        
        # Artifacts to load
        self.imputation_values: Optional[Dict[str, float]] = None
        self.latex_symbols_train: Optional[pd.DataFrame] = None
        self.has_features_list: Optional[List[str]] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.scaler_dense: Optional[StandardScaler] = None
        self.embeddings_model: Optional[SentenceTransformer] = None
        
        # Columns to translate
        self.columns_to_translate = [
            'prob_desc_description',
            'prob_desc_input_spec',
            'prob_desc_output_spec',
            'prob_desc_notes'
        ]
    
    def load_artifacts(self) -> None:
        """
        Load all preprocessing artifacts from disk.
        
        Raises
        ------
        FileNotFoundError
            If required artifacts are not found
        """
        print("Loading preprocessing artifacts...")
        
        # Load imputation values
        imputation_path = self.artifacts_dir / "imputation_values.json"
        if not imputation_path.exists():
            raise FileNotFoundError(f"Imputation values not found: {imputation_path}")
        with open(imputation_path, 'r') as f:
            self.imputation_values = json.load(f)
        print(f"  Loaded imputation values: {list(self.imputation_values.keys())}")
        
        # Load LaTeX symbols from train (to identify which has_* features to create)
        # We need to load the train preprocessed data to extract the has_* features
        train_path = self.artifacts_dir / "train_preprocessed.parquet"
        if train_path.exists():
            df_train_sample = pd.read_parquet(train_path, nrows=1000)
            self.has_features_list = [
                col for col in df_train_sample.columns 
                if col.startswith('has_')
            ]
            print(f"  Identified {len(self.has_features_list)} LaTeX binary features from train")
        else:
            print("  WARNING: train_preprocessed.parquet not found, LaTeX features may be incomplete")
            self.has_features_list = []
        
        # Load TF-IDF vectorizer
        tfidf_path = self.artifacts_dir / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            print(f"  Loaded TF-IDF vectorizer (vocab size: {len(self.tfidf_vectorizer.vocabulary_)})")
        else:
            print("  WARNING: TF-IDF vectorizer not found, TF-IDF features will not be computed")
        
        # Load scaler for dense features
        scaler_path = self.artifacts_dir / "scaler_dense_features.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler_dense = pickle.load(f)
            print(f"  Loaded StandardScaler for dense features")
        else:
            print("  WARNING: Dense features scaler not found, normalization will be skipped")
        
        # Load embeddings model
        try:
            self.embeddings_model = SentenceTransformer(self.embeddings_model_name)
            print(f"  Loaded embeddings model: {self.embeddings_model_name}")
        except Exception as e:
            print(f"  ERROR: Failed to load embeddings model: {e}")
            raise
        
        print("Artifacts loaded successfully")
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply full preprocessing pipeline to input data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe with same structure as training data
        
        Returns
        -------
        df_processed : pd.DataFrame
            Preprocessed dataframe with all features
        X_tfidf : np.ndarray or scipy.sparse matrix
            TF-IDF features
        X_embeddings : np.ndarray
            Embeddings features
        X_dense : np.ndarray
            Dense features (numeric + LaTeX)
        """
        if self.imputation_values is None:
            raise ValueError("Artifacts must be loaded before preprocessing. Call load_artifacts() first.")
        
        df = df.copy()
        print(f"\nPreprocessing {len(df)} samples...")
        
        # Step 1: Text pattern cleaning
        print("\n[1/10] Text pattern cleaning...")
        df = clean_text_patterns(df, column='prob_desc_notes')
        
        # Step 2: Translation
        print("\n[2/10] Translation to English...")
        for col in self.columns_to_translate:
            df = translate_column(
                df,
                column=col,
                target_lang='en',
                new_column='_translated'
            )
        
        # Step 3: Numeric variable conversion
        print("\n[3/10] Numeric variable conversion...")
        df = convert_time_limit_column(df, column='prob_desc_time_limit')
        df = handle_difficulty_invalid_values(df, column='difficulty', invalid_value=-1)
        
        # Step 4: Text/LaTeX separation
        print("\n[4/10] Text/LaTeX separation...")
        latex_analysis = df['prob_desc_description_translated'].apply(preprocess_text_full)
        df['clean_description'] = latex_analysis.apply(lambda x: x[0])
        df['latex_features_desc'] = latex_analysis.apply(lambda x: x[1])
        df['nb_latex_blocks'] = df['latex_features_desc'].apply(lambda x: x['nb_latex_blocks'])
        df['nb_latex_symbols'] = df['latex_features_desc'].apply(lambda x: x['nb_latex_symbols'])
        df['latex_density'] = df['latex_features_desc'].apply(lambda x: x['latex_density'])
        df['latex_symbols_density'] = df['latex_features_desc'].apply(lambda x: x['latex_symbols_density'])
        
        # Step 5: LaTeX binary features (using same symbols as train)
        print("\n[5/10] LaTeX binary features...")
        if self.has_features_list:
            # Extract LaTeX symbols from current data
            latex_stats = extract_all_latex_symbols(df, column='prob_desc_description_translated')
            
            # Create only the has_* features that were selected during training
            # Map feature names to LaTeX symbols
            for feat_name in self.has_features_list:
                symbol_name = feat_name.replace('has_', '')
                # Try to find the symbol in latex_stats columns
                found = False
                for symbol in latex_stats.columns:
                    # Normalize symbol name for comparison
                    symbol_safe = symbol.replace('\\', '').replace('{', '').replace('}', '')
                    if symbol_safe == symbol_name:
                        df[feat_name] = latex_stats[symbol].values
                        found = True
                        break
                
                if not found:
                    # Symbol not present in this data, set to 0
                    df[feat_name] = 0
            
            print(f"  Created {len(self.has_features_list)} LaTeX binary features")
        else:
            print("  WARNING: No LaTeX binary features to create")
        
        # Step 6: Text length features
        print("\n[6/10] Text length features...")
        # Note: In the training pipeline, text length features are created ONLY for prob_desc_description_translated
        # This matches the preprocessing notebook (Section 8)
        df = create_text_length_features(
            df,
            text_columns=['prob_desc_description_translated'],
            compute_latex_ratio=True,
            latex_density_columns={'prob_desc_description_translated': 'latex_density'}
        )
        
        # Step 7: Unified document creation
        print("\n[7/10] Unified document creation...")
        # For unified document, use all translated columns (matching notebook Section 9)
        text_columns_unified = [
            'prob_desc_description_translated',
            'prob_desc_input_spec_translated',
            'prob_desc_output_spec_translated',
            'prob_desc_notes_translated'
        ]
        df['unified_document'] = df.apply(
            lambda row: create_unified_document(row, text_columns_unified), axis=1
        )
        df['unified_document_without_latex'] = df['unified_document'].apply(
            lambda x: remove_latex_from_text(x, replacement_token="[LATEX]")
        )
        
        # Step 8: Missing value imputation
        print("\n[8/10] Missing value imputation...")
        for col, value in self.imputation_values.items():
            if col in df.columns:
                df[col].fillna(value, inplace=True)
                print(f"  Imputed {col}: {value}")
        
        # Step 9: Extract dense features
        print("\n[9/10] Extracting dense features...")
        # Features LaTeX numériques (4 features - matching notebook Section 5)
        features_latex_numeriques = [
            'nb_latex_blocks', 'nb_latex_symbols', 'latex_density',
            'latex_symbols_density'
        ]
        # Note: latex_features_desc is NOT included in dense features (it's a dict column)
        
        # Features LaTeX binaires (has_* columns)
        features_latex_binaires = self.has_features_list if self.has_features_list else []
        
        # Features numériques de base (2 features)
        features_numeriques_base = [
            'difficulty', 'time_limit_seconds'
        ]
        
        # Features de longueur de texte (uniquement pour prob_desc_description_translated - 4 features)
        # Matching notebook Section 8
        features_longueur_texte = [
            'prob_desc_description_translated_char_length',
            'prob_desc_description_translated_word_count',
            'prob_desc_description_translated_numeric_ratio',
            'prob_desc_description_translated_latex_ratio'
        ]
        
        # Filter to only include columns that exist
        all_dense_features = (
            features_latex_numeriques + 
            features_latex_binaires + 
            features_numeriques_base + 
            features_longueur_texte
        )
        all_dense_features = [f for f in all_dense_features if f in df.columns]
        
        X_dense = df[all_dense_features].values
        
        # Normalize dense features if scaler is available
        if self.scaler_dense is not None:
            X_dense = self.scaler_dense.transform(X_dense)
            print(f"  Normalized {len(all_dense_features)} dense features")
        else:
            print(f"  Using {len(all_dense_features)} dense features (not normalized)")
        
        # Step 10: Compute embeddings
        print("\n[10/10] Computing embeddings...")
        if self.embeddings_model is not None:
            texts_for_embeddings = df['prob_desc_description_translated'].apply(
                lambda x: self._clean_text_for_embeddings(x)
            ).tolist()
            X_embeddings = self.embeddings_model.encode(
                texts_for_embeddings,
                batch_size=32,
                show_progress_bar=False
            )
            print(f"  Computed embeddings: {X_embeddings.shape}")
        else:
            raise ValueError("Embeddings model not loaded")
        
        # Step 11: Compute TF-IDF
        print("\n[11/11] Computing TF-IDF...")
        if self.tfidf_vectorizer is not None:
            X_tfidf = self.tfidf_vectorizer.transform(df['unified_document_without_latex'].tolist())
            print(f"  Computed TF-IDF: {X_tfidf.shape}")
        else:
            print("  WARNING: TF-IDF not computed (vectorizer not available)")
            X_tfidf = None
        
        print("\nPreprocessing completed successfully")
        
        return df, X_tfidf, X_embeddings, X_dense
    
    def _clean_text_for_embeddings(self, text: str) -> str:
        """
        Clean text for embeddings (same logic as in training).
        
        Parameters
        ----------
        text : str
            Raw text
        
        Returns
        -------
        str
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove LaTeX blocks and replace with token
        text = remove_latex_from_text(text, replacement_token="[LATEX]")
        
        return text

