"""
Prediction logic for inference.

This module provides the Predictor class that combines preprocessing
and model inference to make predictions on raw input data.
"""

import numpy as np
from typing import List, Dict, Optional
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from .preprocessor import InferencePreprocessor
from .model_loader import ModelLoader


class Predictor:
    """
    Main prediction class that orchestrates preprocessing and model inference.
    
    This class combines the InferencePreprocessor and ModelLoader to
    provide a simple interface for making predictions on raw input data.
    """
    
    def __init__(
        self,
        artifacts_dir: str = "data/processed",
        models_dir: str = "models",
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the predictor.
        
        Parameters
        ----------
        artifacts_dir : str
            Directory containing preprocessing artifacts
        models_dir : str
            Directory containing trained models
        embeddings_model_name : str
            Name of the SentenceTransformer model for embeddings
        """
        self.preprocessor = InferencePreprocessor(
            artifacts_dir=artifacts_dir,
            embeddings_model_name=embeddings_model_name
        )
        self.model_loader = ModelLoader(models_dir=models_dir)
        
        self.is_ready = False
    
    def initialize(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the predictor by loading artifacts and model.
        
        Parameters
        ----------
        model_path : str, optional
            Path to the model file. If None, model must be loaded separately.
        """
        print("Initializing predictor...")
        self.preprocessor.load_artifacts()
        
        if model_path is not None:
            self.model_loader.load_model(model_path)
        
        self.is_ready = True
        print("Predictor initialized successfully")
    
    def predict(
        self,
        df: pd.DataFrame,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions on input data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe
        return_proba : bool
            If True, return probability scores instead of binary predictions
        
        Returns
        -------
        np.ndarray
            Predictions (binary or probabilities) of shape (n_samples, n_labels)
        
        Raises
        ------
        ValueError
            If predictor is not initialized
        """
        if not self.is_ready:
            raise ValueError("Predictor not initialized. Call initialize() first.")
        
        if self.model_loader.model is None:
            raise ValueError("Model not loaded. Load a model before predicting.")
        
        # Preprocess data
        df_processed, X_tfidf, X_embeddings, X_dense = self.preprocessor.preprocess(df)
        
        # Combine features (same order as training: TF-IDF, embeddings, dense)
        if X_tfidf is not None:
            # Convert dense arrays to sparse for stacking with TF-IDF
            X_embeddings_sparse = csr_matrix(X_embeddings)
            X_dense_sparse = csr_matrix(X_dense)
            
            X_full = hstack([X_tfidf, X_embeddings_sparse, X_dense_sparse])
        else:
            # If TF-IDF is not available, combine embeddings and dense features
            X_full = np.hstack([X_embeddings, X_dense])
        
        # Make predictions
        if return_proba:
            predictions = self.model_loader.model.predict_proba(X_full)
        else:
            predictions = self.model_loader.model.predict(X_full)
        
        return predictions
    
    def predict_with_labels(
        self,
        df: pd.DataFrame,
        priority_tags: Optional[List[str]] = None
    ) -> List[Dict[str, any]]:
        """
        Make predictions and return with label names.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe
        priority_tags : list, optional
            List of priority tag names. If None, uses default from config.
        
        Returns
        -------
        list
            List of dictionaries with predictions for each sample
        """
        if priority_tags is None:
            from src.utils.eda_helpers import PRIORITY_TAGS
            priority_tags = PRIORITY_TAGS
        
        predictions = self.predict(df, return_proba=False)
        
        results = []
        for i, row in df.iterrows():
            sample_pred = predictions[i]
            predicted_tags = [priority_tags[j] for j in range(len(priority_tags)) if sample_pred[j] == 1]
            
            results.append({
                'src_uid': row.get('src_uid', f'sample_{i}'),
                'predicted_tags': predicted_tags,
                'num_tags': len(predicted_tags)
            })
        
        return results
