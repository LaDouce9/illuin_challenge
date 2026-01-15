"""
Model loading utilities for inference.

This module provides the ModelLoader class for loading trained models
and associated artifacts for making predictions.
"""

import pickle
from pathlib import Path
from typing import Optional, Any
import json


class ModelLoader:
    """
    Loader for trained models and prediction artifacts.
    
    This class handles loading of:
    - Trained model (pickle file)
    - Model configuration
    - Any other model-specific artifacts
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model loader.
        
        Parameters
        ----------
        models_dir : str
            Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.model: Optional[Any] = None
        self.model_config: Optional[dict] = None
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Parameters
        ----------
        model_path : str
            Path to the model file (relative to models_dir or absolute)
        
        Raises
        ------
        FileNotFoundError
            If model file is not found
        """
        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = self.models_dir / model_path
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        print(f"Loading model from {model_file}...")
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        print("Model loaded successfully")
    
    def load_config(self, config_path: str) -> dict:
        """
        Load model configuration from JSON file.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        
        Returns
        -------
        dict
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = Path("scripts") / config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.model_config = config
        return config

Model loading utilities for inference.

This module provides the ModelLoader class for loading trained models
and associated artifacts for making predictions.
"""

import pickle
from pathlib import Path
from typing import Optional, Any
import json


class ModelLoader:
    """
    Loader for trained models and prediction artifacts.
    
    This class handles loading of:
    - Trained model (pickle file)
    - Model configuration
    - Any other model-specific artifacts
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model loader.
        
        Parameters
        ----------
        models_dir : str
            Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.model: Optional[Any] = None
        self.model_config: Optional[dict] = None
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Parameters
        ----------
        model_path : str
            Path to the model file (relative to models_dir or absolute)
        
        Raises
        ------
        FileNotFoundError
            If model file is not found
        """
        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = self.models_dir / model_path
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        print(f"Loading model from {model_file}...")
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        print("Model loaded successfully")
    
    def load_config(self, config_path: str) -> dict:
        """
        Load model configuration from JSON file.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        
        Returns
        -------
        dict
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = Path("scripts") / config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.model_config = config
        return config

