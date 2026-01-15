#!/usr/bin/env python3
"""
CLI script for making predictions on code classification data.

This script loads raw JSON files, applies preprocessing, and makes
predictions using a trained model.

Usage:
    python scripts/predict.py --input sample.json
    python scripts/predict.py --input-dir data/raw/predictions/
    python scripts/predict.py --input sample.json --output predictions.json
    python scripts/predict.py --input sample.json --config prediction_config.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prediction.predictor import Predictor
from src.utils.eda_helpers import load_dataset, PRIORITY_TAGS


def load_input_files(input_path: str) -> List[Dict[str, Any]]:
    """
    Load input files (single file or directory).
    
    Parameters
    ----------
    input_path : str
        Path to a single JSON file or directory containing JSON files
    
    Returns
    -------
    list
        List of dictionaries loaded from JSON files
    """
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        # Single file
        with open(input_path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [data]
    
    elif input_path_obj.is_dir():
        # Directory of files
        json_files = list(input_path_obj.glob('*.json'))
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {input_path}")
        
        data = []
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        
        return data
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def save_predictions(predictions: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save predictions to JSON file.
    
    Parameters
    ----------
    predictions : list
        List of prediction dictionaries
    output_path : str
        Path to output JSON file
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"\nPredictions saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Make predictions on code classification data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on a single file
  python scripts/predict.py --input sample.json
  
  # Predict on all files in a directory
  python scripts/predict.py --input-dir data/raw/predictions/
  
  # Save predictions to file
  python scripts/predict.py --input sample.json --output predictions.json
  
  # Use custom config
  python scripts/predict.py --input sample.json --config prediction_config.json
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=str,
        help='Path to a single JSON input file'
    )
    input_group.add_argument(
        '--input-dir',
        type=str,
        dest='input_dir',
        help='Path to directory containing JSON input files'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output JSON file (default: print to stdout)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='prediction_config.json',
        help='Path to prediction configuration file (default: prediction_config.json)'
    )
    
    # Model options
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (overrides config)'
    )
    
    # Artifacts directory
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        default='data/processed',
        help='Directory containing preprocessing artifacts (default: data/processed)'
    )
    
    # Models directory
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    args = parser.parse_args()
    
    # Determine input path
    input_path = args.input if args.input else args.input_dir
    
    # Load configuration
    config_path = Path("scripts") / args.config
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_path = args.model or config.get('model', {}).get('path')
    else:
        print(f"Warning: Config file not found: {config_path}")
        print("Using command-line arguments only")
        model_path = args.model
    
    if model_path is None:
        print("Error: Model path must be specified either in config or via --model")
        sys.exit(1)
    
    try:
        # Load input data
        print(f"Loading input from: {input_path}")
        input_data = load_input_files(input_path)
        print(f"Loaded {len(input_data)} sample(s)")
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(input_data)
        
        # Initialize predictor
        predictor = Predictor(
            artifacts_dir=args.artifacts_dir,
            models_dir=args.models_dir
        )
        predictor.initialize(model_path=model_path)
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = predictor.predict_with_labels(df)
        
        # Output results
        if args.output:
            save_predictions(predictions, args.output)
        else:
            # Print to stdout
            print("\nPredictions:")
            print(json.dumps(predictions, indent=2, ensure_ascii=False))
        
        print("\nPrediction completed successfully")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
CLI script for making predictions on code classification data.

This script loads raw JSON files, applies preprocessing, and makes
predictions using a trained model.

Usage:
    python scripts/predict.py --input sample.json
    python scripts/predict.py --input-dir data/raw/predictions/
    python scripts/predict.py --input sample.json --output predictions.json
    python scripts/predict.py --input sample.json --config prediction_config.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prediction.predictor import Predictor
from src.utils.eda_helpers import load_dataset, PRIORITY_TAGS


def load_input_files(input_path: str) -> List[Dict[str, Any]]:
    """
    Load input files (single file or directory).
    
    Parameters
    ----------
    input_path : str
        Path to a single JSON file or directory containing JSON files
    
    Returns
    -------
    list
        List of dictionaries loaded from JSON files
    """
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        # Single file
        with open(input_path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [data]
    
    elif input_path_obj.is_dir():
        # Directory of files
        json_files = list(input_path_obj.glob('*.json'))
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {input_path}")
        
        data = []
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        
        return data
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def save_predictions(predictions: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save predictions to JSON file.
    
    Parameters
    ----------
    predictions : list
        List of prediction dictionaries
    output_path : str
        Path to output JSON file
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"\nPredictions saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Make predictions on code classification data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on a single file
  python scripts/predict.py --input sample.json
  
  # Predict on all files in a directory
  python scripts/predict.py --input-dir data/raw/predictions/
  
  # Save predictions to file
  python scripts/predict.py --input sample.json --output predictions.json
  
  # Use custom config
  python scripts/predict.py --input sample.json --config prediction_config.json
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=str,
        help='Path to a single JSON input file'
    )
    input_group.add_argument(
        '--input-dir',
        type=str,
        dest='input_dir',
        help='Path to directory containing JSON input files'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output JSON file (default: print to stdout)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='prediction_config.json',
        help='Path to prediction configuration file (default: prediction_config.json)'
    )
    
    # Model options
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (overrides config)'
    )
    
    # Artifacts directory
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        default='data/processed',
        help='Directory containing preprocessing artifacts (default: data/processed)'
    )
    
    # Models directory
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    args = parser.parse_args()
    
    # Determine input path
    input_path = args.input if args.input else args.input_dir
    
    # Load configuration
    config_path = Path("scripts") / args.config
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_path = args.model or config.get('model', {}).get('path')
    else:
        print(f"Warning: Config file not found: {config_path}")
        print("Using command-line arguments only")
        model_path = args.model
    
    if model_path is None:
        print("Error: Model path must be specified either in config or via --model")
        sys.exit(1)
    
    try:
        # Load input data
        print(f"Loading input from: {input_path}")
        input_data = load_input_files(input_path)
        print(f"Loaded {len(input_data)} sample(s)")
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(input_data)
        
        # Initialize predictor
        predictor = Predictor(
            artifacts_dir=args.artifacts_dir,
            models_dir=args.models_dir
        )
        predictor.initialize(model_path=model_path)
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = predictor.predict_with_labels(df)
        
        # Output results
        if args.output:
            save_predictions(predictions, args.output)
        else:
            # Print to stdout
            print("\nPredictions:")
            print(json.dumps(predictions, indent=2, ensure_ascii=False))
        
        print("\nPrediction completed successfully")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

