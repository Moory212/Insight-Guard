
import torch
import joblib
import pandas as pd
from pathlib import Path
from transformers import BertForSequenceClassification, BertTokenizer
import warnings
warnings.filterwarnings('ignore')

class ModelLoader:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def discover_models(self):
        """Discover all available models in the models directory"""
        models = []

        if not self.model_dir.exists():
            return models

        # Discover BERT models (HuggingFace format)
        # Look for subfolders containing config.json (weights may be missing)
        for model_folder in self.model_dir.glob("*/"):
            config_path = model_folder / "config.json"
            if config_path.exists():
                # This is a valid BERT model folder (weights may be downloaded from Hub)
                models.append({
                    'name': model_folder.name,
                    'path': str(model_folder),
                    'type': 'BERT',
                    'size_mb': 0,  # size unknown, but we can compute if needed
                    'format': 'huggingface'
                })
                continue  # avoid double counting if also .pt file

        # Also check for .pt files (single-file BERT models)
        pt_files = list(self.model_dir.glob("*.pt"))
        for pt_file in pt_files:
            if 'bert' in pt_file.stem.lower():
                size_mb = pt_file.stat().st_size / (1024 * 1024)
                models.append({
                    'name': pt_file.stem,
                    'path': str(pt_file),
                    'type': 'BERT',
                    'size_mb': round(size_mb, 1),
                    'format': 'pytorch'
                })

        # Discover traditional models (pipelines, model+vectorizer)
        # Look for pipeline files and model files
        pipeline_files = list(self.model_dir.glob("*_pipeline.pkl"))
        model_files = list(self.model_dir.glob("*_model.pkl"))

        # Process pipeline files
        for pipeline_file in pipeline_files:
            # Remove '_pipeline' suffix for display name
            model_name = pipeline_file.stem.replace('_pipeline', '')
            size_mb = pipeline_file.stat().st_size / (1024 * 1024)
            models.append({
                'name': model_name,
                'path': str(pipeline_file),
                'type': 'Traditional ML',
                'size_mb': round(size_mb, 1),
                'format': 'pipeline'
            })

        # Process model files (only if no pipeline exists with same base name)
        for model_file in model_files:
            base_name = model_file.stem.replace('_model', '')
            # Check if a pipeline with same base already exists (avoid duplicates)
            if not any(m['name'] == base_name and m.get('format') == 'pipeline' for m in models):
                size_mb = model_file.stat().st_size / (1024 * 1024)
                models.append({
                    'name': base_name,
                    'path': str(model_file),
                    'type': 'Traditional ML',
                    'size_mb': round(size_mb, 1),
                    'format': 'model_only'
                })

        return models
    
    def load_bert_model(self, model_name):
        """Load a BERT model from Hugging Face Hub or local path."""
        # If the model name matches our uploaded model, use the Hub ID
        if model_name == "bert_best_model":
            # Use the Hugging Face repository ID
            model_id = "SOKEH/insightguardbert"
            try:
                from transformers import BertTokenizer, BertForSequenceClassification
                tokenizer = BertTokenizer.from_pretrained(model_id)
                model = BertForSequenceClassification.from_pretrained(model_id, num_labels=2)
                return model, tokenizer
            except Exception as e:
                # Fallback to local files (if present)
                print(f"Failed to load from Hub, trying local: {e}")
                return self._load_local_bert(model_name)
        else:
            # For other models (if any), try local first
            return self._load_local_bert(model_name)

    def _load_local_bert(self, model_name):
        """Helper to load BERT from local models folder."""
        model_path = self.model_dir / model_name
        if not model_path.exists():
            return None, None
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            tokenizer = BertTokenizer.from_pretrained(str(model_path))
            model = BertForSequenceClassification.from_pretrained(str(model_path), num_labels=2)
            return model, tokenizer
        except Exception as e:
            print(f"Error loading local BERT: {e}")
            return None, None
    
    def load_traditional_model(self, model_name):
        """Load traditional ML model with associated components"""
        try:
            # Try to load pipeline first
            pipeline_path = self.model_dir / f"{model_name}_pipeline.pkl"
            if pipeline_path.exists():
                pipeline = joblib.load(pipeline_path)
                print(f"✅ Loaded pipeline: {model_name}")
                return pipeline, "pipeline"
            
            # Try to load model + vectorizer
            model_path = self.model_dir / f"{model_name}_model.pkl"
            vectorizer_path = self.model_dir / f"{model_name}_vectorizer.pkl"
            
            if model_path.exists():
                model = joblib.load(model_path)
                
                if vectorizer_path.exists():
                    vectorizer = joblib.load(vectorizer_path)
                    print(f"✅ Loaded model + vectorizer: {model_name}")
                    return (model, vectorizer), "model+vectorizer"
                else:
                    print(f"✅ Loaded model only: {model_name}")
                    return model, "model_only"
            
            # Try to load the file directly
            direct_path = self.model_dir / f"{model_name}.pkl"
            if direct_path.exists():
                model = joblib.load(direct_path)
                print(f"✅ Loaded model directly: {model_name}")
                return model, "model_only"
            
            print(f"❌ Model {model_name} not found")
            return None, None
            
        except Exception as e:
            print(f"❌ Error loading traditional model {model_name}: {e}")
            return None, None
    
    def get_model_info(self, model_name):
        """Get information about a specific model"""
        models = self.discover_models()
        for model in models:
            if model['name'] == model_name:
                return model
        return None
    
    def list_models(self):
        """List all available models"""
        models = self.discover_models()
        
        print(f"\n{'='*60}")
        print("AVAILABLE MODELS")
        print(f"{'='*60}")
        
        if not models:
            print("No models found")
            return
        
        for model in models:
            print(f"\n📁 {model['name']}")
            print(f"   Type: {model['type']}")
            print(f"   Size: {model['size_mb']} MB")
            
            if model['type'] == 'Traditional ML':
                if model.get('has_pipeline'):
                    print(f"   Format: Full pipeline")
                elif model.get('has_vectorizer'):
                    print(f"   Format: Model + Vectorizer")
                else:
                    print(f"   Format: Model only")
        
        print(f"\nTotal models: {len(models)}")

if __name__ == "__main__":
    loader = ModelLoader()
    loader.list_models()