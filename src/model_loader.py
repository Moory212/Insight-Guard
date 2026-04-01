
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
        
        # Discover BERT models (.pt files and HuggingFace format)
        bert_models = []
        
        # Check for HuggingFace format
        for model_folder in self.model_dir.glob("*/"):
            if (model_folder / "pytorch_model.bin").exists() or (model_folder / "model.safetensors").exists():
                config_path = model_folder / "config.json"
                if config_path.exists():
                    bert_models.append({
                        'name': model_folder.name,
                        'path': model_folder,
                        'type': 'BERT',
                        'format': 'huggingface'
                    })
        
        # Check for .pt files
        pt_files = list(self.model_dir.glob("*.pt"))
        for pt_file in pt_files:
            if 'bert' in pt_file.stem.lower():
                bert_models.append({
                    'name': pt_file.stem,
                    'path': pt_file,
                    'type': 'BERT',
                    'format': 'pytorch'
                })
        
        # Add BERT models
        for bert_model in bert_models:
            size_mb = bert_model['path'].stat().st_size / (1024 * 1024)
            models.append({
                'name': bert_model['name'],
                'path': str(bert_model['path']),
                'type': 'BERT',
                'size_mb': round(size_mb, 1),
                'format': bert_model['format']
            })
        
        # Discover traditional models (.pkl files)
        pkl_files = list(self.model_dir.glob("*.pkl"))
        model_files = []
        
        for pkl_file in pkl_files:
            filename = pkl_file.stem
            
            # Skip vectorizers and metadata
            if any(x in filename for x in ['vectorizer', 'metadata', 'pipeline']):
                continue
            
            # Check if it's a model file
            if any(x in filename for x in ['model', 'classifier', 'regression', 'svm', 'forest', 'bayes']):
                model_files.append(pkl_file)
        
        for model_file in model_files:
            model_name = model_file.stem
            size_mb = model_file.stat().st_size / (1024 * 1024)
            
            # Check for corresponding files
            vectorizer_exists = (self.model_dir / f"{model_name}_vectorizer.pkl").exists()
            pipeline_exists = (self.model_dir / f"{model_name}_pipeline.pkl").exists()
            
            models.append({
                'name': model_name,
                'path': str(model_file),
                'type': 'Traditional ML',
                'size_mb': round(size_mb, 1),
                'has_vectorizer': vectorizer_exists,
                'has_pipeline': pipeline_exists
            })
        
        return models
    
    def load_bert_model(self, model_name):
        """Load a BERT model from Hugging Face Hub or local path."""
        # If the model name matches our uploaded model, use the Hub ID
        if model_name == "bert_best_model":
            # Use the Hugging Face repository ID
            model_id = "SOKEH/insightguardbert"   # replace with your actual ID
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