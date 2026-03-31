# src/model_registry.py
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib

class ModelRegistry:
    def __init__(self, registry_file="models/model_registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load the registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_registry(self):
        """Save registry to file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def register_model(self, model_name, model_type, metrics=None, params=None):
        """Register a new model in the registry"""
        if metrics is None:
            metrics = {}
        if params is None:
            params = {}
        
        self.registry[model_name] = {
            'type': model_type,
            'metrics': metrics,
            'params': params,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self._save_registry()
        print(f"✅ Registered model: {model_name}")
    
    def update_model_metrics(self, model_name, metrics):
        """Update metrics for an existing model"""
        if model_name in self.registry:
            self.registry[model_name]['metrics'].update(metrics)
            self.registry[model_name]['updated_at'] = datetime.now().isoformat()
            self._save_registry()
            print(f"✅ Updated metrics for: {model_name}")
        else:
            print(f"❌ Model {model_name} not found in registry")
    
    def get_model(self, model_name):
        """Get model information"""
        return self.registry.get(model_name)
    
    def get_best_model(self, metric='f1_score', model_type=None):
        """Get the best model based on a metric"""
        best_model = None
        best_score = -1
        
        for name, info in self.registry.items():
            if model_type and info['type'] != model_type:
                continue
            
            score = info['metrics'].get(metric, -1)
            if score > best_score:
                best_score = score
                best_model = name
        
        return best_model, best_score
    
    def list_models(self, model_type=None):
        """List all models, optionally filtered by type"""
        if model_type:
            return {k: v for k, v in self.registry.items() if v['type'] == model_type}
        return self.registry
    
    def export_to_dataframe(self):
        """Export registry to pandas DataFrame"""
        data = []
        
        for name, info in self.registry.items():
            row = {
                'model_name': name,
                'type': info['type'],
                'created_at': info['created_at'],
                'updated_at': info['updated_at']
            }
            
            # Add metrics
            for metric_name, metric_value in info['metrics'].items():
                row[metric_name] = metric_value
            
            # Add some params
            for param_name, param_value in info['params'].items():
                if isinstance(param_value, (str, int, float, bool)):
                    row[param_name] = param_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_model_summary(self, output_file="models/model_summary.csv"):
        """Save model summary to CSV"""
        df = self.export_to_dataframe()
        if not df.empty:
            df.to_csv(output_file, index=False)
            print(f"✅ Saved model summary to: {output_file}")
    
    def cleanup_old_models(self, max_age_days=30):
        """Remove old models from registry"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed = []
        
        for name, info in list(self.registry.items()):
            created_at = datetime.fromisoformat(info['created_at'])
            if created_at < cutoff_date:
                del self.registry[name]
                removed.append(name)
        
        if removed:
            self._save_registry()
            print(f"✅ Removed {len(removed)} old models: {', '.join(removed)}")
    
    def get_model_recommendation(self):
        """Get model recommendation based on metrics"""
        models_df = self.export_to_dataframe()
        
        if models_df.empty:
            return None
        
        # Calculate composite score
        if 'accuracy' in models_df.columns and 'f1_score' in models_df.columns:
            models_df['composite_score'] = (
                models_df['accuracy'] * 0.4 + 
                models_df['f1_score'] * 0.6
            )
            best_model = models_df.loc[models_df['composite_score'].idxmax()]
            
            return {
                'model_name': best_model['model_name'],
                'type': best_model['type'],
                'accuracy': best_model.get('accuracy', 0),
                'f1_score': best_model.get('f1_score', 0),
                'composite_score': best_model.get('composite_score', 0)
            }
        
        return None

if __name__ == "__main__":
    registry = ModelRegistry()
    
    # Test registration
    registry.register_model(
        "test_model",
        "logistic_regression",
        metrics={"accuracy": 0.95, "f1_score": 0.94},
        params={"C": 1.0, "penalty": "l2"}
    )
    
    # List models
    print("\nRegistered models:")
    for name, info in registry.list_models().items():
        print(f"- {name}: {info['type']}")
    
    # Export to CSV
    registry.save_model_summary()