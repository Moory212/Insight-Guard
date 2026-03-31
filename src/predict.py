# predict.py - ENHANCED VERSION WITH BERT
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing import TextPreprocessor

class FakeNewsPredictor:
    def __init__(self, model_dir="models", model_type="auto"):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing trained models
            model_type: "auto" (detect automatically), "traditional", or "bert"
        """
        self.model_dir = Path(model_dir)
        self.preprocessor = TextPreprocessor()
        self.model_type = model_type
        
        # Try to load the model
        self.model, self.pipeline, self.bert_trainer = self.load_model()
        
    def load_model(self):
        """Load the appropriate model based on availability or specified type"""
        # Check for BERT model if requested or auto
        bert_model_path = self.model_dir / "bert_model.pt"
        bert_tokenizer_path = self.model_dir / "bert_model_tokenizer"
        
        # Check for traditional model
        traditional_model_files = list(self.model_dir.glob("*_model.pkl"))
        
        # Determine which model to load
        load_bert = False
        load_traditional = False
        
        if self.model_type == "bert":
            if bert_model_path.exists():
                load_bert = True
            else:
                raise FileNotFoundError("BERT model not found. Please train a BERT model first.")
        elif self.model_type == "traditional":
            if traditional_model_files:
                load_traditional = True
            else:
                raise FileNotFoundError("Traditional model not found. Please train a traditional model first.")
        else:  # auto
            if bert_model_path.exists() and traditional_model_files:
                print("Both BERT and traditional models found. Loading BERT by default.")
                load_bert = True
            elif bert_model_path.exists():
                print("Loading BERT model...")
                load_bert = True
            elif traditional_model_files:
                print("Loading traditional model...")
                load_traditional = True
            else:
                raise FileNotFoundError("No models found. Please train a model first.")
        
        # Load BERT model
        if load_bert:
            try:
                from src.bert_model import BertTrainer
                
                # Load BERT model
                checkpoint = torch.load(bert_model_path, map_location='cpu')
                bert_trainer = BertTrainer()
                
                # Reconstruct model
                from src.bert_model import BertFakeNewsClassifier
                model = BertFakeNewsClassifier()
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                print(f"Loaded BERT model: {bert_model_path}")
                return model, None, bert_trainer
                
            except ImportError as e:
                print(f"Error loading BERT model: {e}")
                print("Please install required packages: pip install torch transformers")
                raise
            except Exception as e:
                print(f"Error loading BERT model: {e}")
                raise
        
        # Load traditional model
        elif load_traditional:
            # Get the latest model by timestamp
            latest_model = max(traditional_model_files, key=lambda x: x.stat().st_mtime)
            model_name = latest_model.stem.replace("_model", "")
            
            print(f"Loading traditional model: {model_name}")
            
            # Load model
            model = joblib.load(latest_model)
            
            # Try to load pipeline
            pipeline_path = self.model_dir / f"{model_name}_pipeline.pkl"
            pipeline = joblib.load(pipeline_path) if pipeline_path.exists() else None
            
            return model, pipeline, None
        
        else:
            raise ValueError("Unable to load any model")
    
    def predict_single(self, text, return_probability=True):
        """Predict single text"""
        if self.bert_trainer is not None:
            # BERT prediction
            return self._predict_bert(text, return_probability)
        else:
            # Traditional model prediction
            return self._predict_traditional(text, return_probability)
    
    def _predict_bert(self, text, return_probability=True):
        """Make prediction using BERT model"""
        # Tokenize and prepare input
        tokenizer = self.bert_trainer.tokenizer
        device = self.bert_trainer.device
        
        # Tokenize the text
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            
        if return_probability:
            return {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': float(torch.max(probs)),
                'fake_probability': float(probs[0][1]),
                'real_probability': float(probs[0][0])
            }
        else:
            return {'prediction': 'Fake' if prediction == 1 else 'Real'}
    
    def _predict_traditional(self, text, return_probability=True):
        """Make prediction using traditional model"""
        # Preprocess
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Transform features if pipeline exists
        if self.pipeline:
            features = self.pipeline.transform([cleaned_text])
        else:
            # Fallback to simple vectorization
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()
            features = vectorizer.fit_transform([cleaned_text])
        
        # Predict
        if return_probability and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
            prediction = self.model.predict(features)[0]
            return {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': float(max(proba)),
                'fake_probability': float(proba[1]),
                'real_probability': float(proba[0])
            }
        else:
            prediction = self.model.predict(features)[0]
            return {'prediction': 'Fake' if prediction == 1 else 'Real'}
    
    def predict_batch(self, texts, return_probabilities=True):
        """Predict multiple texts"""
        results = []
        
        for text in texts:
            try:
                result = self.predict_single(text, return_probabilities)
                result['text'] = text[:100] + "..." if len(text) > 100 else text
                results.append(result)
            except Exception as e:
                print(f"Error predicting text: {str(e)[:100]}")
                results.append({
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'prediction': 'Error',
                    'error': str(e)[:100]
                })
        
        return pd.DataFrame(results)
    
    def predict_batch_bert(self, texts, return_probabilities=True):
        """Predict multiple texts using BERT (more efficient batch processing)"""
        if self.bert_trainer is None:
            raise ValueError("BERT model not loaded. Cannot use batch BERT prediction.")
        
        tokenizer = self.bert_trainer.tokenizer
        device = self.bert_trainer.device
        batch_size = self.bert_trainer.batch_size
        
        predictions = []
        probabilities = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoding = tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # Format results
        results = []
        for idx, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
            if return_probabilities:
                results.append({
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'prediction': 'Fake' if pred == 1 else 'Real',
                    'confidence': float(max(prob)),
                    'fake_probability': float(prob[1]),
                    'real_probability': float(prob[0])
                })
            else:
                results.append({
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'prediction': 'Fake' if pred == 1 else 'Real'
                })
        
        return pd.DataFrame(results)
    
    def extract_explanations(self, text, top_features=10):
        """Extract feature importance for prediction explanation (traditional models only)"""
        if self.bert_trainer is not None:
            print("Note: Feature explanations are not available for BERT models.")
            return []
        
        cleaned_text = self.preprocessor.clean_text(text)
        
        if self.pipeline and hasattr(self.model, 'coef_'):
            # Get feature names
            if hasattr(self.pipeline.named_steps['vectorizer'], 'get_feature_names_out'):
                feature_names = self.pipeline.named_steps['vectorizer'].get_feature_names_out()
            else:
                feature_names = self.pipeline.named_steps['vectorizer'].get_feature_names()
            
            # Transform text
            features = self.pipeline.transform([cleaned_text])
            
            # Get coefficients
            coef = self.model.coef_[0]
            
            # Get feature indices present in text
            feature_indices = features.nonzero()[1]
            
            # Get top contributing features
            contributions = [(feature_names[i], coef[i], features[0, i]) 
                           for i in feature_indices]
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return contributions[:top_features]
        
        return []
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.bert_trainer is not None:
            return {
                'model_type': 'BERT',
                'model_architecture': 'bert-base-uncased',
                'device': str(self.bert_trainer.device),
                'max_length': self.bert_trainer.max_length,
                'batch_size': self.bert_trainer.batch_size
            }
        else:
            model_name = type(self.model).__name__
            has_pipeline = self.pipeline is not None
            
            info = {
                'model_type': 'Traditional',
                'model_name': model_name,
                'has_pipeline': has_pipeline,
                'has_probability': hasattr(self.model, 'predict_proba')
            }
            
            if has_pipeline:
                pipeline_steps = list(self.pipeline.named_steps.keys())
                info['pipeline_steps'] = pipeline_steps
            
            return info


# Example usage
if __name__ == "__main__":
    # Test the predictor
    print("Testing Fake News Predictor...")
    
    try:
        # Try to load BERT model first, fall back to traditional
        predictor = FakeNewsPredictor(model_type="auto")
        
        # Get model info
        model_info = predictor.get_model_info()
        print(f"\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Test predictions
        test_texts = [
            "Scientists have discovered a new planet that could support human life.",
            "Breaking: Aliens have been spotted in downtown New York! Click to see shocking photos.",
            "The new climate change report shows significant warming trends over the past decade.",
            "You won't believe what this celebrity did! Secret video reveals all.",
            "The government announces new economic stimulus package to support small businesses."
        ]
        
        print(f"\nMaking predictions on test texts:")
        for text in test_texts:
            result = predictor.predict_single(text)
            print(f"\nText: {text[:50]}...")
            print(f"Prediction: {result['prediction']}")
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.2%}")
        
        # Batch prediction
        print(f"\nBatch predictions:")
        batch_results = predictor.predict_batch(test_texts)
        print(batch_results[['text', 'prediction', 'confidence']])
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained a model first.")