# src/train.py - ENHANCED VERSION
import torch
import nltk
import joblib
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_auc_score, f1_score,
                           precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from preprocessing import TextPreprocessor
from model_registry import ModelRegistry
from data_augmentation import DataAugmenter
from pathlib import Path

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class FakeNewsClassifier:
    def __init__(self, model_dir="models", random_state=42, use_augmentation=True):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.data_loader = DataLoader(random_state=random_state)
        self.registry = ModelRegistry(self.model_dir / "model_registry.json")
        self.use_augmentation = use_augmentation
        
        if use_augmentation:
            self.augmenter = DataAugmenter(method='synonym')
    
    def augment_training_data(self, df, text_col='text', label_col='label'):
        """Augment training data if enabled"""
        if not self.use_augmentation:
            return df
        
        print("\n🔧 Augmenting training data...")
        return self.augmenter.augment_dataset(df, text_col, label_col, balance_classes=True)
    
    def create_feature_pipelines(self):
        """Create multiple feature extraction pipelines"""
        pipelines = {
            'tfidf_ngram': Pipeline([
                ('vectorizer', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words='english',
                    min_df=3,
                    max_df=0.95,
                    dtype=np.float32
                ))
            ]),
            
            'tfidf_char': Pipeline([
                ('vectorizer', TfidfVectorizer(
                    analyzer='char',
                    ngram_range=(3, 5),
                    max_features=5000,
                    dtype=np.float32
                ))
            ]),
            
            'bow_ngram': Pipeline([
                ('vectorizer', CountVectorizer(
                    max_features=8000,
                    ngram_range=(1, 2),
                    stop_words='english',
                    min_df=3,
                    max_df=0.95,
                    dtype=np.float32
                ))
            ])
        }
        return pipelines
    
    def create_models(self):
        """Create multiple model configurations"""
        models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=self.random_state,
                    solver='liblinear',
                    n_jobs=-1
                ),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                }
            },
        }
        
        return models
    
    def train_with_cross_validation(self, X_train, y_train, cv_folds=5):
        """Train multiple models with cross-validation"""
        results = {}
        
        # Create feature pipelines
        feature_pipelines = self.create_feature_pipelines()
        models = self.create_models()
        
        best_score = 0
        best_model_info = None
        
        for feat_name, feat_pipeline in feature_pipelines.items():
            print(f"\n{'='*60}")
            print(f"Feature Pipeline: {feat_name}")
            print(f"{'='*60}")
            
            # Transform features
            print(f"Transforming features...")
            try:
                X_train_feat = feat_pipeline.fit_transform(X_train)
                print(f"  Feature matrix shape: {X_train_feat.shape}")
                print(f"  Memory: {X_train_feat.data.nbytes / (1024**2):.1f} MB")
            except MemoryError:
                print(f"  Memory error! Skipping {feat_name} pipeline")
                continue
            
            for model_name, model_config in models.items():
                print(f"\nTraining {model_name}...")
                
                # Skip heavy models for char features
                if 'char' in feat_name and model_name in ['random_forest', 'gradient_boosting']:
                    print(f"  Skipping {model_name} for char features (too heavy)")
                    continue
                
                try:
                    # Use Stratified K-Fold
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    
                    # Grid search
                    grid_search = GridSearchCV(
                        model_config['model'],
                        model_config['params'],
                        cv=cv,
                        scoring='f1_weighted',
                        n_jobs=-1,
                        verbose=0,
                        error_score='raise'
                    )
                    
                    start_time = time()
                    grid_search.fit(X_train_feat, y_train)
                    train_time = time() - start_time
                    
                    # Cross-validation scores
                    cv_scores = cross_val_score(
                        grid_search.best_estimator_,
                        X_train_feat,
                        y_train,
                        cv=cv_folds,
                        scoring='f1_weighted',
                        n_jobs=-1
                    )
                    
                    # Store results
                    results[f"{feat_name}_{model_name}"] = {
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'train_time': train_time,
                        'model': grid_search.best_estimator_,
                        'feature_pipeline': feat_pipeline,
                        'feature_name': feat_name,
                        'model_name': model_name
                    }
                    
                    print(f"  Best CV Score: {grid_search.best_score_:.4f}")
                    print(f"  Best Params: {grid_search.best_params_}")
                    print(f"  Training Time: {train_time:.2f}s")
                    
                    # Update best model
                    if grid_search.best_score_ > best_score:
                        best_score = grid_search.best_score_
                        best_model_info = results[f"{feat_name}_{model_name}"]
                        
                except MemoryError as e:
                    print(f"  Memory error training {model_name}: {e}")
                    continue
                except Exception as e:
                    print(f"  Error training {model_name}: {e}")
                    continue
        
        if not results:
            print("\n❌ No models were trained successfully!")
            return None, None
        
        return results, best_model_info
    
    def train_single_model(self, X_train, y_train, model_name='logistic_regression'):
        """Train a single model efficiently"""
        print(f"\nTraining single model: {model_name}")
        
        # Use tfidf features
        vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words='english',
            dtype=np.float32
        )
        
        X_train_feat = vectorizer.fit_transform(X_train)
        print(f"Feature matrix shape: {X_train_feat.shape}")
        
        # Select model
        if model_name == 'logistic_regression':
            model = LogisticRegression(
                C=10,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1
            )
        elif model_name == 'naive_bayes':
            model = MultinomialNB(alpha=0.1)
        elif model_name == 'svm_linear':
            model = LinearSVC(
                C=1,
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=2000,
                dual=False
            )
        else:
            model = LogisticRegression()
        
        # Train
        start_time = time()
        model.fit(X_train_feat, y_train)
        train_time = time() - start_time
        
        print(f"Training completed in {train_time:.2f}s")
        
        # Create pipeline
        from sklearn.pipeline import make_pipeline
        pipeline = make_pipeline(vectorizer, model)
        
        return model, vectorizer, pipeline, train_time
    
    def evaluate_model(self, model_info, X_test, y_test):
        """Comprehensive model evaluation"""
        model = model_info.get('model')
        pipeline = model_info.get('pipeline')
        feature_pipeline = model_info.get('feature_pipeline')
        
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION")
        print(f"{'='*60}")
        
        # Make predictions
        if pipeline:
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
        elif feature_pipeline:
            X_test_feat = feature_pipeline.transform(X_test)
            y_pred = model.predict(X_test_feat)
            y_pred_proba = model.predict_proba(X_test_feat)[:, 1] if hasattr(model, 'predict_proba') else None
        else:
            # Use default vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=8000)
            X_train_feat = vectorizer.fit_transform(X_test)  # Not ideal, but works
            X_test_feat = vectorizer.transform(X_test)
            y_pred = model.predict(X_test_feat)
            y_pred_proba = model.predict_proba(X_test_feat)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        if roc_auc:
            print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🎯 Confusion Matrix:")
        print(f"           Predicted")
        print(f"           Real  Fake")
        print(f"Actual Real  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Fake  {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
        
        return metrics, y_pred, y_pred_proba
    
    def save_models(self, model_info, model_type='traditional'):
        """Save trained models and register them"""
        model_name = model_info.get('model_name', 'unnamed_model')
        
        # Save main model
        model_path = self.model_dir / f"{model_name}_model.pkl"
        joblib.dump(model_info['model'], model_path)
        print(f"\n✅ Model saved: {model_path}")
        
        # Save pipeline if exists
        if 'pipeline' in model_info:
            pipeline_path = self.model_dir / f"{model_name}_pipeline.pkl"
            joblib.dump(model_info['pipeline'], pipeline_path)
            print(f"✅ Pipeline saved: {pipeline_path}")
        
        # Save vectorizer if exists
        if 'vectorizer' in model_info:
            vectorizer_path = self.model_dir / f"{model_name}_vectorizer.pkl"
            joblib.dump(model_info['vectorizer'], vectorizer_path)
            print(f"✅ Vectorizer saved: {vectorizer_path}")
        
        # Save feature pipeline if exists
        if 'feature_pipeline' in model_info:
            feature_pipeline_path = self.model_dir / f"{model_name}_feature_pipeline.pkl"
            joblib.dump(model_info['feature_pipeline'], feature_pipeline_path)
        
        # Register model
        metrics = model_info.get('metrics', {})
        params = model_info.get('best_params', {})
        
        self.registry.register_model(
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            params=params
        )
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'metrics': metrics,
            'params': params,
            'timestamp': pd.Timestamp.now(),
            'random_state': self.random_state
        }
        metadata_path = self.model_dir / f"{model_name}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        return model_path
    
    def run_training_pipeline(self, fast_mode=True, sample_size=None, cv_folds=5):
        """Complete training pipeline"""
        print(f"\n{'='*60}")
        print(f"FAKE NEWS DETECTION TRAINING")
        print(f"{'='*60}")
        
        # Load data
        print("\n📂 Loading datasets...")
        train_df, val_df, test_df = self.data_loader.load_all_datasets()
        
        # Convert labels
        train_df['label'] = train_df['label'].astype(int)
        val_df['label'] = val_df['label'].astype(int)
        test_df['label'] = test_df['label'].astype(int)
        
        # Sample if needed
        if sample_size and sample_size < len(train_df):
            print(f"\n📊 Sampling {sample_size} instances...")
            train_df = train_df.sample(n=sample_size, random_state=self.random_state)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Training samples:   {len(train_df):,}")
        print(f"   Validation samples: {len(val_df):,}")
        print(f"   Test samples:       {len(test_df):,}")
        print(f"   Fake/Real ratio:    {train_df['label'].mean():.2%} fake")
        
        # Augment training data
        if self.use_augmentation:
            train_df = self.augment_training_data(train_df)
        
        # Preprocess text
        print("\n🔧 Preprocessing text...")
        X_train = self.preprocessor.batch_preprocess(train_df['text'])
        y_train = train_df['label'].values
        
        X_val = self.preprocessor.batch_preprocess(val_df['text'])
        y_val = val_df['label'].values
        
        X_test = self.preprocessor.batch_preprocess(test_df['text'])
        y_test = test_df['label'].values
        
        # Training
        if fast_mode:
            print(f"\n{'='*60}")
            print(f"🚀 FAST MODE: Training single model")
            print(f"{'='*60}")
            
            # Train logistic regression
            model, vectorizer, pipeline, train_time = self.train_single_model(
                X_train, y_train, model_name='logistic_regression'
            )
            
            model_name = "logistic_regression_fast"
            
            # Evaluate on validation set
            print(f"\n📊 Validating model...")
            model_info = {
                'model': model,
                'vectorizer': vectorizer,
                'pipeline': pipeline,
                'model_name': model_name
            }
            
            val_metrics, _, _ = self.evaluate_model(model_info, X_val, y_val)
            
            # Update model info with metrics
            model_info['metrics'] = val_metrics
            model_info['train_time'] = train_time
            
            # Save model
            self.save_models(model_info)
            
            # Final test evaluation
            print(f"\n{'='*60}")
            print(f"🎯 FINAL TEST SET EVALUATION")
            print(f"{'='*60}")
            
            test_metrics, _, _ = self.evaluate_model(model_info, X_test, y_test)
            
            traditional_results = model_info
            
        else:
            print(f"\n{'='*60}")
            print(f"🔍 COMPREHENSIVE MODE: Training multiple models")
            print(f"{'='*60}")
            
            # Train multiple models
            results, best_model_info = self.train_with_cross_validation(X_train, y_train, cv_folds)
            
            if not results:
                print("❌ Failed to train models. Switching to fast mode...")
                return self.run_training_pipeline(fast_mode=True, sample_size=sample_size)
            
            # Evaluate best model on validation
            print(f"\n📊 Evaluating best model ({best_model_info['model_name']})...")
            val_metrics, _, _ = self.evaluate_model(best_model_info, X_val, y_val)
            best_model_info['metrics'] = val_metrics
            
            # Save best model
            self.save_models(best_model_info)
            
            # Final test evaluation
            print(f"\n{'='*60}")
            print(f"🎯 FINAL TEST SET EVALUATION")
            print(f"{'='*60}")
            
            test_metrics, _, _ = self.evaluate_model(best_model_info, X_test, y_test)
            
            traditional_results = best_model_info
        
        # Update registry with test metrics
        if 'model_name' in traditional_results:
            self.registry.update_model_metrics(
                traditional_results['model_name'],
                traditional_results.get('metrics', {})
            )
        
        # Export model summary
        self.registry.save_model_summary()
        
        # Get model recommendation
        recommendation = self.registry.get_model_recommendation()
        if recommendation:
            print(f"\n{'='*60}")
            print(f"🏆 RECOMMENDED MODEL")
            print(f"{'='*60}")
            print(f"   Model:     {recommendation['model_name']}")
            print(f"   Type:      {recommendation['type']}")
            print(f"   Accuracy:  {recommendation['accuracy']:.4f}")
            print(f"   F1 Score:  {recommendation['f1_score']:.4f}")
        
        print(f"\n✅ Training pipeline completed successfully!")
        print(f"📁 Models saved in: {self.model_dir.absolute()}")
        
        return {'traditional': traditional_results}


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fake news detection models')
    parser.add_argument('--fast', action='store_true', default=False, 
                       help='Use fast mode (single model)')
    parser.add_argument('--sample', type=int, default=None, 
                       help='Sample size for training')
    parser.add_argument('--cv', type=int, default=5, 
                       help='Cross-validation folds')
    parser.add_argument('--no-augment', action='store_true', 
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = FakeNewsClassifier(
        random_state=42,
        use_augmentation=not args.no_augment
    )
    
    # Run training
    results = classifier.run_training_pipeline(
        fast_mode=args.fast,
        sample_size=args.sample,
        cv_folds=args.cv
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    
    if 'traditional' in results:
        model_info = results['traditional']
        print(f"Model: {model_info.get('model_name', 'Unknown')}")
        
        metrics = model_info.get('metrics', {})
        if metrics:
            print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
    
    print(f"\n📊 Model registry saved: models/model_registry.json")
    print(f"📊 Model summary saved: models/model_summary.csv")