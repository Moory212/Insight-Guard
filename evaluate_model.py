# evaluate_models.py
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.append('src')

from data_loader import DataLoader
from preprocessing import TextPreprocessor
from model_loader import ModelLoader

class ModelEvaluator:
    def __init__(self, model_dir="models", data_dir="data/raw"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.model_loader = ModelLoader(model_dir)
        self.preprocessor = TextPreprocessor()
        
        # Create output directory
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_test_data(self, sample_size=None):
        """Load and prepare test data"""
        print("Loading test data...")
        
        # Load data
        data_loader = DataLoader(random_state=42)
        _, _, test_df = data_loader.load_all_datasets()
        
        # Sample if needed
        if sample_size and sample_size < len(test_df):
            test_df = test_df.sample(n=sample_size, random_state=42)
        
        # Preprocess
        X_test = self.preprocessor.batch_preprocess(test_df['text'], show_progress=True)
        y_test = test_df['label'].astype(int).values
        
        print(f"Test data: {len(X_test)} samples")
        print(f"Class distribution: {np.bincount(y_test)}")
        
        return X_test, y_test, test_df
    
    def evaluate_single_model(self, model_name, X_test, y_test):
        """Evaluate a single model"""
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model_info = self.model_loader.get_model_info(model_name)
        if not model_info:
            print(f"Model {model_name} not found!")
            return None
        
        results = {
            'model_name': model_name,
            'model_type': model_info['type']
        }
        
        try:
            if model_info['type'] == 'BERT':
                # Load BERT model
                model, tokenizer = self.model_loader.load_bert_model(model_name)
                if not model:
                    print(f"Failed to load BERT model: {model_name}")
                    return None
                
                # Prepare BERT trainer for predictions
                from bert_model import BertTrainer
                trainer = BertTrainer(batch_size=32)
                
                # Make predictions
                predictions, probabilities = trainer.predict(model, X_test)
                
                y_pred = np.array(predictions)
                y_proba = np.array(probabilities)[:, 1] if len(probabilities[0]) > 1 else None
                
            else:
                # Load traditional model
                model_data, model_format = self.model_loader.load_traditional_model(model_name)
                if not model_data:
                    print(f"Failed to load traditional model: {model_name}")
                    return None
                
                # Make predictions based on format
                if model_format == "pipeline":
                    y_pred = model_data.predict(X_test)
                    y_proba = model_data.predict_proba(X_test)[:, 1] if hasattr(model_data, 'predict_proba') else None
                elif model_format == "model+vectorizer":
                    model, vectorizer = model_data
                    X_test_vec = vectorizer.transform(X_test)
                    y_pred = model.predict(X_test_vec)
                    y_proba = model.predict_proba(X_test_vec)[:, 1] if hasattr(model, 'predict_proba') else None
                else:
                    y_pred = model_data.predict(X_test)
                    y_proba = None
            
            # Calculate metrics
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['f1_score'] = f1_score(y_test, y_pred, average='weighted')
            results['precision'] = precision_score(y_test, y_pred, average='weighted')
            results['recall'] = recall_score(y_test, y_pred, average='weighted')
            
            if y_proba is not None:
                results['roc_auc'] = roc_auc_score(y_test, y_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm
            
            # Detailed report
            report = classification_report(y_test, y_pred, output_dict=True)
            results['classification_report'] = report
            
            # Print results
            print(f"\n📊 Performance Metrics:")
            print(f"   Accuracy:  {results['accuracy']:.4f}")
            print(f"   F1 Score:  {results['f1_score']:.4f}")
            print(f"   Precision: {results['precision']:.4f}")
            print(f"   Recall:    {results['recall']:.4f}")
            if 'roc_auc' in results:
                print(f"   ROC-AUC:   {results['roc_auc']:.4f}")
            
            print(f"\n📋 Confusion Matrix:")
            print(f"           Predicted")
            print(f"           Real  Fake")
            print(f"Actual Real  {cm[0,0]:4d}  {cm[0,1]:4d}")
            print(f"       Fake  {cm[1,0]:4d}  {cm[1,1]:4d}")
            
            # Save confusion matrix plot
            self.plot_confusion_matrix(cm, model_name)
            
            return results
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        plot_path = self.output_dir / f"{model_name}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        print(f"📊 Saved confusion matrix: {plot_path}")
    
    def evaluate_all_models(self, sample_size=None):
        """Evaluate all available models"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE MODEL EVALUATION")
        print(f"{'='*60}")
        
        # Load test data
        X_test, y_test, _ = self.load_test_data(sample_size)
        
        # Get all models
        models = self.model_loader.discover_models()
        
        if not models:
            print("❌ No models found to evaluate!")
            return
        
        print(f"\nFound {len(models)} models to evaluate:")
        for model in models:
            print(f"  - {model['name']} ({model['type']})")
        
        # Evaluate each model
        all_results = []
        failed_models = []
        
        for model_info in models:
            model_name = model_info['name']
            
            results = self.evaluate_single_model(model_name, X_test, y_test)
            
            if results:
                all_results.append(results)
            else:
                failed_models.append(model_name)
        
        # Save comprehensive results
        if all_results:
            self.save_evaluation_results(all_results, failed_models)
        
        # Print summary
        self.print_evaluation_summary(all_results, failed_models)
        
        return all_results
    
    def save_evaluation_results(self, results, failed_models):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to CSV
        metrics_data = []
        for result in results:
            row = {
                'model_name': result['model_name'],
                'model_type': result['model_type'],
                'accuracy': result.get('accuracy', 0),
                'f1_score': result.get('f1_score', 0),
                'precision': result.get('precision', 0),
                'recall': result.get('recall', 0),
                'roc_auc': result.get('roc_auc', None)
            }
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('f1_score', ascending=False)
        
        csv_path = self.output_dir / f"model_evaluation_{timestamp}.csv"
        metrics_df.to_csv(csv_path, index=False)
        
        # Save detailed results
        detailed_path = self.output_dir / f"detailed_results_{timestamp}.json"
        
        import json
        serializable_results = []
        for result in results:
            serializable = {
                'model_name': result['model_name'],
                'model_type': result['model_type'],
                'metrics': {k: v for k, v in result.items() 
                           if k not in ['confusion_matrix', 'classification_report']},
                'confusion_matrix': result.get('confusion_matrix', []).tolist(),
                'classification_report': result.get('classification_report', {})
            }
            serializable_results.append(serializable)
        
        with open(detailed_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save failed models
        if failed_models:
            failed_path = self.output_dir / f"failed_models_{timestamp}.txt"
            with open(failed_path, 'w') as f:
                f.write("\n".join(failed_models))
        
        print(f"\n💾 Saved evaluation results:")
        print(f"   Metrics CSV: {csv_path}")
        print(f"   Detailed JSON: {detailed_path}")
        if failed_models:
            print(f"   Failed models: {failed_path}")
    
    def print_evaluation_summary(self, results, failed_models):
        """Print evaluation summary"""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n📊 Models evaluated successfully: {len(results)}")
        print(f"❌ Models failed: {len(failed_models)}")
        
        if failed_models:
            print(f"Failed models: {', '.join(failed_models)}")
        
        if results:
            # Find best model by F1 score
            best_model = max(results, key=lambda x: x.get('f1_score', 0))
            
            print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
            print(f"   Type:      {best_model['model_type']}")
            print(f"   F1 Score:  {best_model['f1_score']:.4f}")
            print(f"   Accuracy:  {best_model['accuracy']:.4f}")
            print(f"   Precision: {best_model['precision']:.4f}")
            print(f"   Recall:    {best_model['recall']:.4f}")
            
            if 'roc_auc' in best_model:
                print(f"   ROC-AUC:   {best_model['roc_auc']:.4f}")
            
            # Print ranking
            print(f"\n📈 Model Ranking (by F1 Score):")
            sorted_results = sorted(results, key=lambda x: x.get('f1_score', 0), reverse=True)
            
            for i, result in enumerate(sorted_results[:10], 1):
                print(f"   {i:2d}. {result['model_name'][:30]:30} "
                      f"{result['model_type'][:15]:15} "
                      f"F1: {result['f1_score']:.4f} "
                      f"Acc: {result['accuracy']:.4f}")
    
    def create_comparison_plots(self, results):
        """Create comparison plots"""
        if not results:
            return
        
        # Create metrics comparison bar plot
        metrics_df = pd.DataFrame([
            {
                'model': r['model_name'],
                'type': r['model_type'],
                'accuracy': r.get('accuracy', 0),
                'f1_score': r.get('f1_score', 0),
                'precision': r.get('precision', 0),
                'recall': r.get('recall', 0)
            }
            for r in results
        ])
        
        # Sort by F1 score
        metrics_df = metrics_df.sort_values('f1_score', ascending=False)
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        
        x = range(len(metrics_df))
        width = 0.2
        
        plt.bar([i - 1.5*width for i in x], metrics_df['accuracy'], width, label='Accuracy')
        plt.bar([i - 0.5*width for i in x], metrics_df['f1_score'], width, label='F1 Score')
        plt.bar([i + 0.5*width for i in x], metrics_df['precision'], width, label='Precision')
        plt.bar([i + 1.5*width for i in x], metrics_df['recall'], width, label='Recall')
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, metrics_df['model'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plot_path = self.output_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        print(f"📊 Saved comparison plot: {plot_path}")
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        
        colors = {'Traditional ML': 'blue', 'BERT': 'red'}
        
        for _, row in metrics_df.iterrows():
            plt.scatter(
                row['accuracy'], 
                row['f1_score'],
                c=colors.get(row['type'], 'gray'),
                s=100,
                alpha=0.6,
                label=row['type'] if row['type'] not in plt.gca().get_legend_handles_labels()[1] else ""
            )
            plt.annotate(
                row['model'][:15],
                (row['accuracy'], row['f1_score']),
                fontsize=8,
                alpha=0.7
            )
        
        plt.xlabel('Accuracy')
        plt.ylabel('F1 Score')
        plt.title('Accuracy vs F1 Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        scatter_path = self.output_dir / "accuracy_vs_f1.png"
        plt.savefig(scatter_path, dpi=100)
        plt.close()
        
        print(f"📊 Saved scatter plot: {scatter_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate fake news detection models')
    parser.add_argument('--model', type=str, help='Evaluate specific model')
    parser.add_argument('--all', action='store_true', help='Evaluate all models')
    parser.add_argument('--sample', type=int, default=5000, 
                       help='Sample size for evaluation')
    parser.add_argument('--plots', action='store_true', 
                       help='Generate comparison plots')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    
    if args.model:
        # Evaluate single model
        X_test, y_test, _ = evaluator.load_test_data(args.sample)
        results = evaluator.evaluate_single_model(args.model, X_test, y_test)
        
        if results and args.plots:
            evaluator.create_comparison_plots([results])
    
    elif args.all:
        # Evaluate all models
        results = evaluator.evaluate_all_models(args.sample)
        
        if results and args.plots:
            evaluator.create_comparison_plots(results)
    
    else:
        print("\nModel Evaluation Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  python evaluate_models.py --all")
        print("  python evaluate_models.py --model logistic_regression_fast")
        print("  python evaluate_models.py --all --plots --sample 10000")

if __name__ == "__main__":
    main()