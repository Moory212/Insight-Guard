# train_bert.py - OPTIMIZED BERT TRAINING WITH AUGMENTATION SUPPORT
import sys
import os
sys.path.append('src')

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import (
    BertTokenizer, BertForSequenceClassification
)
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader as CustomDataLoader
from preprocessing import TextPreprocessor

class FakeNewsDataset(Dataset):
    """Custom Dataset for BERT"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTTrainer:
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.model = None
        self.best_model_state = None
        
    def create_dataset(self, texts, labels):
        """Create a FakeNewsDataset from texts and labels"""
        return FakeNewsDataset(texts, labels, self.tokenizer, self.max_length)
    
    def train(self, train_dataset, val_dataset, epochs=3, batch_size=16, 
              learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1):
        """Train BERT model using custom training loop"""
        print(f"Training BERT for {epochs} epochs...")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
        
        # Load model
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2
        )
        self.model.to(self.device)
        
        # Optimizer
        from transformers import AdamW, get_linear_schedule_with_warmup
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Calculate total steps and warmup steps
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        # Scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training variables
        best_val_f1 = 0
        patience_counter = 0
        patience = 3
        
        print(f"\nTraining configuration:")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        
        # Training loop
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*50}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            train_preds = []
            train_labels = []
            
            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Collect predictions and labels
                with torch.no_grad():
                    preds = torch.argmax(outputs.logits, dim=1)
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
                
                # Print progress
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
            
            # Calculate training metrics
            train_loss = total_loss / len(train_loader)
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            train_acc = accuracy_score(train_labels, train_preds)
            
            print(f"\nTraining - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Accuracy: {train_acc:.4f}")
            
            # Validation phase
            val_f1, val_acc, val_loss = self.evaluate(val_loader)
            print(f"Validation - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  ✅ New best model! F1: {val_f1:.4f}")
                
                # Save the model
                self.save_model("models/bert_best_model")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement for {patience_counter} epoch(s)")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                    # Load best model
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        # Load best model for final evaluation
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        print(f"\nBest validation F1: {best_val_f1:.4f}")
        return self.model
    
    def evaluate(self, data_loader):
        """Evaluate model on a dataset"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Collect predictions
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        accuracy = accuracy_score(all_labels, all_preds)
        
        return f1, accuracy, avg_loss
    
    def save_model(self, save_path="models/bert_model"):
        """Save the trained model and tokenizer"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        print(f"\nModel saved to: {save_path}")
        
        # Also save in .pt format for compatibility
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config.to_dict()
        }, save_path.parent / "bert_model.pt")
        
        return save_path
    
    def predict(self, texts, batch_size=32):
        """Make predictions on new texts"""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        self.model.eval()
        all_preds = []
        all_probs = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)

def load_augmented_data(augment_file):
    """Load augmented examples from CSV file (must have 'text' and 'label' columns)"""
    if not os.path.exists(augment_file):
        print(f"⚠️  Augmentation file {augment_file} not found. Skipping.")
        return None
    try:
        aug_df = pd.read_csv(augment_file)
        # Ensure required columns exist
        if 'text' not in aug_df.columns or 'label' not in aug_df.columns:
            print(f"⚠️  Augmentation file must contain 'text' and 'label' columns. Skipping.")
            return None
        # Ensure label is integer
        aug_df['label'] = aug_df['label'].astype(int)
        print(f"✓ Loaded {len(aug_df)} augmented samples from {augment_file}")
        return aug_df[['text', 'label']]
    except Exception as e:
        print(f"⚠️  Error loading augmentation file: {e}")
        return None

def train_bert_model(sample_size=None, epochs=3, batch_size=16, augment_file=None):
    """Main function to train BERT model with optional augmentation"""
    print("=" * 60)
    print("BERT MODEL TRAINING")
    print("=" * 60)
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    else:
        print("⚠️  No GPU found. Training will be slow on CPU.")
    
    # Load and prepare data
    print("\nLoading data...")
    data_loader = CustomDataLoader(random_state=42)
    preprocessor = TextPreprocessor()
    
    train_df, val_df, test_df = data_loader.load_all_datasets()
    
    # Convert labels
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)
    
    # ===== AUGMENTATION =====
    if augment_file:
        aug_df = load_augmented_data(augment_file)
        if aug_df is not None:
            # Append to training data only
            original_len = len(train_df)
            train_df = pd.concat([train_df, aug_df], ignore_index=True)
            # Remove duplicates (optional)
            train_df = train_df.drop_duplicates(subset=['text']).reset_index(drop=True)
            print(f"  Added {len(train_df) - original_len} unique augmented samples to training set.")
    
    # Sample data if specified
    if sample_size and sample_size < len(train_df):
        print(f"Sampling {sample_size} instances for BERT training...")
        train_df = train_df.sample(n=sample_size, random_state=42)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Preprocess text
    print("\nPreprocessing text...")
    X_train = preprocessor.batch_preprocess(train_df['text'])
    y_train = train_df['label'].values
    
    X_val = preprocessor.batch_preprocess(val_df['text'])
    y_val = val_df['label'].values
    
    X_test = preprocessor.batch_preprocess(test_df['text'])
    y_test = test_df['label'].values
    
    # Initialize BERT trainer
    bert_trainer = BERTTrainer(model_name='bert-base-uncased', max_length=128)
    
    # Create datasets
    train_dataset = bert_trainer.create_dataset(X_train, y_train)
    val_dataset = bert_trainer.create_dataset(X_val, y_val)
    test_dataset = bert_trainer.create_dataset(X_test, y_test)
    
    # Train model
    print("\nStarting BERT training...")
    bert_trainer.train(
        train_dataset,
        val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=2e-5
    )
    
    # Save the final model
    model_path = bert_trainer.save_model("models/bert_final_model")
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)
    test_f1, test_acc, test_loss = bert_trainer.evaluate(test_loader)
    
    print(f"\nTest set results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    # Detailed predictions
    print("\nMaking predictions on test set...")
    test_preds, test_probs = bert_trainer.predict(X_test, batch_size=batch_size*2)
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=['Real', 'Fake']))
    
    cm = confusion_matrix(y_test, test_preds)
    print("\nConfusion Matrix:")
    print(f"           Predicted")
    print(f"           Real  Fake")
    print(f"Actual Real  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Fake  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}")
    
    print(f"\n✅ BERT training completed!")
    print(f"Model saved to: {model_path}")
    
    return bert_trainer

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BERT model for fake news detection')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for training (default: None, use all data)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--use_gpu', action='store_true', help='Force GPU usage (if available)')
    parser.add_argument('--skip_confirm', action='store_true', help='Skip confirmation prompt')
    parser.add_argument('--augment_file', type=str, default=None, 
                        help='Path to CSV file with augmented examples (must have text and label columns)')
    
    args = parser.parse_args()
    
    if args.use_gpu and not torch.cuda.is_available():
        print("⚠️  GPU requested but not available. Using CPU instead.")
    
    print("\n" + "=" * 60)
    print("BERT TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Sample size: {args.sample}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"GPU Model: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    if args.augment_file:
        print(f"Augmentation file: {args.augment_file}")
    
    if not args.skip_confirm:
        confirm = input("\nProceed with training? (y/n): ")
        if confirm.lower() != 'y':
            print("Training cancelled.")
            return
    
    try:
        bert_trainer = train_bert_model(
            sample_size=args.sample,
            epochs=args.epochs,
            batch_size=args.batch_size,
            augment_file=args.augment_file
        )
        
        # Example predictions
        print("\n" + "=" * 60)
        print("EXAMPLE PREDICTION")
        print("=" * 60)
        
        example_texts = [
            "The president announced new economic policies today.",
            "Aliens have landed in New York and are taking over the city.",
            "Scientists discover breakthrough in renewable energy technology."
        ]
        
        print("\nExample predictions:")
        preds, probs = bert_trainer.predict(example_texts)
        
        for i, text in enumerate(example_texts):
            label = "Fake" if preds[i] == 1 else "Real"
            confidence = probs[i][preds[i]]
            print(f"\nText: {text[:80]}...")
            print(f"Prediction: {label} (confidence: {confidence:.2%})")
            print(f"Probabilities: Real: {probs[i][0]:.2%}, Fake: {probs[i][1]:.2%}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()