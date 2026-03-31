# src/bert_model.py - OPTIMIZED VERSION
import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FakeNewsDataset(Dataset):
    """Optimized Dataset for BERT"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize for faster loading
        self.encodings = self._pre_tokenize()
        
    def _pre_tokenize(self):
        """Pre-tokenize all texts for faster loading"""
        print("Tokenizing dataset...")
        encodings = self.tokenizer(
            self.texts.tolist() if hasattr(self.texts, 'tolist') else list(self.texts),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encodings
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BertFakeNewsClassifier(nn.Module):
    """Optimized BERT model for fake news classification"""
    def __init__(self, num_classes=2, dropout_rate=0.3, freeze_bert=False):
        super(BertFakeNewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT layers if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use mean pooling of last hidden state
        last_hidden_state = outputs.last_hidden_state
        mean_pooled = torch.mean(last_hidden_state, dim=1)
        
        mean_pooled = self.dropout(mean_pooled)
        logits = self.classifier(mean_pooled)
        
        return logits

class BertTrainer:
    """Optimized trainer for BERT model"""
    def __init__(self, model_name='bert-base-uncased', max_length=128, 
                 batch_size=16, learning_rate=2e-5, epochs=3, gradient_accumulation_steps=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
    def prepare_data(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        """Prepare datasets and dataloaders with optimized batch sizes"""
        train_dataset = FakeNewsDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = FakeNewsDataset(X_val, y_val, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = None
        if X_test is not None and y_test is not None:
            test_dataset = FakeNewsDataset(X_test, y_test, self.tokenizer, self.max_length)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=0
            )
            
        return train_loader, val_loader, test_loader
    
    def train(self, X_train, y_train, X_val, y_val, freeze_bert=False):
        """Train the BERT model with optimizations"""
        print("Preparing BERT data...")
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        
        model = BertFakeNewsClassifier(freeze_bert=freeze_bert).to(self.device)
        
        # Optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        print(f"Training BERT for {self.epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        best_val_f1 = 0
        best_model_state = None
        patience = 3
        patience_counter = 0
        
        for epoch in range(self.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'='*50}")
            
            # Training phase
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Mixed precision training
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                        loss = loss / self.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                
                train_loss += loss.item() * self.gradient_accumulation_steps
                
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    train_preds.extend(preds)
                    train_labels.extend(labels.cpu().numpy())
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_acc = accuracy_score(train_labels, train_preds)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            val_precision = precision_score(val_labels, val_preds, average='weighted')
            val_recall = recall_score(val_labels, val_preds, average='weighted')
            
            print(f"\n📊 Training Metrics:")
            print(f"   Loss: {train_loss/len(train_loader):.4f}")
            print(f"   Accuracy: {train_acc:.4f}")
            
            print(f"\n📊 Validation Metrics:")
            print(f"   Loss: {val_loss/len(val_loader):.4f}")
            print(f"   Accuracy: {val_acc:.4f}")
            print(f"   F1 Score: {val_f1:.4f}")
            print(f"   Precision: {val_precision:.4f}")
            print(f"   Recall: {val_recall:.4f}")
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"   🏆 New best model! (F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"   No improvement for {patience_counter} epoch(s)")
                
                if patience_counter >= patience:
                    print(f"   Early stopping triggered!")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n✅ Loaded best model with validation F1: {best_val_f1:.4f}")
        
        return model
    
    def evaluate(self, model, X_test, y_test):
        """Comprehensive evaluation on test set"""
        print("\nEvaluating BERT on test set...")
        
        _, _, test_loader = self.prepare_data([], [], [], [], X_test, y_test)
        model.eval()
        
        test_preds = []
        test_labels = []
        test_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                test_preds.extend(preds)
                test_labels.extend(labels.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='weighted')
        test_precision = precision_score(test_labels, test_preds, average='weighted')
        test_recall = recall_score(test_labels, test_preds, average='weighted')
        
        print(f"\n📊 Test Set Results:")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   F1 Score: {test_f1:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f}")
        
        # Classification report
        from sklearn.metrics import classification_report
        print(f"\n📋 Classification Report:")
        print(classification_report(test_labels, test_preds, target_names=['Real', 'Fake']))
        
        return {
            'accuracy': test_acc,
            'f1_score': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'predictions': test_preds,
            'probabilities': test_probs
        }
    
    def predict(self, model, texts, batch_size=None):
        """Make predictions on new texts"""
        if batch_size is None:
            batch_size = self.batch_size
        
        model.eval()
        predictions = []
        probabilities = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoding = self.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return predictions, probabilities
    
    def save_model(self, model, path, save_tokenizer=True):
        """Save the model, tokenizer, and config"""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'bert_config': model.bert.config.to_dict(),
            'classifier_config': {
                'num_classes': model.classifier.out_features,
                'dropout_rate': model.dropout.p
            }
        }, path)
        
        # Save tokenizer
        if save_tokenizer:
            tokenizer_dir = os.path.splitext(path)[0] + "_tokenizer"
            self.tokenizer.save_pretrained(tokenizer_dir)
        
        print(f"✅ Model saved to: {path}")
        
        return path
    
    def load_model(self, path):
        """Load the model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Reconstruct model
        model = BertFakeNewsClassifier(
            num_classes=checkpoint.get('classifier_config', {}).get('num_classes', 2),
            dropout_rate=checkpoint.get('classifier_config', {}).get('dropout_rate', 0.3)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✅ Model loaded from: {path}")
        
        return model


# Helper function for quick predictions
def load_bert_predictor(model_path="models/bert_model.pt", tokenizer_path=None):
    """Load BERT model for predictions"""
    trainer = BertTrainer()
    
    # Load model
    model = trainer.load_model(model_path)
    model.eval()
    
    # Load tokenizer
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return model, tokenizer, trainer


if __name__ == "__main__":
    # Test the BERT trainer
    print("Testing BERT Trainer...")
    
    # Create dummy data
    texts = [
        "This is a real news article about science.",
        "Fake news alert! This is completely made up.",
        "Another legitimate news story.",
        "More fake content to test with."
    ]
    labels = [0, 1, 0, 1]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.25, random_state=42
    )
    
    # Initialize trainer
    trainer = BertTrainer(
        model_name='bert-base-uncased',
        max_length=32,
        batch_size=2,
        learning_rate=2e-5,
        epochs=2
    )
    
    # Train
    model = trainer.train(X_train, y_train, X_val, y_val, freeze_bert=False)
    
    # Test predictions
    test_texts = ["This is a test news article."]
    predictions, probabilities = trainer.predict(model, test_texts)
    
    print(f"\nTest prediction: {predictions[0]} (Fake: {probabilities[0][1]:.3f}, Real: {probabilities[0][0]:.3f})")