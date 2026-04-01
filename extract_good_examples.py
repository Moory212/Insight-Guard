import sys
import os

# Add the src directory to the path
sys.path.append('src')

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from data_loader import DataLoader
from preprocessing import TextPreprocessor
import numpy as np

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('models/bert_best_model').to(device)
tokenizer = BertTokenizer.from_pretrained('models/bert_best_model')

# Load test data
data_loader = DataLoader()
_, _, test_df = data_loader.load_all_datasets()
preprocessor = TextPreprocessor()

# Preprocess test texts (use batches to avoid memory issues)
test_texts = preprocessor.batch_preprocess(test_df['text'])
test_labels = test_df['label'].values

# Make predictions
model.eval()
preds = []
with torch.no_grad():
    for i in range(0, len(test_texts), 32):
        batch = test_texts[i:i+32]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        preds.extend(batch_preds)

# Convert preds to numpy array for boolean indexing
preds = np.array(preds)

# Find first correct real (label 0) and fake (label 1)
correct_real_idx = np.where((preds == 0) & (test_labels == 0))[0][0]
correct_fake_idx = np.where((preds == 1) & (test_labels == 1))[0][0]

# Print the texts
print("\n--- CORRECTLY CLASSIFIED REAL ARTICLE ---\n")
print(test_df.iloc[correct_real_idx]['text'])
print("\n" + "="*80 + "\n")
print("--- CORRECTLY CLASSIFIED FAKE ARTICLE ---\n")
print(test_df.iloc[correct_fake_idx]['text'])