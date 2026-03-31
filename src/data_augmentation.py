# src/data_augmentation.py
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataAugmenter:
    def __init__(self, method='synonym', aug_p=0.3):
        """
        Initialize data augmenter
        
        Args:
            method: 'synonym', 'contextual', 'char', or 'random'
            aug_p: Probability of augmentation for each word
        """
        self.method = method
        self.aug_p = aug_p
        
        if method == 'synonym':
            self.aug = naw.SynonymAug(aug_src='wordnet', aug_p=aug_p)
        elif method == 'contextual':
            self.aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased',
                action="substitute",
                aug_p=aug_p,
                device='cuda' if self._has_gpu() else 'cpu'
            )
        elif method == 'char':
            self.aug = nac.KeyboardAug(aug_char_p=aug_p, aug_word_p=aug_p)
        elif method == 'random':
            self.aug = naw.RandomWordAug(action="swap", aug_p=aug_p)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
    
    def _has_gpu(self):
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def augment_text(self, text: str, n_aug: int = 1) -> List[str]:
        """
        Generate augmented versions of text
        
        Args:
            text: Input text
            n_aug: Number of augmented versions to generate
            
        Returns:
            List of augmented texts (including original)
        """
        if not text or not isinstance(text, str):
            return [text]
        
        try:
            augmented_texts = [text]
            
            for _ in range(n_aug):
                aug_text = self.aug.augment(text)
                if isinstance(aug_text, list):
                    aug_text = aug_text[0]
                augmented_texts.append(aug_text)
            
            return list(set(augmented_texts))  # Remove duplicates
            
        except Exception as e:
            print(f"Warning: Augmentation failed for text: {e}")
            return [text]
    
    def augment_dataset(self, df: pd.DataFrame, text_col: str = 'text', 
                       label_col: str = 'label', target_samples: int = None,
                       balance_classes: bool = True) -> pd.DataFrame:
        """
        Augment dataset to increase size or balance classes
        
        Args:
            df: Input DataFrame
            text_col: Name of text column
            label_col: Name of label column
            target_samples: Target number of samples per class
            balance_classes: Whether to balance class distribution
            
        Returns:
            Augmented DataFrame
        """
        if df.empty:
            return df
        
        print(f"\n📊 Original dataset:")
        print(f"   Total samples: {len(df)}")
        print(f"   Class distribution:")
        class_counts = df[label_col].value_counts()
        for label, count in class_counts.items():
            print(f"     Class {label}: {count} samples")
        
        augmented_samples = []
        
        if balance_classes:
            # Balance classes
            max_class_size = class_counts.max()
            
            for label in class_counts.index:
                class_df = df[df[label_col] == label]
                current_size = len(class_df)
                
                if current_size < max_class_size:
                    # Calculate how many augmentations needed
                    needed = max_class_size - current_size
                    
                    # Augment samples
                    for i in range(needed):
                        sample = class_df.iloc[i % current_size]
                        text = sample[text_col]
                        
                        # Generate augmented text
                        aug_texts = self.augment_text(text, n_aug=1)
                        if len(aug_texts) > 1:
                            aug_text = aug_texts[1]  # Take augmented version
                            augmented_samples.append({
                                text_col: aug_text,
                                label_col: label
                            })
        
        # If target_samples specified, augment to reach target
        elif target_samples and len(df) < target_samples:
            needed = target_samples - len(df)
            n_per_sample = max(1, needed // len(df))
            
            for _, row in df.iterrows():
                text = row[text_col]
                label = row[label_col]
                
                aug_texts = self.augment_text(text, n_aug=n_per_sample)
                for aug_text in aug_texts[1:]:  # Skip original
                    augmented_samples.append({
                        text_col: aug_text,
                        label_col: label
                    })
                
                if len(augmented_samples) >= needed:
                    break
        
        # Combine with original
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            result_df = pd.concat([df, augmented_df], ignore_index=True)
        else:
            result_df = df.copy()
        
        # Shuffle
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n📊 Augmented dataset:")
        print(f"   Total samples: {len(result_df)}")
        print(f"   Added samples: {len(augmented_samples)}")
        print(f"   New class distribution:")
        new_class_counts = result_df[label_col].value_counts()
        for label, count in new_class_counts.items():
            print(f"     Class {label}: {count} samples")
        
        return result_df
    
    def create_augmented_folds(self, df: pd.DataFrame, text_col: str = 'text',
                              label_col: str = 'label', n_folds: int = 5) -> List[Tuple]:
        """
        Create augmented cross-validation folds
        
        Args:
            df: Input DataFrame
            text_col: Name of text column
            label_col: Name of label column
            n_folds: Number of folds
            
        Returns:
            List of (train_df, val_df) tuples
        """
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = []
        
        X = df[text_col].values
        y = df[label_col].values
        
        for train_idx, val_idx in skf.split(X, y):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            
            # Augment training data only
            train_df_aug = self.augment_dataset(train_df, text_col, label_col)
            
            folds.append((train_df_aug, val_df))
        
        print(f"\n✅ Created {n_folds} augmented cross-validation folds")
        return folds
    
    def create_mixed_augmentations(self, text: str, n_variations: int = 3) -> List[str]:
        """
        Create multiple augmentations using different methods
        
        Args:
            text: Input text
            n_variations: Number of variations to create
            
        Returns:
            List of augmented texts
        """
        variations = [text]
        
        # Try different augmentation methods
        methods_to_try = ['synonym', 'random']
        if self._has_gpu():
            methods_to_try.append('contextual')
        
        for method in methods_to_try:
            if len(variations) >= n_variations + 1:
                break
            
            try:
                if method == 'synonym':
                    aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.3)
                elif method == 'random':
                    aug = naw.RandomWordAug(action="swap", aug_p=0.3)
                elif method == 'contextual':
                    aug = naw.ContextualWordEmbsAug(
                        model_path='bert-base-uncased',
                        action="substitute",
                        aug_p=0.3
                    )
                
                aug_text = aug.augment(text)
                if isinstance(aug_text, list):
                    aug_text = aug_text[0]
                
                if aug_text != text and aug_text not in variations:
                    variations.append(aug_text)
                    
            except:
                continue
        
        return variations[:n_variations + 1]


# Example usage
if __name__ == "__main__":
    # Test the augmenter
    augmenter = DataAugmenter(method='synonym')
    
    # Test single text
    test_text = "The government announced new economic policies today."
    augmented = augmenter.augment_text(test_text, n_aug=2)
    
    print("Original:", test_text)
    print("Augmented:")
    for i, text in enumerate(augmented[1:], 1):
        print(f"  {i}. {text}")
    
    # Test dataset augmentation
    test_df = pd.DataFrame({
        'text': [
            "Breaking news about the economy",
            "New scientific discovery announced",
            "Political scandal revealed",
            "Sports team wins championship"
        ],
        'label': [1, 0, 1, 0]
    })
    
    print("\nDataset augmentation test:")
    augmented_df = augmenter.augment_dataset(test_df, target_samples=8)
    print(augmented_df)