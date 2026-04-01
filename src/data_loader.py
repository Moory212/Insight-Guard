from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw")

class DataLoader:
    def __init__(self, test_size=0.2, val_size=0.1, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
    def load_unified_dataset(self):
        """Load the unified dataset (supports .gz compressed version)."""
        gz_path = DATA_DIR / "unified_dataset.csv.gz"
        csv_path = DATA_DIR / "unified_dataset.csv"

        if gz_path.exists():
            df = pd.read_csv(gz_path, compression='gzip')
            print(f"Loaded compressed unified dataset with {len(df)} samples")
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"Loaded unified dataset with {len(df)} samples")
        else:
            raise FileNotFoundError(
                "Unified dataset not found. Please run `python download_datasets.py --unify` first."
            )

        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        return df
    
    def load_kaggle_fake_true(self, use_second_files=True):
        """Load Kaggle datasets with optional second versions"""
        fake = pd.read_csv(DATA_DIR / "Fake.csv", low_memory=False)
        true = pd.read_csv(DATA_DIR / "True.csv", low_memory=False)
        
        if use_second_files:
            try:
                fake2 = pd.read_csv(DATA_DIR / "Fake 2.csv", low_memory=False)
                true2 = pd.read_csv(DATA_DIR / "True 2.csv", low_memory=False)
                fake = pd.concat([fake, fake2], ignore_index=True)
                true = pd.concat([true, true2], ignore_index=True)
            except FileNotFoundError:
                print("Warning: Secondary files not found, using only primary")
        
        fake["label"] = 1
        true["label"] = 0
        
        # Combine title and text
        for df in [fake, true]:
            df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
        
        combined = pd.concat([fake[["text", "label"]], true[["text", "label"]]], ignore_index=True)
        combined["source"] = "kaggle"
        return combined
    
    def load_liar_with_proper_splits(self):
        """Load LIAR with proper train/val/test splits to avoid data leakage"""
        liar_path = DATA_DIR / "liar"
        
        cols = ["id", "label", "statement", "subject", "speaker", "job", "state", "party", 
                "barely_true", "false", "half_true", "mostly_true", "pants_on_fire", "context"]
        
        splits = {}
        
        for split_name in ["train", "test", "valid"]:
            df = pd.read_csv(
                liar_path / f"{split_name}.tsv",
                sep="\t",
                names=cols,
                encoding="utf-8"
            )
            
            def map_label(x):
                x = str(x).lower()
                if x in ["false", "pants-fire", "barely-true"]:
                    return 1
                elif x in ["true", "mostly-true", "half-true"]:
                    return 0
                else:
                    return -1
            
            df["label"] = df["label"].apply(map_label)
            df = df[df["label"] != -1]
            df["text"] = df["statement"]
            df["source"] = "liar"
            splits[split_name] = df[["text", "label", "source"]]
        
        return splits
    
    def load_welfake(self):
        """Load WELFake dataset"""
        try:
            df = pd.read_csv(DATA_DIR / "WELFake_Dataset.csv", low_memory=False)
            text_col = next((c for c in df.columns if "text" in c.lower()), None)
            label_col = next((c for c in df.columns if "label" in c.lower()), None)
            if text_col and label_col:
                df = df.rename(columns={text_col: "text", label_col: "label"})
                df["source"] = "welfake"
                return df[["text", "label", "source"]]
        except Exception as e:
            print(f"Error loading WELFake: {e}")
        return pd.DataFrame(columns=["text", "label", "source"])
    
    def load_fakenewsnet_enhanced(self):
        """Enhanced FakeNewsNet loader with metadata"""
        df = pd.read_csv(DATA_DIR / "FakeNewsNet.csv", low_memory=False)
        
        text_candidates = ["text", "content", "news_text", "title", "news"]
        label_candidates = ["label", "class", "is_fake", "fake", "truth_label", "type"]
        
        text_col = next((c for c in text_candidates if c in df.columns), "text")
        label_col = next((c for c in label_candidates if c in df.columns), "label")
        
        if label_col not in df.columns:
            return pd.DataFrame(columns=["text", "label", "source"])
        
        def normalize_label(x):
            if pd.isna(x):
                return 0
            x = str(x).lower()
            if x in ["1", "fake", "false", "f", "rumor", "clickbait"]:
                return 1
            elif x in ["0", "real", "true", "t", "trusted"]:
                return 0
            else:
                try:
                    return int(float(x))
                except:
                    return 0
        
        df["label"] = df[label_col].apply(normalize_label)
        df["text"] = df[text_col].astype(str)
        df["source"] = "fakenewsnet"
        
        return df[["text", "label", "source"]]
    
    def load_all_datasets(self, include_liar_test=False, use_unified=True):
        """Load all datasets. If use_unified=True, load the unified dataset and split it.
        Otherwise, use the original multi-source loading."""
        if use_unified:
            print("Loading unified dataset...")
            df = self.load_unified_dataset()
            
            # Split into train/val/test
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=df["label"]
            )
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state,
                stratify=train_val_df["label"]
            )
            
            print(f"Train size: {len(train_df)}")
            print(f"Validation size: {len(val_df)}")
            print(f"Test size: {len(test_df)}")
            print(f"Label distribution in train: {train_df['label'].value_counts().to_dict()}")
            
            return train_df, val_df, test_df
        
        else:
            # Original loading logic (kept for backward compatibility)
            print("Loading individual datasets...")
            kaggle_df = self.load_kaggle_fake_true()
            liar_splits = self.load_liar_with_proper_splits()
            welfake_df = self.load_welfake()
            fakenewsnet_df = self.load_fakenewsnet_enhanced()
            
            combined_df = pd.concat([
                kaggle_df,
                welfake_df,
                fakenewsnet_df
            ], ignore_index=True)
            
            train_val_df, test_df = train_test_split(
                combined_df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=combined_df["label"]
            )
            
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size/(1-self.test_size),
                random_state=self.random_state,
                stratify=train_val_df["label"]
            )
            
            train_df = pd.concat([train_df, liar_splits["train"]], ignore_index=True)
            val_df = pd.concat([val_df, liar_splits["valid"]], ignore_index=True)
            
            if include_liar_test:
                test_df = pd.concat([test_df, liar_splits["test"]], ignore_index=True)
            
            print(f"Train size: {len(train_df)}")
            print(f"Validation size: {len(val_df)}")
            print(f"Test size: {len(test_df)}")
            print(f"Label distribution in train: {train_df['label'].value_counts().to_dict()}")
            
            return train_df, val_df, test_df
    
    def get_class_weights(self, y):
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))