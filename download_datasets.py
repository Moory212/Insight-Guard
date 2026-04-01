# download_datasets_working.py - ENHANCED WITH FIXED URLs AND BETTER HANDLING
import os
import sys
import requests
import zipfile
import tarfile
import gzip
import pandas as pd
import io
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class DatasetDownloader:
    def __init__(self):
        self.datasets_info = self._get_datasets_info()
        
    def _get_datasets_info(self):
        """Define datasets with working URLs and preprocessing"""
        return {
            # COVID-19 (working)
            "COVID19_FakeNews": {
                "url": "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/Constraint_Train.csv",
                "filename": "COVID19_Train.csv",
                "description": "COVID-19 Fake News Dataset",
                "type": "csv",
                "preprocess_func": self._preprocess_covid19
            },
            
            # FakeNewsCorpus sample (working)
            "FakeRealNews": {
                "url": "https://github.com/several27/FakeNewsCorpus/raw/master/news_sample.csv",
                "filename": "FakeNewsCorpus_sample.csv",
                "description": "Fake News Corpus Sample",
                "type": "csv",
                "preprocess_func": self._preprocess_fakenewscorpus
            },
            
            # LIAR Dataset (working alternative)
            "LIAR_Original": {
                "url": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
                "filename": "liar_dataset.zip",
                "description": "LIAR: A Benchmark Dataset for Fake News Detection",
                "type": "zip",
                "extract_to": "liar",
                "preprocess_func": self._preprocess_liar
            },
            
            # Fake News Challenge (working)
            "FakeNewsChallenge": {
                "url": "https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_bodies.csv",
                "filename": "fnc_bodies.csv",
                "description": "Fake News Challenge Dataset - Bodies",
                "type": "csv",
                "additional_files": [
                    ("https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/train_stances.csv", "fnc_stances.csv")
                ],
                "preprocess_func": self._preprocess_fnc
            },
            
            # BuzzFeed (fix preprocessing)
            "BuzzFeedNews": {
                "url": "https://github.com/BuzzFeedNews/2016-10-facebook-fact-check/raw/master/data/facebook-fact-check.csv",
                "filename": "buzzfeed_fact_check.csv",
                "description": "BuzzFeed Fact Check Dataset",
                "type": "csv",
                "preprocess_func": self._preprocess_buzzfeed
            },
            
            # ISOT full dataset (working from UCI or Kaggle mirror)
            "ISOT_Full": {
                "url": "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/dataset/ISOT_Fake_News_Dataset/Fake.csv",
                "filename": "Fake.csv",
                "description": "ISOT Fake News Dataset (Fake articles)",
                "type": "csv",
                "additional_files": [
                    ("https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/dataset/ISOT_Fake_News_Dataset/True.csv", "True.csv")
                ],
                "preprocess_func": self._preprocess_isot_full
            },
            
            # Alternative ISOT (working from UCI)
            "ISOT_UCI": {
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00514/fake.zip",
                "filename": "fake_uci.zip",
                "description": "ISOT Fake News Dataset (UCI Archive)",
                "type": "zip",
                "extract_to": "isot_uci",
                "preprocess_func": self._preprocess_isot_uci
            },
            
            # Kaggle Fake News (working via direct download from a known mirror)
            "Kaggle_FakeNews": {
                "url": "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_news_train.csv",
                "filename": "kaggle_fake_news.csv",
                "description": "Kaggle Fake News Dataset (from GitHub mirror)",
                "type": "csv",
                "preprocess_func": self._preprocess_kaggle
            },
            
            # Twitter Sentiment (TSV - fixed)
            "TwitterSentiment": {
                "url": "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv",
                "filename": "twitter_sentiment.tsv",
                "description": "Twitter Sentiment Dataset (can be adapted)",
                "type": "tsv",
                "preprocess_func": self._preprocess_twitter_sentiment
            },
            
            # OpenSources (working via Kaggle? Use GitHub mirror)
            "OpenSources": {
                "url": "https://raw.githubusercontent.com/OpenSourcesGroup/opensources/master/sources/sources.csv",
                "filename": "opensources.csv",
                "description": "OpenSources.co credibility dataset",
                "type": "csv",
                "preprocess_func": self._preprocess_opensources
            },
            
            # Pheme (needs reliable source) – skip for now
            # "Pheme": {
            #     "url": "https://figshare.com/ndownloader/articles/6392078/versions/1",
            #     "filename": "pheme.zip",
            #     "description": "PHEME Dataset of Rumours",
            #     "type": "zip",
            #     "extract_to": "pheme",
            #     "preprocess_func": self._preprocess_pheme
            # },
            
            # FakeNewsNet (Kaggle) – requires manual setup, skip
        }
    
    def download_file(self, url, filename, description="", chunk_size=8192):
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"✓ Already downloaded: {filename}")
            return filepath
        
        print(f"\nDownloading {description}...")
        print(f"URL: {url}")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            print(f"✓ Downloaded: {filename}")
            return filepath
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to download {filename}: {e}")
            return None
    
    def download_and_extract_zip(self, url, zip_filename, extract_subdir, description=""):
        zip_path = self.download_file(url, zip_filename, description)
        if not zip_path:
            return None
        extract_path = DATA_DIR / extract_subdir
        extract_path.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"✓ Extracted {zip_filename} to {extract_path}")
            return extract_path
        except Exception as e:
            print(f"✗ Failed to extract {zip_filename}: {e}")
            return None
    
    # ----- Preprocessing functions -----
    def _preprocess_covid19(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'tweet' in df.columns:
                df = df.rename(columns={'tweet': 'text'})
            if 'label' in df.columns:
                df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() in ['fake', '1'] else 0)
            df[['text', 'label']].to_csv(DATA_DIR / "COVID19_processed.csv", index=False)
            print(f"✓ Processed COVID-19: {len(df)} samples")
            return True
        except Exception as e:
            print(f"✗ COVID-19 preprocessing failed: {e}")
            return False
    
    def _preprocess_fakenewscorpus(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'type' in df.columns:
                df['label'] = df['type'].apply(lambda x: 1 if str(x).lower() in ['fake', 'bs', 'conspiracy'] else 0)
            if 'content' in df.columns:
                df['text'] = df['content']
            df[['text', 'label']].to_csv(DATA_DIR / "FakeNewsCorpus_processed.csv", index=False)
            print(f"✓ Processed FakeNewsCorpus: {len(df)} samples")
            return True
        except Exception as e:
            print(f"✗ FakeNewsCorpus preprocessing failed: {e}")
            return False
    
    def _preprocess_liar(self, zip_path):
        extract_path = DATA_DIR / "liar"
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            dfs = []
            for split in ['train.tsv', 'test.tsv', 'valid.tsv']:
                file_path = extract_path / split
                if file_path.exists():
                    df = pd.read_csv(file_path, sep='\t', header=None,
                                     names=['id','label','statement','subject','speaker','job','state','party',
                                            'barely_true','false','half_true','mostly_true','pants_on_fire','context'])
                    fake_labels = ['pants-fire', 'false']
                    df['label'] = df['label'].apply(lambda x: 1 if x in fake_labels else 0)
                    dfs.append(df[['statement', 'label']].rename(columns={'statement':'text'}))
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                combined.to_csv(DATA_DIR / "liar_processed.csv", index=False)
                print(f"✓ Processed LIAR: {len(combined)} samples")
                return True
        except Exception as e:
            print(f"✗ LIAR preprocessing failed: {e}")
            return False
    
    def _preprocess_fnc(self, bodies_path):
        try:
            bodies = pd.read_csv(bodies_path)
            stances_path = DATA_DIR / "fnc_stances.csv"
            if stances_path.exists():
                stances = pd.read_csv(stances_path)
                merged = pd.merge(stances, bodies, on='Body ID')
                # For fake news, we might consider 'unrelated' as fake, but FNC is stance detection.
                # We'll just save merged for later use.
                merged.to_csv(DATA_DIR / "fnc_merged.csv", index=False)
                print(f"✓ Merged FNC: {len(merged)} rows")
            return True
        except Exception as e:
            print(f"✗ FNC preprocessing failed: {e}")
            return False
    
    def _preprocess_buzzfeed(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Map ratings to binary: 'no factual content', 'mixture of true and false', 'mostly false' -> fake (1); rest -> real (0)
            fake_ratings = ['no factual content', 'mixture of true and false', 'mostly false', 'pants on fire']
            if 'Rating' in df.columns:
                df['label'] = df['Rating'].apply(lambda x: 1 if str(x).lower() in [r.lower() for r in fake_ratings] else 0)
            # Some versions have 'title' or 'text' – create text column
            if 'text' not in df.columns:
                if 'title' in df.columns:
                    df['text'] = df['title']
                elif 'Post URL' in df.columns:
                    # No text, use URL as placeholder? Better skip.
                    print("✗ BuzzFeed: no text column")
                    return False
            df[['text', 'label']].to_csv(DATA_DIR / "buzzfeed_processed.csv", index=False)
            print(f"✓ Processed BuzzFeed: {len(df)} samples")
            return True
        except Exception as e:
            print(f"✗ BuzzFeed preprocessing failed: {e}")
            return False
    
    def _preprocess_isot_full(self, fake_path):
        # fake_path is the path to Fake.csv; True.csv should also exist
        try:
            fake_df = pd.read_csv(DATA_DIR / "Fake.csv")
            true_df = pd.read_csv(DATA_DIR / "True.csv")
            fake_df['label'] = 1
            true_df['label'] = 0
            combined = pd.concat([fake_df, true_df], ignore_index=True)
            # Ensure text column: if both title and text exist, combine
            if 'title' in combined.columns and 'text' in combined.columns:
                combined['text'] = combined['title'] + " " + combined['text']
            elif 'text' not in combined.columns and 'title' in combined.columns:
                combined['text'] = combined['title']
            combined[['text', 'label']].to_csv(DATA_DIR / "isot_full_processed.csv", index=False)
            print(f"✓ Processed ISOT Full: {len(combined)} samples")
            return True
        except Exception as e:
            print(f"✗ ISOT Full preprocessing failed: {e}")
            return False
    
    def _preprocess_isot_uci(self, zip_path):
        extract_path = DATA_DIR / "isot_uci"
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            # Find CSV files
            fake_path = extract_path / "Fake.csv"
            true_path = extract_path / "True.csv"
            if fake_path.exists() and true_path.exists():
                return self._preprocess_isot_full(fake_path)  # reuse
            else:
                print("✗ UCI ISOT files not found after extraction")
                return False
        except Exception as e:
            print(f"✗ ISOT UCI preprocessing failed: {e}")
            return False
    
    def _preprocess_kaggle(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Many Kaggle datasets have 'text' and 'label' columns
            if 'text' in df.columns and 'label' in df.columns:
                df[['text', 'label']].to_csv(DATA_DIR / "kaggle_processed.csv", index=False)
                print(f"✓ Processed Kaggle: {len(df)} samples")
                return True
            else:
                print(f"✗ Kaggle columns not as expected: {df.columns.tolist()}")
                return False
        except Exception as e:
            print(f"✗ Kaggle preprocessing failed: {e}")
            return False
    
    def _preprocess_twitter_sentiment(self, tsv_path):
        try:
            df = pd.read_csv(tsv_path, sep='\t', header=None, names=['text', 'label'])
            # Sentiment label: 1 = positive, 0 = negative – we can keep as is for sentiment, but for fake news it's not directly applicable.
            # We'll just save processed.
            df.to_csv(DATA_DIR / "twitter_sentiment_processed.csv", index=False)
            print(f"✓ Processed Twitter Sentiment: {len(df)} samples")
            return True
        except Exception as e:
            print(f"✗ Twitter Sentiment preprocessing failed: {e}")
            return False
    
    def _preprocess_opensources(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            # OpenSources is a list of sources with credibility labels.
            # For our purpose, we might use it later for domain-based features.
            # Save as is.
            df.to_csv(DATA_DIR / "opensources_processed.csv", index=False)
            print(f"✓ Processed OpenSources: {len(df)} samples")
            return True
        except Exception as e:
            print(f"✗ OpenSources preprocessing failed: {e}")
            return False
    
    # ----- Download orchestration -----
    def download_dataset(self, dataset_name):
        if dataset_name not in self.datasets_info:
            print(f"✗ Unknown dataset: {dataset_name}")
            return False
        
        info = self.datasets_info[dataset_name]
        print(f"\n{'='*60}")
        print(f"Downloading: {dataset_name}")
        print(f"Description: {info['description']}")
        print(f"{'='*60}")
        
        if info['type'] == 'csv':
            filepath = self.download_file(info['url'], info['filename'], info['description'])
            if not filepath:
                return False
            if 'additional_files' in info:
                for url, filename in info['additional_files']:
                    self.download_file(url, filename, f"{dataset_name} - {filename}")
            if 'preprocess_func' in info:
                info['preprocess_func'](filepath)
            return True
        
        elif info['type'] == 'zip':
            extract_path = self.download_and_extract_zip(info['url'], info['filename'], info.get('extract_to', dataset_name), info['description'])
            if not extract_path:
                return False
            if 'preprocess_func' in info:
                info['preprocess_func'](DATA_DIR / info['filename'])
            return True
        
        elif info['type'] == 'tsv':
            filepath = self.download_file(info['url'], info['filename'], info['description'])
            if not filepath:
                return False
            if 'preprocess_func' in info:
                info['preprocess_func'](filepath)
            return True
        
        else:
            print(f"✗ Unknown dataset type: {info['type']}")
            return False
    
    def download_all(self):
        print("Starting download of all datasets...")
        downloaded, failed = [], []
        for name in self.datasets_info:
            if self.download_dataset(name):
                downloaded.append(name)
            else:
                failed.append(name)
            import time
            time.sleep(0.5)
        
        print(f"\n{'='*60}\nDOWNLOAD SUMMARY\n{'='*60}")
        print(f"Successfully downloaded: {len(downloaded)} datasets: {', '.join(downloaded)}")
        print(f"Failed: {len(failed)} datasets: {', '.join(failed)}")
        self.verify_downloads()
        return downloaded, failed
    
    def verify_downloads(self):
        print(f"\n{'='*60}\nVERIFYING DOWNLOADED DATASETS\n{'='*60}")
        files = list(DATA_DIR.glob("*"))
        csv_files = list(DATA_DIR.glob("*.csv"))
        zip_files = list(DATA_DIR.glob("*.zip"))
        print(f"\nTotal files in data/raw: {len(files)}")
        if csv_files:
            print(f"\nCSV files ({len(csv_files)}):")
            for f in csv_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                try:
                    df = pd.read_csv(f, nrows=2)
                    cols = df.columns.tolist()
                    print(f"  {f.name} ({size_mb:.2f} MB) - Columns: {cols}")
                except:
                    print(f"  {f.name} ({size_mb:.2f} MB) - Could not read")
        if zip_files:
            print(f"\nZIP files ({len(zip_files)}):")
            for f in zip_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name} ({size_mb:.2f} MB)")
    
    def create_unified_dataset(self):
        """Combine all processed datasets and also scan raw CSVs for text/label"""
        print(f"\n{'='*60}\nCREATING UNIFIED DATASET\n{'='*60}")
        processed_files = list(DATA_DIR.glob("*_processed.csv"))
        raw_csvs = [f for f in DATA_DIR.glob("*.csv") if '_processed' not in f.name and f.name not in ['fnc_bodies.csv','fnc_stances.csv','fnc_merged.csv']]
        all_data = []
        
        # First, load processed files
        for f in processed_files:
            try:
                df = pd.read_csv(f)
                if 'text' in df.columns and 'label' in df.columns:
                    all_data.append(df[['text', 'label']])
                    print(f"✓ Added {f.name}: {len(df)} samples")
                else:
                    print(f"✗ {f.name} missing text/label columns")
            except Exception as e:
                print(f"✗ Error reading {f.name}: {e}")
        
        # Then try to extract from raw CSVs
        for f in raw_csvs:
            try:
                df = pd.read_csv(f)
                # Heuristic: find text and label columns
                text_col = None
                label_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    if any(k in col_lower for k in ['text', 'content', 'tweet', 'title', 'statement', 'headline']):
                        text_col = col
                    if any(k in col_lower for k in ['label', 'class', 'fake', 'real', 'type', 'rating']):
                        label_col = col
                if text_col and label_col:
                    df_processed = df[[text_col, label_col]].copy()
                    df_processed.columns = ['text', 'label']
                    # Clean label: convert to binary (1 for fake/unreliable, 0 for real/reliable)
                    # This is heuristic – user may need to adjust
                    def clean_label(val):
                        val_str = str(val).lower().strip()
                        if val_str in ['fake', '1', 'true', '1.0', 'f', 'pants-fire', 'false', 'bs', 'conspiracy', 'no factual content', 'mixture of true and false', 'mostly false']:
                            return 1
                        elif val_str in ['real', '0', 'false', '0.0', 'r', 'true', 'mostly true', 'half true', 'barely true']:
                            return 0
                        else:
                            # try numeric
                            try:
                                return int(float(val))
                            except:
                                return 0
                    df_processed['label'] = df_processed['label'].apply(clean_label)
                    all_data.append(df_processed)
                    print(f"✓ Extracted from {f.name}: {len(df_processed)} samples")
            except Exception as e:
                pass  # silently skip unreadable files
        
        if not all_data:
            print("No valid data found!")
            return None
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.dropna()
        combined = combined[combined['text'].str.strip() != '']
        combined['label'] = combined['label'].astype(int)
        initial_len = len(combined)
        combined = combined.drop_duplicates(subset=['text'])
        final_len = len(combined)
        
        print(f"\nUnified dataset: {final_len:,} samples ({initial_len - final_len:,} duplicates removed)")
        print(f"Fake: {combined['label'].sum():,}, Real: {(combined['label'] == 0).sum():,} (balance: {combined['label'].mean():.2%} fake)")
        
        unified_path = DATA_DIR / "unified_dataset.csv"
        combined.to_csv(unified_path, index=False)
        print(f"✓ Saved unified dataset to {unified_path}")
        
        # Also create a balanced sample
        fake = combined[combined['label'] == 1]
        real = combined[combined['label'] == 0]
        min_count = min(len(fake), len(real), 10000)
        fake_sample = fake.sample(n=min_count, random_state=42)
        real_sample = real.sample(n=min_count, random_state=42)
        balanced = pd.concat([fake_sample, real_sample], ignore_index=True).sample(frac=1, random_state=42)
        balanced.to_csv(DATA_DIR / "balanced_dataset.csv", index=False)
        print(f"✓ Saved balanced dataset ({len(balanced)} samples) to {DATA_DIR / 'balanced_dataset.csv'}")
        
        return combined

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download fake news datasets (FIXED VERSION)")
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--dataset', type=str, help='Download specific dataset')
    parser.add_argument('--verify', action='store_true', help='Verify downloaded files')
    parser.add_argument('--unify', action='store_true', help='Create unified dataset')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.list:
        print("\nAvailable datasets:")
        for name, info in downloader.datasets_info.items():
            print(f"  {name:25} - {info['description']}")
        return
    
    if args.dataset:
        downloader.download_dataset(args.dataset)
    
    if args.all:
        downloader.download_all()
    
    if args.verify:
        downloader.verify_downloads()
    
    if args.unify:
        downloader.create_unified_dataset()
    
    if not any(vars(args).values()):
        print(__doc__)

if __name__ == "__main__":
    main()