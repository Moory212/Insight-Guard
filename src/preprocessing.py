import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class TextPreprocessor:
    def __init__(self, use_lemmatization=True, remove_stopwords=True):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.remove_stopwords = remove_stopwords
        
        # Extended stopwords for fake news detection
        self.fake_news_stopwords = {
            'breaking', 'shocking', 'amazing', 'unbelievable', 'must', 
            'click', 'share', 'subscribe', 'watch', 'video', 'http', 'https',
            'www', 'com', 'like', 'comment', 'breakingnews'
        }
        
        # Emoticons and special characters
        self.emoticon_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
        
    def clean_text(self, text):
        """Main cleaning pipeline"""
        if not isinstance(text, str):
            return ""
        
        try:
            text = contractions.fix(text)
        except (IndexError, ValueError) as e:

            contractions_dict = {
                "won't": "will not",
                "can't": "cannot",
                "n't": " not",
                "'re": " are",
                "'s": " is",
                "'d": " would",
                "'ll": " will",
                "'t": " not",
                "'ve": " have",
                "'m": " am"
            }
            for contraction, expansion in contractions_dict.items():
                text = text.replace(contraction, expansion)
        except Exception:
            pass
        
        # Step 2: Lowercase
        text = text.lower()
        
        # Step 3: Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Step 4: Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Step 5: Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\?\!,]', '', text)
        
        # Step 6: Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 7: Remove numbers (optional - can be feature)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Step 8: Tokenize and process
        tokens = word_tokenize(text)
        
        # Step 9: Remove stopwords and custom words
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stopwords 
                     and word not in self.fake_news_stopwords and len(word) > 2]
        
        # Step 10: Lemmatization
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def extract_features(self, text):
        """Extract additional linguistic features"""
        if not isinstance(text, str):
            return {}
        
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
        
        # Count specific patterns
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Count sensational words
        sensational_words = ['shocking', 'amazing', 'unbelievable', 'breaking', 'exclusive']
        features['sensational_score'] = sum(text.count(word) for word in sensational_words)
        
        # Count first person pronouns (often used in fake news)
        first_person = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        features['first_person_count'] = sum(text.count(word) for word in first_person)
        
        return features
    
    def batch_preprocess(self, texts, show_progress=True):
        """Process multiple texts efficiently with error handling"""
        from tqdm import tqdm
        
        processed_texts = []
        
        if show_progress:
            iterator = tqdm(texts, desc="Preprocessing")
        else:
            iterator = texts
            
        for text in iterator:
            try:
                processed_text = self.clean_text(text)
                processed_texts.append(processed_text)
            except Exception as e:
                # If cleaning fails, append empty string and continue
                # Uncomment below for debugging
                # print(f"Warning: Failed to preprocess text: {str(e)[:100]}")
                processed_texts.append("")
        
        return processed_texts