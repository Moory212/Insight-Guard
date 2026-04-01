# test_nltk.py
try:
    import nltk
    
    print("Testing NLTK installation...")
    
    # Test basic imports
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    print("✅ NLTK imports working")
    
    # Test if data is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
        print("✅ punkt tokenizer data found")
    except LookupError:
        print("❌ punkt tokenizer data NOT found")
        print("   Run: nltk.download('punkt')")
    
    try:
        nltk.data.find('corpora/stopwords')
        print("✅ stopwords data found")
    except LookupError:
        print("❌ stopwords data NOT found")
        print("   Run: nltk.download('stopwords')")
    
    # Test functionality
    test_text = "This is a test sentence for NLTK."
    tokens = word_tokenize(test_text)
    print(f"✅ Tokenization test: {tokens}")
    
except ImportError:
    print("❌ NLTK not installed")
    print("   Install with: pip install nltk")
except Exception as e:
    print(f"❌ NLTK error: {e}")