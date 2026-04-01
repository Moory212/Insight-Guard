import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import torch

# Only show the BERT model in the UI
ALLOWED_MODELS = {"bert_best_model"}

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Download stopwords
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

# Import our model loader
from model_loader import ModelLoader

# Set page config
st.set_page_config(
    page_title="Insight Guard - Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global styles - Dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0c10 0%, #111317 100%);
        color: #e5e7eb;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
    }
    .logo-text {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #60A5FA, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .tagline {
        font-size: 1.1rem;
        color: #9CA3AF;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    .model-card {
        background: #1F2937;
        border-radius: 1rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #374151;
        transition: all 0.2s ease;
        cursor: pointer;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    .model-card:hover {
        transform: translateY(-2px);
        border-color: #3B82F6;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3), 0 4px 6px -2px rgba(0,0,0,0.2);
    }
    .model-card.selected {
        border-left: 4px solid #3B82F6;
        background: #111827;
        border-color: #3B82F6;
    }
    .model-name {
        font-weight: 600;
        font-size: 1rem;
        color: #F3F4F6;
    }
    .model-type {
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        background: #1E293B;
        color: #60A5FA;
        display: inline-block;
        font-weight: 500;
    }
    .result-card {
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #1F2937;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -1px rgba(0,0,0,0.2);
    }
    .result-real {
        border-left: 6px solid #10B981;
        background: linear-gradient(135deg, #1F2937, #0B2B1F);
    }
    .result-fake {
        border-left: 6px solid #EF4444;
        background: linear-gradient(135deg, #1F2937, #2C1A1A);
    }
    .confidence-meter {
        background: #374151;
        border-radius: 1rem;
        height: 0.5rem;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        background: #3B82F6;
        height: 100%;
        border-radius: 1rem;
        transition: width 0.3s ease;
    }
    .stTextArea textarea {
        background: #111827;
        border-radius: 0.75rem;
        border: 1px solid #374151;
        color: #e5e7eb;
        font-size: 0.95rem;
        line-height: 1.5;
        transition: border-color 0.2s;
    }
    .stTextArea textarea:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.2);
    }
    .stButton button {
        border-radius: 2rem;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
        background: #3B82F6;
        color: white;
        border: none;
    }
    .stButton button:hover {
        background: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    .css-1d391kg {
        background: #0F1117;
        border-right: 1px solid #1F2937;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F3F4F6;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6B7280;
        font-size: 0.8rem;
        border-top: 1px solid #1F2937;
        margin-top: 2rem;
    }
    .stSelectbox, .stCheckbox {
        color: #e5e7eb;
    }
    .stSelectbox label, .stCheckbox label {
        color: #9CA3AF;
    }
</style>
""", unsafe_allow_html=True)

# Initialize ModelLoader
model_loader = ModelLoader(model_dir="models")

# Session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Header
st.markdown("""
<div class="main-header">
    <div class="logo-text">🛡️ Insight Guard</div>
    <div class="tagline">AI-Powered News Authenticity Checker</div>
</div>
""", unsafe_allow_html=True)

# Discover models
def discover_models():
    return model_loader.discover_models()

# Load model with pipeline override
def load_model(model_info):
    model_name = model_info['name']
    model_type = model_info['type']
    cache_key = f"{model_name}_{model_type}"
    if cache_key in st.session_state.model_cache:
        return st.session_state.model_cache[cache_key]
    
    if model_type == "BERT":
        model, tokenizer = model_loader.load_bert_model(model_name)
        if model:
            model_data = {
                'type': 'bert',
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_name
            }
            st.write(f"✅ BERT model loaded: {model_name}")
        else:
            st.write(f"❌ Failed to load BERT model: {model_name}")
            return None
    else:
        # Traditional models (not used now)
        pipeline_path = Path(model_loader.model_dir) / f"{model_name}_pipeline.pkl"
        if pipeline_path.exists():
            pipeline = joblib.load(pipeline_path)
            model_data = {
                'type': 'traditional',
                'data': pipeline,
                'format': 'pipeline',
                'model_name': model_name
            }
            st.write(f"✅ Pipeline loaded: {model_name}")
        else:
            result = model_loader.load_traditional_model(model_name)
            if result:
                model_data = {
                    'type': 'traditional',
                    'data': result[0],
                    'format': result[1],
                    'model_name': model_name
                }
                st.write(f"✅ Traditional model loaded: {model_name}")
            else:
                st.write(f"❌ Failed to load traditional model: {model_name}")
                return None
    
    st.session_state.model_cache[cache_key] = model_data
    return model_data

# Prediction function
def predict_with_model(model_data, text):
    if not text or not text.strip():
        return "Invalid", 0.5, 0.5, 0.5
    
    def clean_text(text):
        STOPWORDS = set(stopwords.words("english"))
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
        return " ".join(words)
    
    cleaned_text = clean_text(text)
    
    try:
        if model_data['type'] == 'bert':
            st.write("🔮 Predicting with BERT model...")
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                # Get temperature from session state (default 1.0)
                temperature = st.session_state.get('temperature', 1.0)
                # Apply temperature scaling
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=1)
                probs = probs.numpy()[0]
            
            real_prob = float(probs[0])
            fake_prob = float(probs[1])
            
            # Get threshold from session state (default 0.5)
            threshold = st.session_state.get('threshold', 0.5)
            # Determine prediction based on threshold
            if real_prob > threshold:
                prediction = "Real"
            else:
                prediction = "Fake"
            confidence = real_prob if prediction == "Real" else fake_prob
            
            st.write(f"Prediction result: {prediction} (Real: {real_prob:.3f}, Fake: {fake_prob:.3f})")
            return prediction, confidence, real_prob, fake_prob
        
        else:  # traditional model (unused, kept for completeness)
            st.write("🔮 Predicting with traditional model...")
            model_obj = model_data['data']
            model_format = model_data.get('format', 'unknown')
            
            if hasattr(model_obj, 'predict') and hasattr(model_obj, 'transform'):
                prediction = model_obj.predict([cleaned_text])[0]
                if hasattr(model_obj, 'predict_proba'):
                    probability = model_obj.predict_proba([cleaned_text])[0]
                else:
                    probability = [0.5, 0.5]
            elif model_format == "model+vectorizer" and isinstance(model_obj, tuple):
                model, vectorizer = model_obj
                features = vectorizer.transform([cleaned_text])
                prediction = model.predict(features)[0]
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(features)[0]
                else:
                    probability = [0.5, 0.5]
            else:
                vectorizer_path = Path(model_loader.model_dir) / f"{model_data['model_name']}_vectorizer.pkl"
                if vectorizer_path.exists():
                    vectorizer = joblib.load(vectorizer_path)
                    features = vectorizer.transform([cleaned_text])
                    prediction = model_obj.predict(features)[0]
                    if hasattr(model_obj, 'predict_proba'):
                        probability = model_obj.predict_proba(features)[0]
                    else:
                        probability = [0.5, 0.5]
                else:
                    st.error(f"Vectorizer not found for model {model_data['model_name']}")
                    return "Error", 0.5, 0.5, 0.5

            prediction_label = "Fake" if prediction == 1 else "Real"
            confidence = max(probability)
            real_prob = probability[0]
            fake_prob = probability[1] if len(probability) > 1 else 0.5
            st.write(f"Prediction result: {prediction_label} ({confidence:.2%})")
            return prediction_label, confidence, real_prob, fake_prob

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error", 0.5, 0.5, 0.5

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 Model Selection")
    available_models = discover_models()
    available_models = [m for m in available_models if m['name'] in ALLOWED_MODELS]

    if st.session_state.selected_model not in [m['name'] for m in available_models]:
        st.session_state.selected_model = None
        st.session_state.current_model = None
    
    if not available_models:
        st.error("❌ No models found. Train a model first.")
    else:
        st.success(f"✅ {len(available_models)} model(s) available")
        
        for i, model_info in enumerate(available_models):
            is_selected = st.session_state.selected_model == model_info['name']
            card_class = "model-card selected" if is_selected else "model-card"
            st.markdown(f"""
            <div class="{card_class}" id="model_{i}">
                <div class="model-name">{model_info['name']}</div>
                <div><span class="model-type">{model_info['type']}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Select", key=f"select_{i}", use_container_width=True):
                st.session_state.selected_model = model_info['name']
                model_data = load_model(model_info)
                if model_data:
                    st.session_state.current_model = model_data
                    st.rerun()
        
        if st.session_state.selected_model:
            st.markdown("---")
            st.markdown("### 📊 Model Details")
            selected_info = next((m for m in available_models if m['name'] == st.session_state.selected_model), None)
            if selected_info:
                st.write(f"**Name:** {selected_info['name']}")
                st.write(f"**Type:** {selected_info['type']}")
                if st.session_state.current_model:
                    st.success("✅ Model loaded")
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        threshold = st.slider(
            "Real news confidence threshold",
            0.0, 1.0, 0.5, 0.01,
            key="threshold",
            help="Lower values make the model more likely to classify as 'Real'."
        )
        temperature = st.slider(
            "Temperature (higher = less confident)",
            0.1, 5.0, 1.0, 0.1,
            key="temperature",
            help="Higher values make probabilities more uniform. Try 2.0 if the model is overconfident."
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        Insight Guard uses state-of-the-art NLP to detect fake news.  
        Choose a model and analyze any text.
        """)

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📝 Input Text")
    
    example_option = st.selectbox(
        "Load an example",
        ["Select an example...", "Real News", "Fake News", "Mixed Content", "Breaking News"]
    )
    
    example_texts = {
        "Real News": (
            "WASHINGTON (Reuters) - Donald Trump Jr. was told prior to meeting a Russian lawyer whom he believed had information damaging to Hillary Clinton that the material was part of a Russian government effort to help his father’s presidential campaign, the New York Times reported on Monday, citing three people with knowledge of the email. Publicist Rob Goldstone, who helped broker the June 2016 meeting, sent the email to President Donald Trump’s eldest son, the Times said. Goldstone’s message indicates that the Russian government was the source of the potentially damaging information, according to the Times."
        ),
        "Fake News": (
            "LAW ENFORCEMENT ON HIGH ALERT Following Threats Against Cops And Whites On 9-11By #BlackLivesMatter And #FYF911 Terrorists [VIDEO],No comment is expected from Barack Obama Members of the #FYF911 or #FukYoFlag and #BlackLivesMatter movements called for the lynching and hanging of white people and cops. They encouraged others on a radio show Tuesday night to turn the tide and kill white people and cops to send a message about the killing of black people in America.One of the F***YoFlag organizers is called Sunshine. She has a radio blog show hosted from Texas called, Sunshine s F***ing Opinion Radio Show. A snapshot of her #FYF911 @LOLatWhiteFear Twitter page at 9:53 p.m. shows that she was urging supporters to Call now!! #fyf911 tonight we continue to dismantle the illusion of white Below is a SNAPSHOT Twitter Radio Call Invite #FYF911The radio show aired at 10:00 p.m. eastern standard time.During the show, callers clearly call for lynching and killing of white people.A 2:39 minute clip from the radio show can be heard here. It was provided to Breitbart Texas by someone who would like to be referred to as Hannibal. He has already received death threats as a result of interrupting #FYF911 conference calls.An unidentified black man said when those mother f**kers are by themselves, that s when when we should start f***ing them up. Like they do us, when a bunch of them ni**ers takin one of us out, that s how we should roll up. He said, Cause we already roll up in gangs anyway. There should be six or seven black mother f**ckers, see that white person, and then lynch their ass. Let s turn the tables. They conspired that if cops started losing people, then there will be a state of emergency. He speculated that one of two things would happen, a big-ass [R s?????] war, or ni**ers, they are going to start backin up. We are already getting killed out here so what the f**k we got to lose? Sunshine could be heard saying, Yep, that s true. That s so f**king true. He said, We need to turn the tables on them. Our kids are getting shot out here. Somebody needs to become a sacrifice on their side.He said, Everybody ain t down for that s**t, or whatever, but like I say, everybody has a different position of war. He continued, Because they don t give a f**k anyway. He said again, We might as well utilized them for that s**t and turn the tables on these n**ers. He said, that way we can start lookin like we ain t havin that many casualties, and there can be more causalities on their side instead of ours. They are out their killing black people, black lives don t matter, that s what those mother f**kers so we got to make it matter to them. Find a mother f**ker that is alone. Snap his ass, and then f***in hang him from a damn tree. Take a picture of it and then send it to the mother f**kers. We just need one example, and then people will start watchin. This will turn the tables on s**t, he said. He said this will start a trickle-down effect. He said that when one white person is hung and then they are just flat-hanging, that will start the trickle-down effect. He continued, Black people are good at starting trends. He said that was how to get the upper-hand. Another black man spoke up saying they needed to kill cops that are killing us. The first black male said, That will be the best method right there. Breitbart Texas previously reported how Sunshine was upset when racist white people infiltrated and disrupted one of her conference calls. She subsequently released the phone number of one of the infiltrators. The veteran immediately started receiving threatening calls.One of the #F***YoFlag movement supporters allegedly told a veteran who infiltrated their publicly posted conference call, We are going to rape and gut your pregnant wife, and your f***ing piece of sh*t unborn creature will be hung from a tree. Breitbart Texas previously encountered Sunshine at a Sandra Bland protest at the Waller County Jail in Texas, where she said all white people should be killed. She told journalists and photographers, You see this nappy-ass hair on my head? That means I am one of those more militant Negroes. She said she was at the protest because these redneck mother-f**kers murdered Sandra Bland because she had nappy hair like me. #FYF911 black radicals say they will be holding the imperial powers that are actually responsible for the terrorist attacks on September 11th accountable on that day, as reported by Breitbart Texas. There are several websites and Twitter handles for the movement. Palmetto Star describes himself as one of the head organizers. He said in a YouTube video that supporters will be burning their symbols of the illusion of their superiority, their false white supremacy, like the American flag, the British flag, police uniforms, and Ku Klux Klan hoods.Sierra McGrone or Nocturnus Libertus posted, you too can help a young Afrikan clean their a** with the rag of oppression. She posted two photos, one that appears to be herself, and a photo of a black man, wiping their naked butts with the American flag.For entire story: Breitbart News"
        ),
        "Mixed Content": (
            "WASHINGTON (Reuters) - Donald Trump Jr. was told prior to meeting a Russian lawyer whom he believed had information damaging to Hillary Clinton that the material was part of a Russian government effort to help his father’s presidential campaign, the New York Times reported on Monday, citing three people with knowledge of the email. Publicist Rob Goldstone, who helped broker the June 2016 meeting, sent the email to President Donald Trump’s eldest son, the Times said. Goldstone’s message indicates that the Russian government was the source of the potentially damaging information, according to the Times. "
            "Now, viral social media posts are claiming that the email also contained evidence of a secret financial transaction between the Trump campaign and a Russian oligarch, which would prove collusion. Anonymous sources on Twitter have shared screenshots of what they say is the full email, showing wire transfer details and a promise of future payments in exchange for policy concessions. The posts have been shared millions of times. "
            "However, the New York Times has confirmed that the email in question contained no such financial details. The leaked screenshots circulating online are fabricated, according to a statement from the Times. 'We have reviewed the original email obtained by our reporters; it does not include any mention of money or financial transactions,' said a spokesperson. Experts caution that such doctored documents are common in disinformation campaigns aimed at exploiting real news events. While the original story raised serious questions about campaign contacts, the additional allegations of financial dealings are unsubstantiated. "
            "This pattern of mixing real facts with falsehoods highlights the challenge of navigating today’s information environment, where genuine revelations are quickly weaponized with fabricated details."
        ),
        "Breaking News": (
            "SHOCKING: Scientists discover that eating chocolate daily can reverse aging, according to a secret "
            "study funded by Big Pharma! The study, supposedly conducted at a prestigious university, found that "
            "a specific compound in dark chocolate activates telomerase, the enzyme that lengthens telomeres. "
            "The lead researcher, who allegedly was silenced, leaked the findings before the paper was retracted. "
            "Fitness gurus are already promoting 'chocolate detox' programs for $497. But leading experts call "
            "the claims 'pseudoscience.' Dr. Michael Roberts of the American Geriatrics Society said: 'There is "
            "no credible evidence that chocolate reverses aging. These claims are irresponsible and dangerous.' "
            "The FDA has not approved any such therapy, and consumers are warned to be skeptical of miracle cures."
        )
    }
    
    news_text = st.text_area(
        "Paste or type your news article",
        height=300,
        placeholder="Enter news text here...",
        value=example_texts.get(example_option, "")
    )
    
    col_options1, col_options2 = st.columns(2)
    with col_options1:
        analyze_all = st.checkbox("Analyze with all models", value=False)
    with col_options2:
        show_details = st.checkbox("Show detailed analysis", value=True)
    
    analyze_btn = st.button("Analyze News", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Prediction Results")
    
    if analyze_btn:
        st.write("Button clicked")  # for debugging
        st.write(f"Available models: {[m['name'] for m in available_models]}")
        st.write(f"Selected model: {st.session_state.selected_model}")
        st.write(f"Text length: {len(news_text.strip())}")
        
        if not news_text.strip():
            st.warning("Please enter some text to analyze.")
        elif not available_models:
            st.error("No models available.")
        else:
            models_to_analyze = []
            if analyze_all:
                models_to_analyze = available_models
            elif st.session_state.selected_model:
                selected_info = next((m for m in available_models if m['name'] == st.session_state.selected_model), None)
                if selected_info:
                    models_to_analyze = [selected_info]
            else:
                models_to_analyze = [available_models[0]]
            
            results = []
            with st.spinner("Analyzing..."):
                for model_info in models_to_analyze:
                    model_data = load_model(model_info)
                    if model_data:
                        prediction, confidence, real_prob, fake_prob = predict_with_model(model_data, news_text)
                        results.append({
                            'model': model_info['name'],
                            'type': model_info['type'],
                            'prediction': prediction,
                            'confidence': confidence,
                            'real_prob': real_prob,
                            'fake_prob': fake_prob
                        })
            
            st.write(f"Number of results: {len(results)}")
            
            if results:
                if len(results) == 1:
                    res = results[0]
                    pred_class = res['prediction'].lower()
                    if pred_class == "fake":
                        st.markdown(f"""
                        <div class="result-card result-fake">
                            <div style="display: flex; align-items: center;">
                                <div style="font-size: 2rem;">🛑</div>
                                <div style="margin-left: 1rem;">
                                    <div style="font-weight: 700; font-size: 1.25rem;">FAKE NEWS DETECTED</div>
                                    <div style="font-size: 0.8rem; color: #9CA3AF;">Using {res['model']}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card result-real">
                            <div style="display: flex; align-items: center;">
                                <div style="font-size: 2rem;">✅</div>
                                <div style="margin-left: 1rem;">
                                    <div style="font-weight: 700; font-size: 1.25rem;">REAL NEWS DETECTED</div>
                                    <div style="font-size: 0.8rem; color: #9CA3AF;">Using {res['model']}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("#### Confidence")
                    st.markdown(f"<div class=\"confidence-meter\"><div class=\"confidence-fill\" style=\"width: {res['confidence']*100}%;\"></div></div>", unsafe_allow_html=True)
                    st.write(f"**{res['confidence']*100:.1f}%**")
                    
                    col_metrics1, col_metrics2 = st.columns(2)
                    with col_metrics1:
                        st.markdown(f'<div class="metric-value">{res["real_prob"]*100:.1f}%</div><div class="metric-label">Real Probability</div>', unsafe_allow_html=True)
                    with col_metrics2:
                        st.markdown(f'<div class="metric-value">{res["fake_prob"]*100:.1f}%</div><div class="metric-label">Fake Probability</div>', unsafe_allow_html=True)
                
                else:
                    st.subheader("Model Comparison")
                    df_results = pd.DataFrame([{
                        'Model': r['model'],
                        'Type': r['type'],
                        'Prediction': r['prediction'],
                        'Confidence': f"{r['confidence']*100:.1f}%",
                        'Real%': f"{r['real_prob']*100:.1f}%",
                        'Fake%': f"{r['fake_prob']*100:.1f}%"
                    } for r in results])
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
                    fake_count = sum(1 for r in results if r['prediction'] == 'Fake')
                    real_count = sum(1 for r in results if r['prediction'] == 'Real')
                    if fake_count > real_count:
                        st.error(f"🛑 Majority says FAKE ({fake_count}/{len(results)} models)")
                    elif real_count > fake_count:
                        st.success(f"✅ Majority says REAL ({real_count}/{len(results)} models)")
                    else:
                        st.warning(f"⚠️ Split decision ({real_count} real, {fake_count} fake)")
                
                if show_details:
                    with st.expander("Text Analysis Details"):
                        st.write(f"**Text Length:** {len(news_text)} characters")
                        st.write(f"**Word Count:** {len(news_text.split())} words")
                        sensational_words = ['breaking', 'shocking', 'amazing', 'unbelievable', 'must', 'click', 'share', 'secret', 'hidden', 'exposed']
                        found = [w for w in sensational_words if w in news_text.lower()]
                        if found:
                            st.warning(f"⚠️ Sensational words found: {', '.join(set(found))}")
                        else:
                            st.success("✅ No sensational words detected")
    else:
        st.info("Select a model and click 'Analyze News' to see results.")

# Footer
st.markdown("""
<div class="footer">
    <p>Insight Guard | Academic Prototype for Educational Purposes Only</p>
    <p>Always verify information from multiple reliable sources.</p>
</div>
""", unsafe_allow_html=True)