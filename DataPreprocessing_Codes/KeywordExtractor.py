import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

def extract_class_keywords(root_folder, n_keywords=15, min_df=2, max_df=0.8):
    """
    Extract unique keywords for each class/subfolder using hybrid TF-IDF + KeyBERT approach
    
    Args:
        root_folder: Path to root directory containing class subfolders
        n_keywords: Number of keywords to extract per class
        min_df: Ignore terms appearing in fewer than this many classes
        max_df: Ignore terms appearing in more than this fraction of classes
    """
    # Initialize NLP tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    kw_model = KeyBERT()
    
    # Load all class data
    class_texts = {}
    class_names = []
    
    # Traverse through each class subfolder
    for class_name in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Read and concatenate all text files in subfolder
        class_content = []
        for filename in os.listdir(class_path):
            if filename.endswith('.txt') and filename != 'keywords.txt':
                filepath = os.path.join(class_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        class_content.append(f.read())
                except:
                    with open(filepath, 'r', encoding='latin-1') as f:
                        class_content.append(f.read())
        
        if class_content:
            full_text = ' '.join(class_content)
            class_texts[class_name] = full_text
            class_names.append(class_name)
    
    # Create corpus for TF-IDF (each class as one document)
    corpus = [class_texts[name] for name in class_names]
    
    # Preprocessing function for vectorizer
    def preprocess(text):
        tokens = re.findall(r'\b\w{3,}\b', text.lower())
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
        return ' '.join(tokens)
    
    # Build TF-IDF matrix across classes
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        stop_words=list(stop_words)
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract keywords for each class
    for class_name in class_names:
        class_text = class_texts[class_name]
        
        # Get top TF-IDF keywords for uniqueness across classes
        class_idx = class_names.index(class_name)
        class_tfidf = tfidf_matrix[class_idx].toarray().flatten()
        top_tfidf_indices = np.argsort(class_tfidf)[-n_keywords*2:][::-1]
        tfidf_keywords = [feature_names[i] for i in top_tfidf_indices]
        
        # Extract context-aware keywords using KeyBERT
        keybert_kws = kw_model.extract_keywords(
            class_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=n_keywords*3,
            use_mmr=True,
            diversity=0.7
        )
        keybert_kws = [kw[0] for kw in keybert_kws]
        
        # Combine and deduplicate keywords (prioritizing KeyBERT context)
        combined_kws = list(set(keybert_kws + tfidf_keywords))
        
        # Score keywords by both relevance and uniqueness
        keyword_scores = {}
        for kw in combined_kws:
            # Context relevance score from KeyBERT
            relevance_score = next((score for k, score in keybert_kws if k == kw), 0)
            
            # Uniqueness score from TF-IDF
            if kw in feature_names:
                idx = np.where(feature_names == kw)[0][0]
                uniqueness_score = class_tfidf[idx]
            else:
                uniqueness_score = 0
            
            # Combined score (weighted toward relevance)
            keyword_scores[kw] = 0.7 * relevance_score + 0.3 * uniqueness_score
        
        # Select top N keywords
        sorted_kws = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        final_keywords = [kw for kw, score in sorted_kws[:n_keywords]]
        
        # Save to keywords.txt in class folder
        output_path = os.path.join(root_folder, class_name, 'keywords.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(final_keywords))
        
        print(f"Generated {len(final_keywords)} keywords for {class_name}")