import re
import numpy as np
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure required NLTK data is downloaded
nltk.download('stopwords')
nltk.download('wordnet')

def extract_keywords_from_string(text_data: str, n_keywords: int = 15):
    """
    Extract top keywords from a single string using TF-IDF + KeyBERT
    
    Args:
        text_data (str): Input string data
        n_keywords (int): Number of final keywords to return
        
    Returns:
        List[str]: Final ranked keywords
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    kw_model = KeyBERT()

    # Preprocess function
    def preprocess(text):
        tokens = re.findall(r'\b\w{3,}\b', text.lower())
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
        return ' '.join(tokens)
    
    preprocessed_text = preprocess(text_data)
    
    # TF-IDF with just one document (works poorly alone, but still informative)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stop_words
    )
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().flatten()
    tfidf_top_indices = np.argsort(tfidf_scores)[-n_keywords*2:][::-1]
    tfidf_keywords = [feature_names[i] for i in tfidf_top_indices if tfidf_scores[i] > 0]

    # KeyBERT keywords (contextual)
    keybert_kws = kw_model.extract_keywords(
        text_data,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=n_keywords * 3,
        use_mmr=True,
        diversity=0.7
    )
    keybert_kws = [(kw[0], kw[1]) for kw in keybert_kws]

    # Combine & score both sets
    combined_kws = list(set([k for k, _ in keybert_kws] + tfidf_keywords))
    keyword_scores = {}

    for kw in combined_kws:
        relevance_score = next((score for k, score in keybert_kws if k == kw), 0)
        uniqueness_score = 0
        if kw in feature_names:
            idx = np.where(feature_names == kw)[0][0]
            uniqueness_score = tfidf_scores[idx]
        keyword_scores[kw] = 0.7 * relevance_score + 0.3 * uniqueness_score

    final_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in final_keywords[:n_keywords]]
