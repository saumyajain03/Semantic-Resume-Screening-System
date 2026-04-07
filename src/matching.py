from typing import Dict, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

model = None  # Global variable for lazy loading

def compute_similarity(resume_text: str, job_description_text: str, skill_weights: Optional[Dict[str, float]] = None):
    """
    Computes similarity between resume and job description using TF-IDF and Cosine Similarity.
    Supports skill weighting to prioritize specific keywords.
    
    Args:
        resume_text: Cleaned resume text.
        job_description_text: Cleaned job description text.
        skill_weights: Dict like {'python': 2.0, 'sql': 1.5} to boost specific words.
        
    Returns:
        similarity_score (float): The cosine similarity score.
        tfidf_df (DataFrame): A dataframe showing the TF-IDF weights.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description_text]).toarray()
    feature_names = vectorizer.get_feature_names_out()

    # Apply Skill Weights if provided
    if skill_weights:
        # We search for each skill in the vocabulary (feature_names)
        for skill, weight in skill_weights.items():
            skill_clean = skill.lower()
            if skill_clean in feature_names:
                # Find the index of the skill in the vector
                idx = list(feature_names).index(skill_clean)
                # Multiply the TF-IDF score by the weight factor
                # This increases the 'importance' of this word in the vector
                tfidf_matrix[:, idx] *= weight

    # Compute Cosine Similarity on the (potentially weighted) vectors
    similarity_score = cosine_similarity([tfidf_matrix[0]], [tfidf_matrix[1]])[0][0]
    
    df = pd.DataFrame(tfidf_matrix, columns=feature_names, index=["Resume", "Job Description"])
    
    return similarity_score, df

def get_model():
    """
    Lazily loads the SentenceTransformer model.
    """
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def compute_semantic_similarity(resume_text: str, job_description_text: str):
    """
    Computes similarity using sentence embeddings for conceptual matching.
    Uses the lightweight 'all-MiniLM-L6-v2' model.
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed. Run 'pip install sentence-transformers'.")

    # Load the model lazily
    model = get_model()

    # Generate Embeddings
    embeddings = model.encode([resume_text, job_description_text])

    # Compute Cosine Similarity
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Semantic matching doesn't have a direct 'keyword' overlap like TF-IDF,
    # so we return an empty DataFrame for compatibility.
    return similarity_score, pd.DataFrame()

if __name__ == "__main__":
    # Quick manual test with weights
    res = "python developer machine learning"
    job = "hiring python developer familiar with machine learning"
    weights = {"python": 5.0} # Heavily weight python
    
    score, df = compute_similarity(res, job, skill_weights=weights)
    print(f"Weighted Similarity Score (Python x5): {score:.4f}")
    
    # Show columns where both documents have non-zero values
    overlap = df.loc[:, (df.iloc[0] > 0.05) & (df.iloc[1] > 0.05)]
    print("\nOverlap with applied weights:")
    print(overlap)
