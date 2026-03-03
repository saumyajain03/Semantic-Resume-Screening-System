from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def compute_similarity(resume_text: str, job_description_text: str):
    """
    Computes similarity between resume and job description using TF-IDF and Cosine Similarity.
    Returns:
        similarity_score (float): The cosine similarity score.
        tfidf_df (DataFrame): A dataframe showing the TF-IDF weights for overlap analysis.
    """
    # Create the vectorizer
    # We use a simple vectorizer that works on the cleaned text phrases
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the documents
    # Document 1: Resume, Document 2: Job Description
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description_text])
    
    # Compute Cosine Similarity
    # tfidf_matrix[0:1] is the first row (resume), tfidf_matrix[1:2] is the second (job)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Create a DataFrame to see the top keywords for overlap analysis later
    feature_names = vectorizer.get_feature_names_out()
    tfidf_data = tfidf_matrix.toarray()
    
    df = pd.DataFrame(tfidf_data, columns=feature_names, index=["Resume", "Job Description"])
    
    return similarity_score, df

if __name__ == "__main__":
    # Quick manual test
    res = "python developer machine learning data science"
    job = "hiring python developer familiar with machine learning and data science"
    
    score, df = compute_similarity(res, job)
    print(f"Similarity Score: {score:.4f}")
    print("\nTop Keywords (TF-IDF overlap):")
    # Show columns where both documents have non-zero values
    overlap = df.loc[:, (df.iloc[0] > 0.05) & (df.iloc[1] > 0.05)]
    print(overlap)
