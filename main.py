import sys
import os
from src.preprocessing import clean_text
from src.matching import compute_similarity

def run_screening(resume_path: str, job_path: str):
    """
    Main entry point for screening a single resume against a job description.
    """
    if not os.path.exists(resume_path) or not os.path.exists(job_path):
        print(f"Error: One of the files does not exist at {resume_path} or {job_path}")
        return

    # 1. Read files
    with open(resume_path, 'r', encoding='utf-8') as f:
        resume_raw = f.read()
    
    with open(job_path, 'r', encoding='utf-8') as f:
        job_raw = f.read()

    # 2. Preprocess
    print("Preprocessing text...")
    resume_clean = clean_text(resume_raw)
    job_clean = clean_text(job_raw)

    # 3. Compute Similarity
    print("Calculating similarity score...")
    score, tfidf_df = compute_similarity(resume_clean, job_clean)

    # 4. Explain Overlap
    # Identify keywords that appear in both (overlap)
    overlap_keywords = tfidf_df.loc[:, (tfidf_df.iloc[0] > 0) & (tfidf_df.iloc[1] > 0)].columns.tolist()
    
    # 5. Output Results
    print("\n" + "="*30)
    print("RESUME SCREENING RESULTS")
    print("="*30)
    print(f"Similarity Score: {score:.2%}")
    print(f"\nMatched Keywords ({len(overlap_keywords)}):")
    if overlap_keywords:
        print(", ".join(overlap_keywords))
    else:
        print("No significant keyword overlap found.")
    
    # Simple recommendation logic
    print("\nConclusion:")
    if score > 0.4:
        print("✅ Strong Match: The candidate has a high keyword overlap.")
    elif score > 0.15:
        print("⚠️ Potential Match: Some relevant keywords found, but review is needed.")
    else:
        print("❌ Low Match: Significant mismatch in keywords.")
    print("="*30)

if __name__ == "__main__":
    # If no paths provided, use samples (to be created next)
    res_path = "data/resumes/sample_resume.txt"
    job_path = "data/jobs/sample_job.txt"
    
    if len(sys.argv) > 2:
        res_path = sys.argv[1]
        job_path = sys.argv[2]
        
    run_screening(res_path, job_path)
