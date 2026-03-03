import sys
import os
import argparse
from typing import Dict, List, Optional
from src.preprocessing import clean_text
from src.matching import compute_similarity, compute_semantic_similarity

# Default Priority Skills for Weighting (TF-IDF Mode only)
DEFAULT_SKILL_WEIGHTS = {
    "python": 2.5,
    "django": 2.0,
    "machine": 1.5,
    "learning": 1.5,
    "sql": 1.5,
    "java": 1.2
}

def display_results(results: List[Dict], mode: str, top_n: int = 1):
    """
    Prints a professional tabular summary of the matching results.
    """
    if not results:
        print("\nNo results to display.")
        return

    print("\n" + "="*60)
    print(f"RESUME SCREENING RANKING (Mode: {mode.upper()})")
    print("="*60)
    print(f"{'RANK':<5} | {'RESUME FILENAME':<30} | {'SCORE':<10}")
    print("-" * 60)
    
    for i, res in enumerate(results, 1):
        print(f"{i:<5} | {res['filename']:<30} | {res['score']:>8.2%}")
    print("="*60)

    # Detailed view for top matches
    for i in range(min(top_n, len(results))):
        top = results[i]
        print(f"\n[Rank {i+1}] Detail: {top['filename']}")
        
        if mode == "tfidf":
            matched_str = ", ".join(top['keywords']) if top['keywords'] else "None"
            print(f"Matched Keywords: {matched_str}")
        else:
            print("Note: Semantic mode provides conceptual similarity (no keyword list).")

        print(f"Status: ", end="")
        if top['score'] > 0.4:
            print("✅ Strong Match")
        elif top['score'] > 0.15:
            print("⚠️ Potential Match")
        else:
            print("❌ Low Match")
    print("="*60 + "\n")

def run_screening(resume_input: str, job_path: str, mode: str, weights: Optional[Dict] = None):
    """
    Orchestrates the resume screening process.
    """
    if not os.path.exists(job_path):
        print(f"Error: Job description not found at {job_path}")
        return

    # 1. Identify all resume files (.txt supported in v1)
    resume_files = []
    if os.path.isdir(resume_input):
        resume_files = [os.path.join(resume_input, f) for f in os.listdir(resume_input) if f.endswith('.txt')]
    elif os.path.isfile(resume_input):
        resume_files = [resume_input]
    else:
        print(f"Error: Invalid resume path: {resume_input}")
        return

    if not resume_files:
        print("No resumes found to process.")
        return

    # 2. Load and Preprocess Job Description
    try:
        with open(job_path, 'r', encoding='utf-8') as f:
            job_raw = f.read()
        job_clean = clean_text(job_raw)
    except Exception as e:
        print(f"Error reading job description: {e}")
        return

    print(f"Analyzing {len(resume_files)} resume(s) using {mode.upper()} engine...")

    results = []
    for res_path in resume_files:
        try:
            with open(res_path, 'r', encoding='utf-8') as f:
                resume_raw = f.read()
            
            resume_clean = clean_text(resume_raw)
            
            if mode == "semantic":
                score, _ = compute_semantic_similarity(resume_clean, job_clean)
                keywords = []
            else:
                score, tfidf_df = compute_similarity(resume_clean, job_clean, skill_weights=weights)
                # Filter for non-zero overlap in both docs
                keywords = tfidf_df.loc[:, (tfidf_df.iloc[0] > 0) & (tfidf_df.iloc[1] > 0)].columns.tolist()

            results.append({
                'filename': os.path.basename(res_path),
                'score': score,
                'keywords': keywords
            })
        except Exception as e:
            print(f"Skipping {res_path} due to error: {e}")

    # 3. Rank by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 4. Display Final Report
    display_results(results, mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Resume Screening & Ranking System")
    parser.add_argument("--resumes", type=str, default="data/resumes/", help="Path to a resume file or directory of .txt resumes")
    parser.add_argument("--job", type=str, default="data/jobs/sample_job.txt", help="Path to the job description .txt file")
    parser.add_argument("--mode", type=str, choices=["tfidf", "semantic"], default="tfidf", help="Similarity engine: 'tfidf' (keyword) or 'semantic' (conceptual)")
    parser.add_argument("--no-weights", action="store_true", help="Disable skill weighting in TF-IDF mode")

    args = parser.parse_args()

    weights = None if (args.no_weights or args.mode == "semantic") else DEFAULT_SKILL_WEIGHTS
    
    run_screening(args.resumes, args.job, mode=args.mode, weights=weights)

