from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from src.preprocessing import clean_text
from src.matching import compute_similarity, compute_semantic_similarity, get_model
from src.ner import extract_entities
import math
from threading import Thread

app = FastAPI(
    title="AI Resume Screening API",
    description="Backend API for AI-powered resume matching and entity extraction.",
    version="1.0.0"
)

# --- Pydantic Models for Request/Response Validation ---

class ResumeInput(BaseModel):
    id: str
    text: str

class MatchRequest(BaseModel):
    job_description: str
    resumes: List[ResumeInput]
    weights: Optional[Dict[str, float]] = None

class MatchResult(BaseModel):
    id: str
    score: float
    matched_keywords: Optional[List[str]] = []

class AnalyzeRequest(BaseModel):
    text: str

class EntityResponse(BaseModel):
    organizations: List[str]
    locations: List[str]
    tech_skills: List[str]

# --- Default Global Settings ---
DEFAULT_SKILL_WEIGHTS = {
    "python": 2.5,
    "django": 2.0,
    "machine": 1.5,
    "learning": 1.5,
    "sql": 1.5,
    "java": 1.2
}

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint to ensure the API is running correctly."""
    return {"status": "ok", "message": "API is up and running"}

@app.get("/api/v1/config/weights")
async def get_weights():
    """Retrieve current default skill weights."""
    return {"weights": DEFAULT_SKILL_WEIGHTS}

@app.post("/api/v1/match/tfidf", response_model=List[MatchResult])
async def match_tfidf(request: MatchRequest):
    """
    Computes classical ML similarity using TF-IDF and Cosine Similarity.
    Allows passing custom weights to prioritize certain keywords.
    """
    if not request.job_description or not request.resumes:
        raise HTTPException(status_code=400, detail="Job description and at least one resume must be provided.")

    job_clean = clean_text(request.job_description)
    active_weights = request.weights if request.weights is not None else DEFAULT_SKILL_WEIGHTS

    results = []
    for resume in request.resumes:
        try:
            resume_clean = clean_text(resume.text)
            score, tfidf_df = compute_similarity(resume_clean, job_clean, skill_weights=active_weights)
            
            # Extract common keywords.
            # Filter where both are > 0.
            keywords = tfidf_df.loc[:, (tfidf_df.iloc[0] > 0) & (tfidf_df.iloc[1] > 0)].columns.tolist()
            
            # Handle potential NaN from cosine similarity if vectors are pure zeros
            final_score = float(score) if not math.isnan(score) else 0.0

            results.append({
                "id": resume.id,
                "score": final_score,
                "matched_keywords": keywords
            })
        except Exception as e:
            # Re-raise standard exceptions or log them
            raise HTTPException(status_code=500, detail=f"Error processing resume {resume.id}: {str(e)}")

    # Sort results highest to lowest
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

@app.post("/api/v1/match/semantic", response_model=List[MatchResult])
async def match_semantic(request: MatchRequest):
    """
    Computes deep-learning similarity using SentenceTransformers embeddings.
    """
    if not request.job_description or not request.resumes:
        raise HTTPException(status_code=400, detail="Job description and at least one resume must be provided.")

    job_clean = clean_text(request.job_description)

    results = []
    for resume in request.resumes:
        try:
            resume_clean = clean_text(resume.text)
            score, _ = compute_semantic_similarity(resume_clean, job_clean)
            
            final_score = float(score) if not math.isnan(score) else 0.0
            
            results.append({
                "id": resume.id,
                "score": final_score,
                "matched_keywords": [] # Semantic mode doesn't yield discrete keywords
            })
        except ImportError:
            raise HTTPException(status_code=500, detail="sentence-transformers is not installed/loaded.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing resume {resume.id}: {str(e)}")

    # Sort results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

@app.post("/api/v1/analyze/entities", response_model=EntityResponse)
async def analyze_entities(request: AnalyzeRequest):
    """
    Extracts organizations, locations, and tech skills from raw text using custom Spacy NER.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided for analysis.")

    try:
        # Note: We pass raw text, NOT cleaned text to NER, because capitalization helps Spacy's models
        entities = extract_entities(request.text)
        return entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during entity extraction: {str(e)}")

def preload_model():
    """
    Preloads the SentenceTransformer model in a background thread.
    """
    get_model()

# Start the model preload thread
Thread(target=preload_model).start()

if __name__ == "__main__":
    import uvicorn
    # To run this script directly for debugging
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, workers=2)
