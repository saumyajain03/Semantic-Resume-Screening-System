# AI Resume Screening & Ranking System 🚀

A modular NLP-driven tool to screen and rank resumes against job descriptions. Built for recruiters to handle high volumes of applications with transparency and precision.

## 🌟 Key Features
- **Multi-Modal Matching**:
  - **TF-IDF Engine**: Exact keyword matching with custom skill-weighting (High Precision).
  - **Semantic Engine**: Sentence embeddings (BERT-based) to understand conceptual overlaps (e.g., "AI" vs "Artificial Intelligence").
- **Named Entity Recognition (NER)**: Automatically extracts Organizations (Universities/Companies) and Locations to provide a structured candidate summary.
- **Batch Processing**: Rank an entire folder of resumes in seconds.
- **Explainable Scores**: Lists matched keywords for TF-IDF results.
- **Configurable Priority**: Boost specific "must-have" skills to influence rankings.

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **NLP**: NLTK (Text cleaning), Scikit-Learn (TF-IDF & Cosine Similarity)
- **Deep Learning**: Sentence-Transformers (BERT Embeddings)
- **Data Handling**: Pandas

## 📁 Project Structure
```text
resume_screening/
├── data/
│   ├── resumes/            # Candidate resumes (.txt)
│   └── jobs/               # Job descriptions (.txt)
├── src/
│   ├── preprocessing.py    # Text cleaning logic
│   ├── matching.py         # TF-IDF & Semantic engines
│   └── ner.py              # Named Entity Recognition
├── screenshots/            # Visual demo images
├── main.py                # Professional CLI orchestration
└── requirements.txt       # Project dependencies
```

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Saumyajain0003/Resume-Screening-and-Job-Matching-System
cd Resume-Screening-and-Job-Matching-System

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Usage

**Keyword Match (Default):**
```bash
python3 main.py --resumes data/resumes/ --job data/jobs/sample_job.txt
```

**Semantic Match (Conceptual):**
```bash
python3 main.py --mode semantic --resumes data/resumes/ --job data/jobs/sample_job.txt
```

## 📊 Demo & Verification

### 1. Keyword Matching (TF-IDF)
Exact technical match identifying skills like Python, SQL, and Git.
![TF-IDF Match](screenshots/Screenshot%202026-03-03%20at%206.39.31 PM.png)

### 2. Semantic Matching (BERT)
Conceptual matching where the system understands that "AI" and "Artificial Intelligence" are related.
![Semantic Match](screenshots/Screenshot%202026-03-03%20at%206.40.37 PM.png)

### 3. Batch Ranking
The system ranking multiple resumes simultaneously and flagging the best candidates.
![Batch Ranking](screenshots/Screenshot%202026-03-03%20at%206.41.26 PM.png)

### 4. Named Entity Recognition (NER)
**The "Contextual Insight" Feature**: Even if a resume score is low due to keyword mismatches, the AI extracts key entities like **Organizations** and **Locations**. This ensures that high-potential candidates (e.g., those from Microsoft/Google) are caught even if their resume isn't keyword-optimized.
![NER Extraction](screenshots/Screenshot%202026-03-03%20at%207.30.12 PM.png)

## 🧠 Design Philosophy
This system addresses the "Black Box" problem in AI screening. By providing both a **Keyword matching** mode (for strict requirements) and a **Semantic mode** (for talent discovery), it gives recruiters a balanced toolkit.

## ⚡ Challenges Faced

Building this project involved overcoming several engineering challenges:

1. **Model Initialization Time**:
   - The `sentence-transformers` package and HuggingFace models are computationally heavy, leading to long API startup times. This was mitigated by implementing lazy loading and preloading models in background threads.

2. **Balancing Precision and Performance**:
   - Achieving high precision in resume-job matching required combining TF-IDF for exact matches and semantic embeddings for conceptual overlaps. Optimizing this hybrid approach while maintaining performance was a key focus.

3. **Scalability**:
   - Processing large batches of resumes efficiently required careful memory management and leveraging multi-threading for parallelism.

4. **Explainability**:
   - Providing recruiters with explainable scores for matching results was critical. This required designing clear and interpretable outputs for both TF-IDF and semantic engines.

5. **Error Handling**:
   - Ensuring robust error handling for diverse resume formats and incomplete job descriptions was essential to maintain reliability.

These challenges were addressed with a focus on engineering best practices, making the system robust and production-ready.

## Author:
Saumya Jain