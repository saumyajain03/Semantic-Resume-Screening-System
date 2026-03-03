# AI Resume Screening & Ranking System 🚀

A modular NLP-driven tool to screen and rank resumes against job descriptions. Built for recruiters to handle high volumes of applications with transparency and precision.

## 🌟 Key Features
- **Multi-Modal Matching**:
  - **TF-IDF Engine**: Exact keyword matching with custom skill-weighting (High Precision).
  - **Semantic Engine**: Sentence embeddings (BERT-based) to understand conceptual overlaps (e.g., "AI" vs "Artificial Intelligence").
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
│   ├── resumes/            # Directory for candidate resumes (.txt)
│   └── jobs/               # Directory for job descriptions (.txt)
├── src/
│   ├── preprocessing.py    # Text cleaning (stopwords, noise removal)
│   └── matching.py         # TF-IDF and Semantic similarity engines
├── main.py                # Command-line interface & orchestration
└── requirements.txt       # Dependencies
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

## 📊 Example Output
```text
============================================================
RESUME SCREENING RANKING (Mode: TFIDF)
============================================================
RANK  | RESUME FILENAME                | SCORE     
------------------------------------------------------------
1     | sample_resume.txt              |   61.46%
2     | resume_mid.txt                 |   11.87%
3     | resume_bad.txt                 |    4.82%
============================================================
```

## 🧠 Design Philosophy
This system addresses the "Black Box" problem in AI screening. By providing both a **Keyword matching** mode (for strict requirements) and a **Semantic mode** (for talent discovery), it gives recruiters a balanced toolkit.
