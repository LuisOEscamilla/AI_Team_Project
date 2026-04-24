# Review-of-Reviews (CSCI 4405/5405)
**Team:** Lucas Montoya & Luis Escamilla

### Overview
An AI-driven tool that summarizes large sets of product reviews into clear,
trustworthy insights using three classical AI techniques in a pipeline:

- **Logic Inference** — rule-based topic detection with fuzzy matching and
  negation-aware per-topic sentiment analysis
- **CSP Solver** — backtracking search with forward checking to find a valid
  selection of reviews satisfying user-defined hard constraints
- **A\* Search** — optimizes the final review subset for maximum topic
  coverage and trust using an admissible heuristic

Includes a Streamlit web UI with adjustable constraint "knobs."

---

### Project Structure
AI_Team_Project/
├── data/
│   └── Womens Clothing E-Commerce Reviews.csv   ← download separately
├── src/
│   ├── app.py            ← Streamlit UI (run this)
│   ├── summarizer.py     ← main pipeline (also runnable standalone)
│   ├── csp_solver.py     ← CSP solver with forward checking
│   ├── logic_engine.py   ← topic inference + sentiment analysis
│   ├── trust_score.py    ← trust/suspicion scoring
│   ├── utils.py          ← data loading and text utilities
│   └── test_csp.py       ← standalone CSP unit tests
├── .gitignore
└── README.md

---

### Dataset
Download from Kaggle and place the CSV in the `/data` folder:
https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

The CSV file should be named exactly:
`Womens Clothing E-Commerce Reviews.csv`

The dataset is not included in this repository.

---

### Setup (run once per machine)

**1. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

You will see `(venv)` at the start of your terminal prompt when it is active.
Always activate the venv before running anything.

**2. Install dependencies**
```bash
pip install streamlit pandas kagglehub
```

---

### Running the App (Streamlit UI)
Make sure the venv is active, then from the project root:
```bash
streamlit run src/app.py
```
Opens at `http://localhost:8501` in your browser.

Use the sidebar to adjust constraints and search parameters, then click
**Run Analysis**.

---

### Running the Terminal Pipeline
```bash
python src/summarizer.py
```

---

### Running the CSP Tests
```bash
python src/test_csp.py
```

---

### Dependencies
| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pandas` | Data loading and filtering |
| `kagglehub` | Dataset download (used by summarizer.py only) |

All standard library — no additional installs needed beyond the three above.