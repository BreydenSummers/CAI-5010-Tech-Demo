# 🤖 CAI-5010 Tech Demo: Image-to-Insight Pipeline

The system is composed of:

- 🧠 **ML-Based Image Feature Extractor**: A machine learning pipeline that extracts structured data from images.
- 📊 **Data Analyzer**: A downstream processor that performs analysis on extracted features — including potential LLM integration.
- 🧵 **Controller Script**: Orchestrates the entire end-to-end process from raw image to analytical output.

---

## 🗂️ Project Structure

```
CAI-5010-Tech-Demo/
├── controller.py              # Entry point to run the full pipeline
├── extractor/
│   ├── __init__.py
│   ├── extract.py             # ML pipeline for feature extraction
│   └── model/                 # Trained models or checkpoints
├── analyzer/
│   ├── __init__.py
│   ├── analyze.py             # Performs data analysis and LLM interaction
│   └── visualizer.py          # (Optional) Generates visual reports
├── data/
│   ├── raw/                   # Input images
│   └── processed/             # ML-generated features
├── results/
│   └── analysis_output.json   # Final output from analyzer
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone git@github.com:your-username/CAI-5010-Tech-Demo.git
```

### 2. Install Dependencies

Set up a virtual environment and install required libraries:

```bash
pip install -r requirements.txt
```

---

## 🧩 Components

### 🔬 Extractor Module (`extractor/`)

This module uses machine learning to:

- Load raw images
- Extract structured features via a trained model
- Save results to `data/processed/`

Run standalone:

```bash
python extractor/extract.py --input data/raw/image.jpg --output data/processed/data.json
```

> 💡 Trained models are located in `extractor/model/`. You may replace or retrain as needed.

---

### 📊 Analyzer Module (`analyzer/`)

This module:

- Reads the output of the extractor
- Runs analytical routines and uses LLMs for interpretation
- Outputs results as structured JSON or visualization

Run standalone:

```bash
python analyzer/analyze.py --input data/processed/data.json --output results/analysis_output.json
```

---

### 🧵 Controller (`controller.py`)

Combines the full pipeline into a single command:

```bash
python controller.py --image data/raw/image.jpg
```

Steps:

1. Calls the ML extractor
2. Passes features to the analyzer
3. Saves outputs in `results/`
