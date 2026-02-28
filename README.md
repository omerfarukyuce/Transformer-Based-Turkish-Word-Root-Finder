<div align="center">

# 🔤🧠Transformer-Based Turkish Words' Root Finder🪵

### Transformer-Based Turkish Morphological Root Extraction

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A deep learning project that automatically extracts **roots** and **suffixes** from Turkish words using an **Encoder–Decoder Transformer** architecture.

[Features](#-features) · [Architecture](#-model-architecture) · [Dataset](#-dataset) · [Setup](#-setup) · [Usage](#-usage) · [Results](#-results)

</div>

---

## 📌 About

Turkish is an **agglutinative** language with a rich suffix system — a single word can carry multiple suffixes, dramatically changing its meaning. This project takes any Turkish word and:

- **Identifies its root** (stem)
- **Separates its suffixes**

Examples:

| Word | Root | Suffixes |
|---|---|---|
| `alıyorum` | `al` (take) | `ıyorum` |
| `görmediği` | `gör` (see) | `mediği` |
| `arkadaşları` | `arkadaş` (friend) | `ları` |
| `öğretmenlerimde` | `öğret` (teach) | `menlerimde` |
| `gülümseyerek` | `gül` (laugh) | `ümseyerek` |

---

## ✨ Features

- 🔤 **Character-level Seq2Seq** — Processes words character by character to generate the root
- 🧠 **Transformer Encoder–Decoder** — Modern architecture based on Multi-Head Attention
- 🔍 **Beam Search & Greedy Decoding** — Comparison of two decoding strategies
- 📊 **Comprehensive EDA** — Word length distributions, most/least frequent roots, suffix analysis
- 🎯 **Attention Visualization** — Heatmaps showing which characters the model focuses on
- 📈 **Character-level Confusion Matrix** — Per-character error analysis
- 🔧 **Post-processing** — Root correction and frequency-based refinement
- 📦 **Export** — Root frequencies, character vocabulary, and inverse vocabulary outputs

---

## 🏗 Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   TRANSFORMER MODEL                     │
│                                                         │
│  ┌──────────────────┐      ┌──────────────────────┐     │
│  │     ENCODER       │      │      DECODER          │     │
│  │                  │      │                      │     │
│  │  Character Embed │      │  Character Embed     │     │
│  │  + Positional Enc│      │  + Positional Enc    │     │
│  │        ↓         │      │        ↓             │     │
│  │  ┌────────────┐  │      │  ┌────────────────┐  │     │
│  │  │ Multi-Head │  │      │  │ Masked MHA     │  │     │
│  │  │ Attention  │  │ ───→ │  │ + Cross MHA    │  │     │
│  │  └────────────┘  │      │  └────────────────┘  │     │
│  │  + LayerNorm     │      │  + LayerNorm         │     │
│  │  + FFN           │      │  + FFN               │     │
│  │  × 2 Layers      │      │  × 2 Layers          │     │
│  └──────────────────┘      └──────────────────────┘     │
│                                    ↓                    │
│                            Dense (Softmax)              │
│                                    ↓                    │
│                           Predicted Root                │
└─────────────────────────────────────────────────────────┘
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Embedding Dimension | 128 |
| Number of Attention Heads | 4 |
| Key Dimension | 32 |
| Encoder Layers | 2 |
| Decoder Layers | 2 |
| Optimizer | Adam (Cosine Decay) |
| Random State | 42 |

---

## 📁 Dataset

The project uses a custom dataset containing **18,545** Turkish words.

**Format:** `word,root,suffixes` (CSV)

```csv
word,root,suffixes
alıyorum,al,ıyorum
görmek,gör,mek
şiddetli,şiddet,li
arkadaşları,arkadaş,ları
```

### Dataset Statistics

- 📝 **18,545** unique word–root–suffix triplets
- 🔤 3 columns: `word`, `root`, `suffixes`
- 🇹🇷 Turkish-specific characters: ç, ğ, ı, ö, ş, ü
- 📊 Root length distributions, suffix distributions, and frequency analyses are detailed in the notebook

---

## 🚀 Setup

### Requirements

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

### Running on Kaggle (Recommended)

1. Sign in to [Kaggle](https://www.kaggle.com)
2. Create a **New Notebook**
3. Upload `turkish-words-roots-suffixes.csv` as a **Dataset**
4. Import the `transformer-based-turkish-words-root-finder.ipynb` notebook
5. Enable **GPU accelerator** (Settings → Accelerator → GPU)
6. Run all cells sequentially

### Running Locally

```bash
# Clone the repo
git clone [https://github.com/<your-username>/turkish-root-finder.git](https://github.com/omerfarukyuce/Transformer-Based-Turkish-Word-Root-Finder.git)
cd turkish-root-finder

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn

# Launch Jupyter Notebook
jupyter notebook
```

---

## 💡 Usage

### Notebook Workflow

The notebook executes the following steps in order:

1. **📥 Data Loading** — Read and preprocess the CSV dataset
2. **📊 Exploratory Data Analysis (EDA)**
   - Word / root / suffix length distributions
   - Most frequent and rarest roots
   - No-suffix words and suffix length buckets
3. **🔨 Model Building** — Define the Encoder–Decoder Transformer network
4. **🏋️ Training** — Train the model with checkpoint saving
5. **📈 Evaluation**
   - Character-level Confusion Matrix
   - Greedy vs Beam Search comparison
   - Rare vs Frequent root performance
   - Word-level Exact Match metric
6. **🔍 Attention Visualization** — Encoder–Decoder attention heatmaps
7. **🔧 Post-processing** — Root correction mechanism
8. **📦 Export** — Root frequencies, character vocabulary, and inverse vocabulary

### Example Prediction

After the model is trained, you can test words in the **Model Testing with Sample Words** section:

```python
# Greedy Decoding
predicted_root = predict_greedy("çalışıyorum")
# → "çalış"

# Beam Search Decoding
predicted_root = predict_beam("öğretmenlerimde")
# → "öğret"
```

---

## 📊 Results

The notebook provides the following metrics after training:

| Metric | Description |
|---|---|
| **Greedy Accuracy** | Word-level accuracy using greedy decoding |
| **Beam Search Accuracy** | Word-level accuracy using beam search |
| **Beam vs Greedy Gain** | Improvement of beam search over greedy |
| **Frequent vs Rare Gap** | Performance difference between frequent and rare roots |

> 💡 Run the notebook to see detailed results and visualizations.

---

## 📂 Project Structure

```
turkish-root-finder/
│
├── 📓 transformer-based-turkish-words-root-finder (5).ipynb  # Main notebook
├── 📊 turkish-words-roots-suffixes.csv                       # Dataset (18,545 words)
├── 📑 dataset-settings.xlsx                                  # Dataset (Excel format)
├── 🔄 convert_excel_to_csv.py                                # Excel → CSV converter
├── 🔍 analyze_notebooks.py                                   # Notebook comparison tool
├── ⚡ optimize_notebook.py                                   # Notebook optimization script
└── 📖 README.md                                              # This file
```

---

## 🛠 Utility Scripts

| Script | Description |
|---|---|
| `convert_excel_to_csv.py` | Converts the Excel file to CSV. Automatically fixes Turkish character issues (DOĞRU/YANLIŞ → TRUE/FALSE conversion by Excel). |
| `analyze_notebooks.py` | Compares different notebook versions and lists added/removed sections. |
| `optimize_notebook.py` | Adds caching and runtime optimizations to the notebook. |

---

## 🧪 Technical Details

### Data Preprocessing
- Words are tokenized at the character level
- Special tokens are used: `<` (START), `>` (END), `<PAD>` (padding)
- Duplicate records are removed

### Training Strategy
- **Cosine Decay** learning rate schedule
- **Checkpoint** saving for best weights
- Train/Test split (Random State = 42)

### Evaluation
- **Greedy Decoding**: Selects the highest-probability character at each step
- **Beam Search**: Evaluates multiple candidate paths in parallel
- **Post-processing**: Improves results using root frequency information and derivational suffix rules

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please:

1. **Fork** this repository
2. Create a new **branch** (`git checkout -b feature/new-feature`)
3. **Commit** your changes (`git commit -m 'Add new feature'`)
4. **Push** your branch (`git push origin feature/new-feature`)
5. Open a **Pull Request**

---

<div align="center">
   
## 🔍 Code And Kaggle Link
Project: [transformer-based-turkish-words-root-finder.ipynb](https://github.com/omerfarukyuce/Transformer-Based-Turkish-Word-Root-Finder/blob/main/transformer-based-turkish-words-root-finder.ipynb)

Kaggle: [🔤🧠Transformer-Based Turkish Words' Root Finder🪵](https://www.kaggle.com/code/merfarukyce/transformer-based-turkish-words-root-finder)

## 📊 Datasets
Dataset: [Turkish words-roots-suffixes](https://www.kaggle.com/datasets/merfarukyce/turkish-words)

</div>




