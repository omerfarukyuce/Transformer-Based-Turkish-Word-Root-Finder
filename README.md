# 🔤🧠Transformer-Based Turkish Words' Root Finder🪵

## 📖 Overview
This project implements a Transformer-based model for identifying the root forms of Turkish words. The model is trained on a dataset of Turkish words along with their corresponding roots and suffixes, using a sequence-to-sequence architecture with attention mechanisms.

## 🚀 Features

- **Character-level Transformer model** for Turkish root finding
- Support for **greedy search** and **beam search** decoding
- Comprehensive **data preprocessing** and **exploratory data analysis**
- **Model checkpointing** and **early stopping** for better training
- **Learning rate scheduling** for improved convergence
- Detailed **training metrics** visualization

## 📊 Dataset

This project uses a comprehensive dataset of Turkish words with their morphological analysis. The dataset is particularly valuable for understanding Turkish morphology and building NLP applications for agglutinative languages.

### Dataset Details
- **Total words**: 10,027 unique Turkish words
- **Language**: Turkish (Türkçe)
- **Format**: CSV with three columns: `word`, `root`, and `suffixes`
- **Source**: Custom compiled dataset of Turkish words and their morphological breakdowns

### Dataset Statistics
- **Total words**: 10,027
- **Words with suffixes**: 8,119 (80.97%)
- **Words without suffixes**: 1,908 (19.03%)
- **Maximum word length**: 34 characters
- **Maximum root length**: 11 characters
- **Maximum suffix length**: 27 characters

### Character Set
Turkish alphabet characters used in the dataset:
- Lowercase: a, b, c, ç, d, e, f, g, ğ, h, ı, i, j, k, l, m, n, o, ö, p, r, s, ş, t, u, ü, v, y, z
- Special tokens: `<` (start), `>` (end), `<PAD>`, `<UNK>`

### Data Examples
| Word          | Root     | Suffix  |
|---------------|----------|---------|
| alıyorum     | al       | ıyorum  |
| görmek       | gör      | mek     |
| dersin       | ders     | in      |
| şiddetli     | şiddet   | li      |
| yumurtalar   | yumurta  | lar     |

## 🏋️‍♂️ Training the Model

1. **Prepare your dataset**
   - Place your dataset in the root directory as `turkish-words-roots-suffixes.csv`
   - The CSV should have columns: `word`, `root`, `suffixes`

2. **Run the training script**
   ```bash
   python train.py
   ```

   Training parameters can be modified in the script:
   - `NUM_ENCODER_LAYERS`: Number of encoder layers
   - `NUM_DECODER_LAYERS`: Number of decoder layers
   - `EMBED_DIM`: Embedding dimension
   - `BATCH_SIZE`: Training batch size
   - `EPOCHS`: Number of training epochs

## 🧪 Model Inference

### Greedy Search
```python
predicted_root = predict_root_greedy("görüyordum")
print(predicted_root)  # Output: gör
```

### Beam Search
```python
predicted_root = beam_search_decode(model, "görüyordum", beam_size=3)
print(predicted_root)  # Output: gör
```

## 📈 Performance

The model achieves the following performance metrics:
- **Training Accuracy**: ~44.1%
- **Validation Accuracy**: ~43.3%
- **Training Loss**: ~0.68
- **Validation Loss**: ~0.74


### Data Directory
You'll need to create a `data` directory and place your `turkish-words-roots-suffixes.csv` file there. The CSV should have the following columns:
- `word`: The complete Turkish word
- `root`: The root/stem of the word
- `suffixes`: The suffix(es) attached to the root

## 🎯 Example Usage

```python
# Example words to test
words = ["evimdeyken", "kitapta", "görüyordum", "koşacaklarmış", "bulutlardayım"]

print("\nGreedy predictions:")
for w in words:
    print(f"{w} → {predict_root_greedy(w)}")

print("\nBeam search predictions (beam_size=3):")
for w in words:
    print(f"{w} → {beam_search_decode(model, w, beam_size=3)}")
```

## 📝 Notes

- The model uses character-level tokenization to handle the agglutinative nature of Turkish
- Special tokens are used for padding (`<PAD>`), unknown characters (`<UNK>`), and sequence boundaries (`<` and `>`)
- The implementation includes label smoothing to improve generalization

## 📜 License

This project is licensed under the [MIT](LICENSE) License.

## 🔍 Code And Kaggle Link
Project: [transformer-based-turkish-words-root-finder.ipynb](https://github.com/omerfarukyuce/Transformer-Based-Turkish-Word-Root-Finder/blob/main/transformer-based-turkish-words-root-finder.ipynb)

Kaggle: [🔤🧠Transformer-Based Turkish Words' Root Finder🪵](https://www.kaggle.com/code/merfarukyce/transformer-based-turkish-words-root-finder)

## 📊 Datasets
Dataset: [Turkish words-roots-suffixes](https://www.kaggle.com/datasets/merfarukyce/turkish-words)
