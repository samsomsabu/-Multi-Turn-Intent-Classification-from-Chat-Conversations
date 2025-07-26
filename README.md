# 🧠 Multi-Turn Intent Classification from Chat Conversations

## 📌 Assignment Objective

The goal of this assignment was to build a system that classifies the **final intent** of a user from a multi-turn, WhatsApp-style conversation. Specifically, the task involves:

* **Extracting user utterances** from conversation histories.
* **Classifying the final intent** using a machine learning model.
* **Providing rationale** for the predicted intent.
* **Exporting predictions** in both JSON and CSV formats.

---

## ✅ Requirements

* Build an **intent classifier** using:

  * Fine-tuned transformer models (e.g., DistilBERT)
  * Zero-shot classification (using `facebook/bart-large-mnli`)
* Predict intent for each conversation in `test_conversations.json`
* Provide human-readable **rationale** for the prediction.
* Output files:

  * `predictions.json`
  * `predictions.csv`
  * `zero_shot_predictions.json` (if applicable)
* Evaluate performance using ground truth via `evaluate.py`

---

## 🛠️ What We Implemented

### 1. **Supervised Model (DistilBERT)**

* Preprocessed multi-turn conversations by extracting **user messages**.
* Created labeled training data (`train.csv`) with curated examples.
* Fine-tuned a **DistilBERT** model for classification.
* Used HuggingFace `Trainer` for training & evaluation.
* Saved model, tokenizer, and label encoder.

### 2. **Zero-Shot Classification (Baseline)**

* Used `facebook/bart-large-mnli` via Hugging Face `pipeline`.
* Defined candidate labels:

  ```
  "Book Appointment", "Product Inquiry", "Pricing Negotiation", 
  "Support Request", "Follow-Up"
  ```
* Applied to unseen conversations (`test_conversations.json`) without retraining.

### 3. **Rationale Generation**

* Simple rule-based rationale generator that:

  * Extracts user text
  * Explains classification using keywords and label context

---

## 📂 Directory Structure

```
.
├── data/
│   ├── train.csv
│   ├── test_conversations.json
│   └── ground_truth.csv
├── model/
│   └── distilbert/ (fine-tuned model + tokenizer)
├── logs/
├── main.py                  # Zero-shot pipeline
├── train_classifier.py      # Fine-tune DistilBERT
├── predict.py               # Predict using trained classifier
├── evaluate.py              # Compare predictions with ground truth
├── rationale.py             # Generate rationale
├── preprocessing.py         # Utility to extract user text
├── zero_shot_model.py       # Zero-shot classifier logic
└── predictions.json
```

---

## 🧪 Observations

### ✅ What Worked

* `predict.py` (DistilBERT) successfully runs and produces output.
* Zero-shot classifier loads and returns predictions.

### ⚠️ Challenges Faced

* **Zero-shot predictions were often inaccurate** due to:

  * Lack of context handling across turns.
  * Ambiguity in short messages.
* **Small dataset (5 examples)** led to poor generalization in fine-tuning.
* Some **labels were ambiguous or overlapping** (e.g., “Follow-Up” vs. “Support Request”).

---

## 🤖 Next Steps / Improvements

* Add more **training samples** across all intents to improve fine-tuning.
* Improve rationale generation using:

  * LLM-based explanation (e.g., GPT-3.5-turbo)
  * Rule-based keyword matching with examples
* Train on **context-aware models** (e.g., Dialogue BERT, T5)
* Apply **ensemble approach**: zero-shot + fine-tuned + rules.

---

## 📝 How to Run

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Train Classifier

```bash
python prepare_dataset.py
python train_classifier.py
```

### 3. Predict Using Fine-Tuned Model

```bash
python predict.py
```

### 4. Predict Using Zero-Shot Model

```bash
python main.py
```

### 5. Evaluate

```bash
python evaluate.py
```

---

## 📊 Example Output

**predictions.json**

```json
[
  {
    "conversation_id": "conv_004",
    "predicted_intent": "Book Appointment"
  }
]
```

**zero\_shot\_predictions.json**

```json
[
  {
    "conversation_id": "conv_003",
    "predicted_intent": "Product Inquiry",
    "rationale": "The intent was classified as 'Product Inquiry' because..."
  }
]
