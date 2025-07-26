Great! Here's a complete `README.md` tailored to your **Multi-Turn Intent Classification from Chat Conversations** project. It clearly explains the task, your approach, and the challenges with zero-shot classification:

---

```markdown
# 🧠 Multi-Turn Intent Classification from Chat Conversations

This project implements a **multi-turn intent classification system** designed to identify user intent from chat-like conversations (e.g., WhatsApp-style threads). It was developed as part of a machine learning assignment and explores both **fine-tuned classification** and **zero-shot learning** approaches.

---

## 📌 Objective

Classify the final **intent of the user** from a conversation involving multiple back-and-forth messages.

### 🎯 Target Intents
- Book Appointment
- Product Inquiry
- Pricing Negotiation
- Support Request
- Follow-Up

---

## 🗂️ Project Structure

```

├── data/
│   ├── sample\_conversations.json       # Sample training conversations
│   ├── test\_conversations.json         # Unlabeled test conversations
│   ├── ground\_truth.csv                # Ground truth labels for test set
│   └── train.csv                       # Final dataset for training
├── model/
│   └── distilbert/                     # Fine-tuned model and label encoder
├── main.py                             # Zero-shot intent predictor
├── predict.py                          # Fine-tuned model inference
├── prepare\_dataset.py                  # Converts data to training format
├── train\_classifier.py                 # Fine-tunes DistilBERT classifier
├── evaluate.py                         # Generates evaluation report
├── rationale.py                        # Rationale generator for predictions
├── zero\_shot\_model.py                  # Zero-shot classification with BART
├── preprocessing.py                    # Extracts final user message
├── requirements.txt                    # Python dependencies

````

---

## 🧪 Methods Tried

### ✅ 1. Fine-Tuning DistilBERT

- Used a supervised approach by fine-tuning `distilbert-base-uncased` on a small labeled dataset.
- Final user messages were extracted from multi-turn chats and labeled with corresponding intent.
- Trained using HuggingFace’s `Trainer` API.
- Achieved **accurate predictions** on known intents (especially with 5–10 examples per class).

### ❌ 2. Zero-Shot Classification (BART-Large-MNLI)

We also tested a zero-shot setup using Hugging Face’s `facebook/bart-large-mnli`:

```python
candidate_labels = [
    "Book Appointment",
    "Product Inquiry",
    "Pricing Negotiation",
    "Support Request",
    "Follow-Up"
]
````

However, the model **failed to reliably classify** the correct intent. For example:

* "Please book the villa..." was classified as `"Inquire"` instead of `"Book Appointment"`.
* Overall performance was weak due to the nuanced and domain-specific phrasing in multi-turn conversations.

🔍 **Conclusion**: Zero-shot worked well only when message wording exactly matched label semantics.

---

## 🚀 How to Run

1. **Clone the repo**:

   ```bash
   git clone https://github.com/samsomsabu/Multi-Turn-Intent-Classification-from-Chat-Conversations.git
   cd Multi-Turn-Intent-Classification-from-Chat-Conversations
   ```

2. **Set up the environment**:

   ```bash
   python -m venv zenv
   zenv\Scripts\activate   # On Windows
   pip install -r requirements.txt
   ```

3. **Prepare data**:

   ```bash
   python prepare_dataset.py
   ```

4. **Train the classifier**:

   ```bash
   python train_classifier.py
   ```

5. **Run predictions** (Fine-tuned model):

   ```bash
   python predict.py
   ```

6. **Run zero-shot predictions**:

   ```bash
   python main.py
   ```

7. **Evaluate results**:

   ```bash
   python evaluate.py
   ```

---

## 🔁 Next Steps

* Increase dataset size for better generalization
* Fine-tune on a domain-specific BERT model (e.g., `bert-base-uncased`)
* Try prompt-based LLM classification (e.g., GPT-4-turbo via API)
* Build a simple Streamlit or CLI interface

---

## 🙋 Author

👤 **Samson Sabu**
Email: \[[samsonsabu6@gmail.com](mailto:samsonsabu6@gmail.com)]
GitHub: [samsomsabu](https://github.com/samsomsabu)

---


