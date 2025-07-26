

# 🧠 Multi-Turn Intent Classification from Chat Conversations

This project implements a **multi-turn intent classification system** to identify user intent from chat-style conversations (e.g., WhatsApp threads). Developed as a machine learning assignment, it explores both **fine-tuned classification** and **zero-shot learning** approaches for classifying user intents based on multi-turn dialogues.

---

## 📌 Objective

The goal is to classify the **final intent of the user** from a multi-turn conversation into one of the following categories:
- **Book Appointment**
- **Product Inquiry**
- **Pricing Negotiation**
- **Support Request**
- **Follow-Up**

---

## 🗂️ Project Structure

```
├── data/
│   ├── sample_conversations.json       # Sample training conversations
│   ├── test_conversations.json         # Unlabeled test conversations
│   ├── ground_truth.csv                # Ground truth labels for test set
│   ├── train.csv                       # Processed dataset for training
├── model/
│   └── distilbert/                     # Fine-tuned DistilBERT model and label encoder
├── main.py                             # Zero-shot intent prediction script
├── predict.py                          # Inference with fine-tuned model
├── prepare_dataset.py                  # Data preprocessing for training
├── train_classifier.py                 # Fine-tuning script for DistilBERT
├── evaluate.py                         # Evaluation script for model performance
├── rationale.py                        # Generates rationales for predictions
├── zero_shot_model.py                  # Zero-shot classification using BART
├── preprocessing.py                    # Extracts final user message from conversations
├── requirements.txt                    # Python dependencies
```

---

## 🧪 Approach

### 1. Fine-Tuned DistilBERT Classifier
- **Model**: `distilbert-base-uncased` fine-tuned on a small labeled dataset.
- **Methodology**:
  - Extracted final user messages from multi-turn conversations.
  - Labeled messages with corresponding intents.
  - Used HuggingFace’s `Trainer` API for training.
- **Performance**: Achieved reliable results with 5–10 examples per intent class.

### 2. Zero-Shot Classification with BART
- **Model**: `facebook/bart-large-mnli`.
- **Candidate Labels**:
  ```python
  candidate_labels = [
      "Book Appointment",
      "Product Inquiry",
      "Pricing Negotiation",
      "Support Request",
      "Follow-Up"
  ]
  ```
- **Challenges**:
  - Struggled with nuanced, domain-specific phrasing in conversations.
  - Example: Classified "Please book the villa..." as "Inquire" instead of "Book Appointment".
  - **Conclusion**: Zero-shot approach performed poorly unless message wording closely matched label semantics.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Dependencies listed in `requirements.txt`

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/samsomsabu/Multi-Turn-Intent-Classification-from-Chat-Conversations.git
   cd Multi-Turn-Intent-Classification-from-Chat-Conversations
   ```

2. **Set up the virtual environment**:
   ```bash
   python -m venv zenv
   source zenv/bin/activate  # On Linux/Mac
   zenv\Scripts\activate     # On Windows
   pip install -r requirements.txt
   ```

### Usage
1. **Prepare the dataset**:
   ```bash
   python prepare_dataset.py
   ```

2. **Train the DistilBERT classifier**:
   ```bash
   python train_classifier.py
   ```

3. **Run predictions with the fine-tuned model**:
   ```bash
   python predict.py
   ```

4. **Run zero-shot predictions**:
   ```bash
   python main.py
   ```

5. **Evaluate model performance**:
   ```bash
   python evaluate.py
   ```

---

## 🔁 Next Steps
- Expand the dataset for improved model generalization.
- Experiment with domain-specific BERT models (e.g., `bert-base-uncased`).
- Explore prompt-based classification using LLMs (e.g., GPT-4-turbo via API).
- Develop a user-friendly interface using Streamlit or a CLI.

---

## 🙋 Author
**Samson Sabu**  
- Email: [samsonsabu6@gmail.com](mailto:samsonsabu6@gmail.com)  
- GitHub: [samsomsabu](https://github.com/samsomsabu)

---

