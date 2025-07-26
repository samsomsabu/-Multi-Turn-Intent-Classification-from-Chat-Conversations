# train_classifier.py
import os
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import joblib

print("🚀 Starting fine-tuning...")

# === 1. Load and validate CSV ===
train_csv_path = "data/train.csv"
if not os.path.exists(train_csv_path):
    raise FileNotFoundError(f"❌ Train file not found at {train_csv_path}")

df = pd.read_csv(train_csv_path)
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("❌ Dataset must have 'text' and 'label' columns.")
print(f"✅ Loaded dataset with {len(df)} rows")

# === 2. Encode labels ===
label_encoder = LabelEncoder()
df["label_enc"] = label_encoder.fit_transform(df["label"])
print(f"✅ Encoded {len(label_encoder.classes_)} unique labels")

# === 3. Tokenization ===
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
dataset = Dataset.from_pandas(df)

print("🔁 Tokenizing data...")
dataset = dataset.map(
    lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=128),
    batched=True
)

# Rename and format for PyTorch
dataset = dataset.rename_column("label_enc", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === 4. Split dataset ===
train_test = dataset.train_test_split(test_size=0.2)
print(f"📊 Train size: {len(train_test['train'])} | Test size: {len(train_test['test'])}")

# === 5. Load model ===
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

# === 6. Training Arguments ===
training_args = TrainingArguments(
    output_dir="./model/distilbert",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    # load_best_model_at_end=True,  # ❌ remove or comment out this too
    # metric_for_best_model="eval_loss",  # ❌ remove
    report_to="none"
)


# === 7. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    tokenizer=tokenizer
)

# === 8. Train ===
print("🚦 Starting training...")
trainer.train()
print("✅ Training complete")

# === 9. Save model ===
print("💾 Saving model and tokenizer...")
model_dir = "./model/distilbert"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
print(f"✅ Model, tokenizer, and label encoder saved to {model_dir}")
