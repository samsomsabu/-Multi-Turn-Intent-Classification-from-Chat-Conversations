# predict.py
import json
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import joblib
import os

print("🚀 Starting predict.py")

def load_model(model_path="model/distilbert"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model path not found: {model_path}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")
    print("✅ Model loaded")
    return tokenizer, model, label_encoder

def predict_intent(convo, tokenizer, model, label_encoder):
    user_text = " ".join([m["text"] for m in convo["messages"] if m["sender"] == "user"])
    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        label = label_encoder.inverse_transform([pred])[0]
        return label

if __name__ == "__main__":
    tokenizer, model, label_encoder = load_model()

    test_file = "data/test_conversations.json"
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"❌ Test data file not found: {test_file}")

    with open(test_file, "r") as f:
        test_data = json.load(f)

    print(f"📦 Total conversations: {len(test_data)}")

    predictions = []
    for convo in test_data:
        print(f"🔍 Predicting for conversation ID: {convo.get('conversation_id')}")
        try:
            intent = predict_intent(convo, tokenizer, model, label_encoder)
            predictions.append({
                "conversation_id": convo["conversation_id"],
                "predicted_intent": intent
            })
        except Exception as e:
            print(f"❌ Error predicting for conversation: {e}")

    output_file = "predictions.json"
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"✅ Predictions saved to {output_file}")
