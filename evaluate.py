# evaluate.py
import json
import pandas as pd
from sklearn.metrics import classification_report

def load_ground_truth(path="data/ground_truth.csv"):
    return pd.read_csv(path).set_index("conversation_id")["intent"].to_dict()

def load_predictions(path="predictions.json"):
    with open(path) as f:
        preds = json.load(f)
    return {p["conversation_id"]: p["predicted_intent"] for p in preds}

def evaluate():
    y_true_dict = load_ground_truth()
    y_pred_dict = load_predictions()

    y_true = []
    y_pred = []

    for cid, true_intent in y_true_dict.items():
        pred_intent = y_pred_dict.get(cid, "Unknown")
        y_true.append(true_intent)
        y_pred.append(pred_intent)

    report = classification_report(y_true, y_pred, digits=3)
    print("Evaluation Report:\n")
    print(report)
    

if __name__ == "__main__":
    evaluate()
    
