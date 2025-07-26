# main.py
from zero_shot_model import zero_shot_predict
from rationale import generate_rationale
import json
import os

print("🚀 Starting zero-shot prediction")

if __name__ == "__main__":
    if not os.path.exists("data/test_conversations.json"):
        print("❌ test_conversations.json not found in data/")
        exit(1)

    with open("data/test_conversations.json") as f:
        test_data = json.load(f)

    print(f"📦 Loaded {len(test_data)} test conversations")

    predictions = []
    for i, convo in enumerate(test_data):
        try:
            print(f"\n🔍 [{i+1}/{len(test_data)}] Predicting for conversation ID: {convo.get('conversation_id')}")
            pred_intent = zero_shot_predict(convo)
            rationale = generate_rationale(convo, pred_intent)
            predictions.append({
                "conversation_id": convo["conversation_id"],
                "predicted_intent": pred_intent,
                "rationale": rationale
            })
            print(f"✅ Intent: {pred_intent}")
            print(f"🧠 Rationale: {rationale}")
        except Exception as e:
            print(f"❌ Error for convo {convo.get('conversation_id')}: {e}")

    with open("zero_shot_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print("\n✅ All predictions saved to zero_shot_predictions.json")
