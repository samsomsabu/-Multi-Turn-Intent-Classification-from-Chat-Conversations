# prepare_dataset.py
import json
import pandas as pd
import os

def prepare_train_csv(conversations_file, labels_file, output_file):
    if not os.path.exists(conversations_file):
        raise FileNotFoundError(f"❌ Conversations file not found: {conversations_file}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"❌ Labels file not found: {labels_file}")

    with open(conversations_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"✅ Loaded {len(conversations)} conversations")

    labels_df = pd.read_csv(labels_file)
    print(f"✅ Loaded {len(labels_df)} labels")

    label_map = dict(zip(labels_df["conversation_id"], labels_df["intent"]))

    data = []
    skipped = 0

    for convo in conversations:
        conv_id = convo.get("conversation_id")
        messages = convo.get("messages", [])

        if conv_id not in label_map:
            skipped += 1
            continue  # no label, skip

        # Concatenate all user messages
        final_user_msg = " ".join([msg["text"] for msg in messages if msg["sender"] == "user"])

        data.append({
            "conversation_id": conv_id,
            "text": final_user_msg,
            "label": label_map[conv_id]
        })

    if not data:
        raise ValueError("❌ No data prepared — check if conversation IDs in JSON and CSV match.")

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df)} records to {output_file}")
    if skipped:
        print(f"⚠️ Skipped {skipped} conversations without matching labels")

if __name__ == "__main__":
    prepare_train_csv(
        "data/sample_conversations.json",
        "data/ground_truth.csv",
        "data/train.csv"
    )
