from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ✅ Updated target intents
candidate_labels = [
    "Book Appointment",
    "Product Inquiry",
    "Pricing Negotiation",
    "Support Request",
    "Follow-Up"
]

def zero_shot_predict(convo):
    print("Candidate labels:", candidate_labels)
    # Combine all user messages into a single string
    text = " ".join([msg["text"] for msg in convo["messages"] if msg["sender"] == "user"])
    result = classifier(text, candidate_labels)
    return result["labels"][0]

# 🔁 Test block
if __name__ == "__main__":
    convo = {
        "messages": [
            {"sender": "user", "text": "I need to reschedule my doctor's appointment."},
            {"sender": "agent", "text": "Sure, what date works for you?"}
        ]
    }
    print(zero_shot_predict(convo))  # ✅ Added closing parenthesis here
