def extract_user_text(convo):
    return " ".join([m["text"].strip() for m in convo["messages"] if m["sender"] == "user"])

def clean_text(text):
    return text.replace("\n", " ").strip().lower()