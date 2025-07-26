print("rationale.py loaded")
from preprocessing import extract_user_text

def generate_rationale(convo, predicted_intent):
    user_text = extract_user_text(convo).lower()
    rationale = f"The intent was classified as '{predicted_intent}' because the user's message includes keywords and context consistent with that intent."
    return rationale