import re
import unicodedata
import hashlib
import langid
import logging

# Define the allowed characters based on the Turkish alphabet and digits
ALLOWED_CHARACTERS_REGEX = re.compile(r"[^A-Za-zÇçĞğIıİiÖöŞşÜü0-9\s.,!?;:()\"\'`-]+")


def is_turkish(text):
    """Check if the text is in Turkish."""
    lang, _ = langid.classify(text)
    return lang == "tr"


def clean_text(text):
    """Clean a single piece of text."""
    if not is_turkish(text):
        return None

    # Unicode Normalization
    text = unicodedata.normalize("NFC", text)

    # Remove Unrelated Characters
    text = ALLOWED_CHARACTERS_REGEX.sub(" ", text)

    # Remove Extra Whitespace
    text = " ".join(text.split())

    return text if text else None


def clean_texts(texts):
    """Clean a batch of texts."""
    cleaned_texts = []
    for text in texts:
        cleaned = clean_text(text)
        if cleaned:
            cleaned_texts.append(cleaned)
    return cleaned_texts
