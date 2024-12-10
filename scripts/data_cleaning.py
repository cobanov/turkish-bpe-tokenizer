import re
import unicodedata
import hashlib
import langid
import logging

# Define the allowed characters based on the Turkish alphabet
ALLOWED_CHARACTERS_REGEX = re.compile(r"[^A-Za-zÇçĞğIıİiÖöŞşÜü0-9\s.,!?;:()\"\'`-]+")


def is_turkish(text):
    """Check if the text is in Turkish."""
    lang, _ = langid.classify(text)
    return lang == "tr"


def clean_text(text):
    """Clean the text by applying language filtering, removing unwanted characters, normalization, and whitespace management."""
    if not is_turkish(text):
        return None

    # Unicode Normalization
    text = unicodedata.normalize("NFC", text)

    # Remove Unrelated Characters
    text = ALLOWED_CHARACTERS_REGEX.sub(" ", text)

    # Remove Extra Whitespace
    text = " ".join(text.split())

    return text if text else None


def prepare_cleaned_data(raw_data_iterator):
    """Generator that yields cleaned text without duplicates using hashes."""
    seen_hashes = set()
    for text in raw_data_iterator:
        cleaned = clean_text(text)
        if cleaned:
            text_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                yield cleaned
