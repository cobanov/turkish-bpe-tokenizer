import os
from scripts import utils
from scripts.data_loading import load_all_data
from scripts.data_cleaning import prepare_cleaned_data
from scripts.tokenizer_training import train_tokenizer
from scripts.evaluation import (
    load_tokenizer,
    tokenize_samples,
    coverage_testing,
)

# Configuration Parameters
PARQUET_DIR = "data/turkish/"  # Directory containing Parquet files
TEXT_FILES_DIR = "data/turkish_texts/"  # Directory containing additional text files
VOCAB_SIZE = 30000  # Desired vocabulary size
MIN_FREQUENCY = 2  # Minimum frequency for a token to be included
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
TOKENIZER_SAVE_PATH = "tokenizer/turkish_bpe_tokenizer.json"


# Sample sentences for tokenization examples
SAMPLE_SENTENCES = [
    "Merhaba, nasılsınız?",
    "Bu bir test cümlesidir.",
    "İstanbul, Türkiye'nin en büyük şehridir.",
]

# Validation sentences for coverage testing
VALIDATION_SENTENCES = [
    "Güneş bugün çok güzel görünüyor.",
    "Yarın hava yağışlı olacak.",
    "Kitap okumak eğlencelidir.",
]


def main():
    # Setup logging
    utils.setup_logging()
    utils.log_memory_usage()

    # Load and clean data
    print("Loading and cleaning data...")
    utils.log_memory_usage()
    raw_data = load_all_data(PARQUET_DIR, TEXT_FILES_DIR)
    cleaned_data = prepare_cleaned_data(raw_data)

    # Train the tokenizer
    print("Training the tokenizer...")
    tokenizer = train_tokenizer(
        cleaned_data_iterator=cleaned_data,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )
    utils.log_memory_usage()

    # Ensure the tokenizer directory exists
    os.makedirs(os.path.dirname(TOKENIZER_SAVE_PATH), exist_ok=True)

    # Save the tokenizer
    tokenizer.save(TOKENIZER_SAVE_PATH)
    print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")
    utils.log_memory_usage()

    # Evaluation
    print("\n--- Evaluation ---")
    tokenizer = load_tokenizer(TOKENIZER_SAVE_PATH)
    if tokenizer is None:
        print("Failed to load the tokenizer. Exiting evaluation.")
        return

    tokenize_samples(tokenizer, SAMPLE_SENTENCES)
    coverage_testing(tokenizer, VALIDATION_SENTENCES)


if __name__ == "__main__":
    main()
