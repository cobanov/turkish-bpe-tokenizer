import os
from scripts import utils
from scripts.data_loading import load_all_data
from scripts.data_cleaning import clean_texts
from scripts.tokenizer_training import train_tokenizer
from scripts.evaluation import (
    load_tokenizer,
    tokenize_samples,
    coverage_testing,
)
import concurrent.futures
import hashlib
import sys

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


def batch_generator(iterator, batch_size=1000):
    """Generator that yields lists of texts in batches."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    # Setup logging
    utils.setup_logging()
    utils.log_memory_usage()

    # Load all raw data
    print("Loading data...")
    raw_data = load_all_data(PARQUET_DIR, TEXT_FILES_DIR)

    # Initialize deduplication set
    seen_hashes = set()

    # Initialize ProcessPoolExecutor
    num_workers = os.cpu_count() or 4  # Default to 4 if os.cpu_count() is None
    print(f"Using {num_workers} worker processes for data cleaning.")

    # Initialize cleaned data list
    cleaned_data = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit cleaning tasks in batches
        futures = []
        for batch in batch_generator(raw_data, batch_size=1000):
            futures.append(executor.submit(clean_texts, batch))

        # As each future completes, process the results
        for future in concurrent.futures.as_completed(futures):
            try:
                cleaned_batch = future.result()
                for cleaned_text in cleaned_batch:
                    # Deduplication
                    text_hash = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()
                    if text_hash not in seen_hashes:
                        seen_hashes.add(text_hash)
                        cleaned_data.append(cleaned_text)
            except Exception as e:
                utils.logger.error(f"Error during data cleaning: {e}")

    print(f"Total cleaned and deduplicated texts: {len(cleaned_data)}")
    utils.log_memory_usage()

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
        sys.exit(1)

    tokenize_samples(tokenizer, SAMPLE_SENTENCES)
    coverage_testing(tokenizer, VALIDATION_SENTENCES)
    utils.log_memory_usage()


if __name__ == "__main__":
    main()
