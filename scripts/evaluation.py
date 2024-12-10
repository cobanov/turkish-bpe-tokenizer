from tokenizers import Tokenizer
import logging


def load_tokenizer(save_path):
    """Load a tokenizer from a file."""
    try:
        tokenizer = Tokenizer.from_file(save_path)
        logging.info(f"Tokenizer loaded from {save_path}")
        return tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer from {save_path}: {e}")
        return None


def tokenize_samples(tokenizer, sample_sentences):
    """Tokenize sample sentences and display the tokens."""
    for sentence in sample_sentences:
        encoding = tokenizer.encode(sentence)
        logging.info(f"Sentence: {sentence}")
        logging.info(f"Tokens: {encoding.tokens}\n")


def coverage_testing(tokenizer, validation_sentences):
    """Calculate the coverage of the tokenizer on a validation set."""
    total_tokens = 0
    unknown_tokens = 0

    for text in validation_sentences:
        encoding = tokenizer.encode(text)
        total_tokens += len(encoding.tokens)
        unknown_tokens += encoding.tokens.count("<UNK>")

    coverage = (
        100 * (total_tokens - unknown_tokens) / total_tokens if total_tokens > 0 else 0
    )
    logging.info(f"Coverage on validation set: {coverage:.2f}%")
