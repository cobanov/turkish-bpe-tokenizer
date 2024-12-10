from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
import logging


def train_tokenizer(cleaned_data_iterator, vocab_size, min_frequency, special_tokens):
    """Train a BPE tokenizer using the provided cleaned data."""
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Set pre-tokenizer to split by whitespace
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Initialize the BPE trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    logging.info("Starting tokenizer training...")
    tokenizer.train_from_iterator(cleaned_data_iterator, trainer=trainer)
    logging.info("Tokenizer training completed.")

    # Post-processing: Add BOS and EOS tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<BOS> $A <EOS>",
        pair="<BOS> $A <EOS> <BOS> $B <EOS>",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )

    return tokenizer
