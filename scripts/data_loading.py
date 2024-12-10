import os
from glob import glob
import pandas as pd
import pyarrow.parquet as pq
import dask.dataframe as dd
import logging


def load_parquet_in_chunks(parquet_dir, chunk_size=100000):
    """Load text data from all Parquet files in the specified directory in chunks."""
    parquet_files = glob(os.path.join(parquet_dir, "*.parquet"))
    for file in parquet_files:
        try:
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(
                batch_size=chunk_size, columns=["text"]
            ):
                table = batch.to_pandas()
                for text in table["text"].dropna().astype(str).tolist():
                    yield text.strip()
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            continue  # Skip to the next file


def load_text_files(text_dir):
    """Load text data from all text files in the specified directory."""
    text_files = glob(os.path.join(text_dir, "*.txt"))
    for file in text_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            continue  # Skip to the next file


def load_all_data(parquet_dir, text_dir):
    """Combine data from Parquet and text files."""
    yield from load_parquet_in_chunks(parquet_dir)
    yield from load_text_files(text_dir)
