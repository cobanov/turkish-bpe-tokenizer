# Turkish BPE Tokenizer Training

This project trains a Byte Pair Encoding (BPE) tokenizer tailored for the Turkish language using a combination of Parquet and text files.

## **Setup Instructions**

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd turkish_tokenizer
   ```

2. **Install Dependencies**

   It's recommended to use a virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare Data**

   - Place all your `.parquet` files in `data/turkish/`.
   - Place all your additional `.txt` files in `data/turkish_texts/`.

4. **Run the Training Script**

   ```bash
   python main.py
   ```

   The script will:

   - Load and clean the data.
   - Train the BPE tokenizer.
   - Save the tokenizer to `tokenizer/turkish_bpe_tokenizer.json`.
   - Evaluate the tokenizer to ensure quality.

5. **Check Logs**

   The process logs are saved in `tokenizer_training.log`. Review this file for detailed information about the training process and any potential issues.

## **Customization**

- **Vocabulary Size**: Adjust `VOCAB_SIZE` in `main.py` as needed.
- **Minimum Frequency**: Adjust `MIN_FREQUENCY` in `main.py`.
- **Special Tokens**: Modify `SPECIAL_TOKENS` in `main.py` if required.

## **License**

[MIT License](LICENSE)
