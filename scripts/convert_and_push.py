from transformers import PreTrainedTokenizerFast
import os

tokenizer_path = "tokenizer/turkish_bpe_tokenizer_50000.json"
output_dir = "tokenizer/"
hub_model_id = "mertcobanov/turkish-bpe-tokenizer-50000"

tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
print("Read from tokenizer file:", tokenizer_path)
tokenizer.save_pretrained(os.path.join(output_dir, "hf_tokenizer_50000"))
print("Saved to:", os.path.join(output_dir, "hf_tokenizer_50000"))
tokenizer.push_to_hub(f"{hub_model_id}")
print(f"Pushed to Hugging Face https://huggingface.co/{hub_model_id}")
