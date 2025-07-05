# Text File Preprocessing with Hugging Face Transformers

This repository contains a simple, reusable script to preprocess raw text files using Hugging Face's transformer tokenizers. The preprocessing includes:

- Loading a raw text file
- Cleaning the text (removing empty lines and trimming whitespace)
- Tokenizing with a pretrained transformer tokenizer (`bert-base-uncased` by default)
- Adding special tokens (`[CLS]`, `[SEP]`)
- Padding and truncating sequences to a fixed maximum length
- Returning model-ready input tensors including `input_ids` and `attention_mask`

---

## Why Use Transformer Tokenization?

Transformer models like BERT require specific input formatting to work correctly. Unlike traditional NLP preprocessing (lowercasing, stemming), transformer tokenization preserves the context by:

- Using **subword tokenization** (WordPiece)
- Adding **special tokens** to indicate sentence boundaries
- Creating **attention masks** to differentiate real tokens from padding

This ensures your text is properly formatted and optimized for transformer-based models.

---

## Usage

### 1. Install Dependencies

Make sure you have the `transformers` library installed:

```bash
pip install transformers


## Run the Preprocessing Script

python preprocess_text_file.py

from transformers import AutoTokenizer

def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def preprocess_texts(text_list, tokenizer_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded_inputs = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    return encoded_inputs

def main():
    file_path = "your_text_file.txt"
    texts = load_text_file(file_path)
    print(f"Loaded {len(texts)} non-empty lines from the file.")
    print("Sample lines:", texts[:3])

    encoded_inputs = preprocess_texts(texts)
    print("\nPreprocessing output keys:", encoded_inputs.keys())
    print("Sample input_ids:", encoded_inputs["input_ids"][:3])
    print("Sample attention_mask:", encoded_inputs["attention_mask"][:3])

if __name__ == "__main__":
    main()

