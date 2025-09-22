# Minimal BPE Tokenizer (Python)

## Overview
This is a simple implementation of a **Byte Pair Encoding (BPE) Tokenizer** in Python.

- Trains on a given corpus (`alice.txt`).
- Encodes input text into integer token IDs.
- Decodes token IDs back to the original text.

## Project Structure
```
bpe_tokenizer/
├── src/
│   ├── tokenizer.py   # Core BPE tokenizer implementation
│   └── main.py        # Training + demo script
├── alice.txt          # Training corpus (add your text file here)
├── output.txt         # Example output of encode/decode
├── README.md          # Documentation
```

## Usage

1. Place your training text file (e.g., `alice.txt`) in the project root.
2. Run the demo:

```bash
cd src
python main.py
```

3. Check the results in `output.txt`.

## Example
```
Sample text: Alice was beginning to get very tired.
Encoded: [12, 33, 47, 9, 21, 88, 54, 72, 90, 15]
Decoded: Alice was beginning to get very tired.
```
