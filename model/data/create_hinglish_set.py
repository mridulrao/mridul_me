from pathlib import Path
import json
from datasets import load_dataset, concatenate_datasets

DATA_DIR = Path("./data")
OUTPUT_FILE = DATA_DIR / "hinglish_cpt_combined.jsonl"

def load_txt_files():
    texts = []
    txt_files = sorted(DATA_DIR.glob("HinglishData*.txt"))
    
    print(f"Found {len(txt_files)} files: {txt_files}")
    
    for file in txt_files:
        content = file.read_text(encoding="utf-8").strip()
        # Split into reasonable chunks (paragraphs or every 3-5 newlines)
        chunks = [chunk.strip() for chunk in content.split("\n\n") if len(chunk.strip()) > 30]
        texts.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {file.name}")
    
    return texts

def main():
    texts = load_txt_files()
    
    # Optional: Filter very short or garbage lines
    texts = [t for t in texts if len(t) > 50]
    
    print(f"Total chunks: {len(texts)}")
    
    # Save as JSONL (same format as your previous CPT data)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")
    
    print(f"Saved combined dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()