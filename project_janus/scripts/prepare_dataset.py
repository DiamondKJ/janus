import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# --- Unified & Final Configuration ---
# Use the EXACT SAME constants as run_evaluation.py for consistency
MODEL_ID = "mistralai/Mistral-7B-v0.1"
MODEL_CACHE_DIR = "../models/cache"

# This dictionary defines the "curriculum" for each hemisphere.

# In scripts/prepare_dataset.py

# --- Corrected & Final DATA_CONFIG ---
DATA_CONFIG = {
    "left_hemisphere": {
        "datasets": [
            # Switched to a more accessible and well-formatted code dataset
            {"name": "codewithzulu/stack-code-m-python-200k", "config": "default", "text_column": "text"},
            # Switched to a clean math instruction dataset
            {"name": "TIGER-Lab/MathInstruct", "config": "default", "text_column": "instruction"},
        ],
        "output_file": "../data/left_brain_corpus.jsonl"
    },
    "right_hemisphere": {
        "datasets": [
            # Using the canonical 'pg19' dataset for literature
            {"name": "pg19", "config": "default", "text_column": "text"},
            # Using a rich caption dataset for descriptive, creative text
            {"name": "conceptual_captions", "config": "default", "text_column": "caption"},
        ],
        "output_file": "../data/right_brain_corpus.jsonl"
    }
}

OUTPUT_DIR = "../data"
MAX_SAMPLES_PER_DATASET = 20000 # Limit samples to keep processing time reasonable

# --- Main Script ---

def process_and_save_dataset(datasets_info: list, output_file: str):
    """
    Downloads, processes, and saves a combined dataset to a .jsonl file.
    """
    print(f"--- Processing curriculum for {os.path.basename(output_file)} ---")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        total_samples_written = 0
        for dataset_info in datasets_info:
            name, config, text_column = dataset_info['name'], dataset_info['config'], dataset_info['text_column']
            
            print(f"\n> Loading dataset: {name} ({config})")
            
            try:
                dataset = load_dataset(name, config, split="train", streaming=True, trust_remote_code=True)
                
                samples_written_for_this_dataset = 0
                for doc in tqdm(dataset, desc=f"Processing {name}"):
                    if samples_written_for_this_dataset >= MAX_SAMPLES_PER_DATASET:
                        break
                    
                    text = doc.get(text_column)
                    if text and isinstance(text, str) and len(text.strip()) > 100:
                        record = {"text": text.strip()}
                        f.write(json.dumps(record) + "\n")
                        samples_written_for_this_dataset += 1
                
                total_samples_written += samples_written_for_this_dataset
                print(f"> Finished {name}. Wrote {samples_written_for_this_dataset} samples.")
            except Exception as e:
                print(f"!! FAILED to process {name}. Error: {e}. Skipping this dataset.")

    print(f"\n--- Curriculum processing complete. ---")
    print(f"Total samples written to {output_file}: {total_samples_written}")


def main():
    """Main function to orchestrate dataset preparation for both hemispheres."""
    print("Verifying tokenizer can be loaded from cache...")
    try:
        # Load by ID, but it will use the cache since it's already downloaded
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
        print("✅ Tokenizer verified successfully.")
    except Exception as e:
        print(f"❌ Could not load tokenizer. Error: {e}")
        print("Please ensure you have run the evaluation script successfully first.")
        return

    # Process the Left Hemisphere curriculum
    process_and_save_dataset(
        datasets_info=DATA_CONFIG["left_hemisphere"]["datasets"],
        output_file=DATA_CONFIG["left_hemisphere"]["output_file"]
    )
    
    # Process the Right Hemisphere curriculum
    process_and_save_dataset(
        datasets_info=DATA_CONFIG["right_hemisphere"]["datasets"],
        output_file=DATA_CONFIG["right_hemisphere"]["output_file"]
    )

    print("\n✅ All datasets have been processed and saved.")

if __name__ == "__main__":
    main()