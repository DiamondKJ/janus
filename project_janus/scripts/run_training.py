import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# --- Unified & Final Configuration ---
MODEL_ID = "mistralai/Mistral-7B-v0.1"
MODEL_CACHE_DIR = "../models/cache"
DATA_DIR = "../data"

def main(hemisphere: str):
    """
    Main function to fine-tune a model for a specific hemisphere.
    :param hemisphere: 'left' or 'right'
    """
    if hemisphere not in ['left', 'right']:
        raise ValueError("Hemisphere must be either 'left' or 'right'")

    print(f"--- Starting Fine-Tuning for: {hemisphere.upper()} HEMISPHERE ---")

    # 1. Determine dataset path and output directory based on hemisphere
    dataset_file = os.path.join(DATA_DIR, f"{hemisphere}_brain_corpus.jsonl")
    output_dir = f"../models/{hemisphere}_hemisphere_v1"

    print(f"Dataset file: {dataset_file}")
    print(f"Output directory: {output_dir}")

    # 2. Load the baseline model and tokenizer from our cache
    print("Loading baseline model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    # Set a padding token if one doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        device_map="auto" # Let accelerate handle device placement on the GPU
    )
    print("Model and tokenizer loaded.")

    # 3. Load the specialized dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 4. Configure Training Arguments
    # These are the hyperparameters for the fine-tuning process.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # 1 epoch is often enough for fine-tuning
        per_device_train_batch_size=1, # Use 1 for A100 40GB to be safe
        gradient_accumulation_steps=4, # Simulates a batch size of 4
        learning_rate=2e-5,
        logging_steps=10, # Log progress every 10 steps
        save_steps=500, # Save a checkpoint every 500 steps
        fp16=True, # Use mixed-precision for speed and memory efficiency
        push_to_hub=False # We are saving locally to our VM
    )
    
    # 5. Initialize the SFTTrainer
    # SFTTrainer from TRL is a convenient wrapper for Supervised Fine-Tuning.
    # It handles all the data formatting and training loops for us.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",  # The column in our .jsonl file that contains the text
        max_seq_length=1024, # Max sequence length to process
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # 6. Start the training
    print("\n--- The Forge is Lit. Starting training... ---")
    trainer.train()
    print("--- Training complete. ---")

    # 7. Save the final model
    print(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    print(f"✅ {hemisphere.upper()} HEMISPHERE has been forged and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for a specific hemisphere.")
    parser.add_argument(
        "--hemisphere",
        type=str,
        required=True,
        choices=['left', 'right'],
        help="The hemisphere to train: 'left' or 'right'"
    )
    args = parser.parse_args()
    main(args.hemisphere)