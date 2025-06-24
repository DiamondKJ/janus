import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

# --- Unified & Final Configuration ---
MODEL_ID = "mistralai/Mistral-7B-v0.1"
MODEL_CACHE_DIR = "/workspace/janus_model_cache"
DATA_DIR = "../data"

def main(hemisphere: str):
    """
    Main function to fine-tune a model for a specific hemisphere using QLoRA.
    :param hemisphere: 'left' or 'right'
    """
    if hemisphere not in ['left', 'right']:
        raise ValueError("Hemisphere must be either 'left' or 'right'")

    print(f"--- Starting QLoRA Fine-Tuning for: {hemisphere.upper()} HEMISPHERE ---")

    # 1. Determine dataset path and output directory
    dataset_file = os.path.join(DATA_DIR, f"{hemisphere}_brain_corpus.jsonl")
    output_dir = f"/workspace/models/{hemisphere}_hemisphere_v1"

    print(f"Dataset file: {dataset_file}")
    print(f"Output directory: {output_dir}")

    # 2. Configure Quantization (QLoRA)
    # This tells the model to load in 4-bit, drastically reducing memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 3. Load the baseline model and tokenizer, now with quantization
    print("Loading quantized baseline model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR
    )
    print("Model and tokenizer loaded.")
    
    # Configure LoRA - we only train these tiny adapter layers
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Load the specialized dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    print(f"Dataset loaded with {len(dataset)} samples.")

        # 5. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,  # <-- CHANGE 1: Lower learning rate for more stable learning
        logging_steps=10,
        save_steps=500,
        fp16=True,
        push_to_hub=False,
        max_grad_norm=0.3    # <-- CHANGE 2: Add gradient clipping to prevent explosions
    )
    
    # 6. Initialize the SFTTrainer with LoRA config
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512, # Reduce sequence length to be safe
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config # <-- This is the key to activating LoRA
    )
    
    # 7. Start the training
    print("\n--- The QLoRA Forge is Lit. Starting training... ---")
    trainer.train()
    print("--- Training complete. ---")

    # 8. Save the final LoRA adapter
    print(f"Saving final LoRA adapter to {output_dir}...")
    trainer.save_model(output_dir)
    print(f"✅ {hemisphere.upper()} HEMISPHERE adapter has been forged and saved.")

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