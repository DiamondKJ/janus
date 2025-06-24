import os
import argparse
import torch

# Set environment variables to be safe, but the core fix is loading from a local path.
os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/huggingface_datasets_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/huggingface_transformers_cache'
os.environ['TMPDIR'] = '/workspace/tmp'
os.environ['TEMP'] = '/workspace/tmp'

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

# --- Configuration ---
MODEL_ID_FOR_NAMING = "mistralai/Mistral-7B-v0.1" # Used only for directory naming
# --- THIS IS THE CRITICAL FIX ---
# We build the direct, local path to the model we ALREADY downloaded.
LOCAL_MODEL_PATH = "/workspace/huggingface_transformers_cache/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b"
DATA_DIR = "../data"

def main(hemisphere: str):
    if hemisphere not in ['left', 'right']:
        raise ValueError("Hemisphere must be either 'left' or 'right'")

    print(f"--- Starting QLoRA Fine-Tuning for: {hemisphere.upper()} HEMISPHERE ---")

    dataset_file = os.path.join(DATA_DIR, f"{hemisphere}_brain_corpus.jsonl")
    output_dir = f"/workspace/models/{hemisphere}_hemisphere_v1"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.environ['TMPDIR'], exist_ok=True)
    print(f"Loading model from LOCAL path: {LOCAL_MODEL_PATH}")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    # --- We now load from the LOCAL_MODEL_PATH, not the online ID ---
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")

    print("Loading dataset...")
    # The dataset cache needs a different folder name to avoid conflicts
    dataset_cache_path = f"/workspace/hf_dataset_cache_{hemisphere}"
    dataset = load_dataset("json", data_files=dataset_file, split="train", cache_dir=dataset_cache_path)
    
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=1, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, learning_rate=5e-5, logging_steps=10,
        save_steps=500, fp16=True, push_to_hub=False, max_grad_norm=0.3
    )
    
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, dataset_text_field="text",
        max_seq_length=512, tokenizer=tokenizer, args=training_args,
        peft_config=lora_config, packing=True
    )
    
    print("\n--- The Forge is Lit. Starting training... ---")
    trainer.train()
    print("--- Training complete. ---")

    print(f"Saving final LoRA adapter to {output_dir}...")
    trainer.save_model(output_dir)
    print(f"✅ {hemisphere.upper()} HEMISPHERE adapter has been forged and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for a specific hemisphere.")
    parser.add_argument("--hemisphere", type=str, required=True, choices=['left', 'right'], help="The hemisphere to train: 'left' or 'right'")
    args = parser.parse_args()
    main(args.hemisphere)