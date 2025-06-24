import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

# The only path we care about. All files will live here.
WORKSPACE_DIR = "/workspace/janus_run"

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-v0.1"
DATA_DIR = "../data"

def main(hemisphere: str):
    if hemisphere not in ['left', 'right']:
        raise ValueError("Hemisphere must be either 'left' or 'right'")

    print(f"--- Starting QLoRA Fine-Tuning for: {hemisphere.upper()} HEMISPHERE ---")

    dataset_file = os.path.join(DATA_DIR, f"{hemisphere}_brain_corpus.jsonl")
    output_dir = os.path.join(WORKSPACE_DIR, "models", f"{hemisphere}_hemisphere_v1")
    cache_dir = os.path.join(WORKSPACE_DIR, "cache")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    auth_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not auth_token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not set.")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    print(f"Using cache directory: {cache_dir}")
    print("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir, token=auth_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
        token=auth_token
    )
    
    lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_file, split="train", cache_dir=os.path.join(cache_dir, "datasets"))
    
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