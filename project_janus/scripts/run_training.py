import os
import argparse
import torch

# We no longer need to force HF_HOME for authentication, but we keep it for caching
os.environ['HF_HOME'] = '/workspace/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/huggingface_datasets_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/huggingface_transformers_cache'
os.environ['TMPDIR'] = '/workspace/tmp'
os.environ['TEMP'] = '/workspace/tmp'

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

MODEL_ID = "mistralai/Mistral-7B-v0.1"
MODEL_CACHE_DIR = os.environ['TRANSFORMERS_CACHE'] 
DATA_DIR = "../data"

def main(hemisphere: str):
    if hemisphere not in ['left', 'right']:
        raise ValueError("Hemisphere must be either 'left' or 'right'")

    print(f"--- Starting QLoRA Fine-Tuning for: {hemisphere.upper()} HEMISPHERE ---")

    dataset_file = os.path.join(DATA_DIR, f"{hemisphere}_brain_corpus.jsonl")
    output_dir = f"/workspace/models/{hemisphere}_hemisphere_v1"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.environ['TMPDIR'], exist_ok=True)
    
    # --- THE KEY FIX: Get the auth token from the environment ---
    auth_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if auth_token is None:
        raise ValueError("Hugging Face token not found. Please set the HUGGING_FACE_HUB_TOKEN environment variable.")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    print("Loading quantized baseline model and tokenizer...")
    # --- Pass the token directly to the download functions ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR, token=auth_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR,
        token=auth_token
    )
    
    lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_file, split="train", cache_dir='/workspace/hf_cache')
    
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