# /JANUS-CORE/forge_analytical_engine.py (Corrected Version)

import os
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling, # <<< NEW: Import the correct data collator
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Forging Script ---
def main():
    # --- 1. Model and Tokenizer Setup ---
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Use 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    logger.info(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- 2. LoRA Configuration ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target attention modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # --- 3. Load and Prepare Dataset ---
    logger.info("Loading and preparing the 'Logician's Library' dataset...")
    # Assuming your dataset for the analytical engine is in this format
    data = load_dataset("text", data_files={"train": "datasets/logicians_library.txt"})

    def tokenize_function(examples):
        # Tokenize and create chunks of a fixed size
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_data = data.map(tokenize_function, batched=True, remove_columns=["text"])

    # --- 4. Set Up Trainer ---
    output_dir = "assets/engines/analytical_engine_v1.0"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3, # Standard number of epochs for fine-tuning
        logging_steps=10,
        save_steps=500,
        fp16=True, # Use mixed precision training
        push_to_hub=False # We are saving locally
    )
    
    # <<< THE FIX: Use the correct, built-in data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_data["train"],
        args=training_args,
        data_collator=data_collator
    )

    # --- 5. Train the Model ---
    logger.info("--- FORGING THE NEW ANALYTICAL-ENGINE ---")
    trainer.train()

    # --- 6. Save the Final Adapter ---
    logger.info("Training complete. Saving final LoRA adapter.")
    trainer.save_model(output_dir)
    # The tokenizer is saved automatically by the Trainer if it's new/modified,
    # but saving it explicitly is good practice.
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()