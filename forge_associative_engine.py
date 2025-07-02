# /JANUS-CORE/forge_associative_engine.py (Corrected Version)

import os
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling, # <<< Import the correct data collator
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # --- 3. Load and Prepare Dataset ---
    logger.info("Loading and preparing the 'Artist's Salon' dataset...")
    # <<< CHANGED: Point to the associative dataset
    data = load_dataset("text", data_files={"train": "datasets/artists_salon.txt"}) 

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_data = data.map(tokenize_function, batched=True, remove_columns=["text"])

    # --- 4. Set Up Trainer ---
    # <<< CHANGED: Point to the associative engine output directory
    output_dir = "assets/engines/associative_engine_v1.0"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=500,
        fp16=True,
        push_to_hub=False
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
    logger.info("--- FORGING THE NEW ASSOCIATIVE-ENGINE ---")
    trainer.train()

    # --- 6. Save the Final Adapter ---
    logger.info("Training complete. Saving final LoRA adapter.")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()