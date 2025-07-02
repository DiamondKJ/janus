# /JANUS-CORE/forge_analytical_engine.py

import os
import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # --- 1. Load the Dataset ---
    # This assumes your dataset is in a 'datasets' subdirectory.
    # Adjust path if necessary.
    dataset_path = os.path.join(project_root, "datasets/right_brain_corpus.jsonl")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at: {dataset_path}")
        return
        
    logger.info(f"Loading Analytical dataset from {dataset_path}...")
    # Use the 'text' column, assuming each line is a JSON object like {"text": "..."}
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    # --- 2. Load the Base Model & Tokenizer ---
    base_model_name = "mistralai/Mistral-7B-v0.1"
    output_path = os.path.join(project_root, "assets/engines/associative_engine_v2.0")
    
    logger.info(f"Loading base model '{base_model_name}' to forge the new Analytical-Engine...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        # We just train it to predict the next token in the text.
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # --- 3. Configure LoRA and Prepare for Training ---
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)

    # --- 4. Train the Analytical-Engine ---
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=50,
        num_train_epochs=1, # One epoch is often enough for a large dataset
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                     'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                     'labels': torch.stack([f['input_ids'] for f in data])}
    )

    logger.info("--- FORGING THE NEW ANALYTICAL-ENGINE ---")
    trainer.train()

    # --- 5. Save the Forged Engine ---
    logger.info("Forge complete. Saving the new Analytical-Engine...")
    trainer.save_model(output_path)
    logger.info(f"Analytical-Engine v2.0 saved to '{output_path}'.")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()