# /JANUS-CORE/forge_id_engine.py (V4.1 - Correct Data Collator)

import os
import sys
import logging
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling # <<< NEW: Import the correct data collator
)
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The function create_training_data_from_dreams remains the same
def create_training_data_from_dreams(dream_log_path, schemas, encoder, device):
    # ... (This function is correct and does not need to be changed) ...
    if not os.path.exists(dream_log_path): return []
    with open(dream_log_path, 'r') as f:
        dreams = [json.loads(line) for line in f]
    if not dreams or not schemas: return []

    dream_narratives = [d['narrative'] for d in dreams]
    schema_names = [s['name'] for s in schemas]
    schema_vectors = torch.stack([s['vector'] for s in schemas]).to(device)
    dream_vectors = encoder.encode(dream_narratives, convert_to_tensor=True, show_progress_bar=True).to(device)
    similarity_matrix = util.cos_sim(dream_vectors, schema_vectors)
    best_schema_indices = torch.argmax(similarity_matrix, dim=1)
    labels = [schema_names[i] for i in best_schema_indices]

    training_data = []
    for i, narrative in enumerate(dream_narratives):
        instruction = f"Generate a dream narrative that embodies the core belief: '{labels[i]}'."
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{narrative}"
        training_data.append({"text": text})
    
    return training_data


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # --- 1. Load Schemas and Encoder ---
    # ... (This section is correct and does not need to be changed) ...
    schema_path = os.path.join(project_root, "assets/memory/schemas.pt")
    if not os.path.exists(schema_path):
        logger.error(f"Schema file not found at '{schema_path}'.")
        return
    schemas = torch.load(schema_path)
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # --- 2. Generate Training Data ---
    dream_log_path = os.path.join(project_root, "reports/dreams.jsonl")
    training_data = create_training_data_from_dreams(dream_log_path, schemas, encoder, device)
    if not training_data:
        logger.error("No training data generated. Aborting.")
        return
    logger.info(f"Successfully generated {len(training_data)} training samples for the Id-Engine.")
    hf_dataset = Dataset.from_pandas(pd.DataFrame(training_data))

    # --- 3. Load Base Model ---
    base_model_name = "mistralai/Mistral-7B-v0.1"
    id_engine_path = os.path.join(project_root, "assets/engines/id_engine_v1.0")
    
    logger.info(f"Loading base model '{base_model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 4. Configure LoRA ---
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # --- 5. Tokenize and Set Up Trainer ---
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

    # <<< THE FIX: Use the standard DataCollatorForLanguageModeling >>>
    # It correctly handles padding and creates the 'labels' tensor for us.
    # mlm=False means we are doing Causal Language Modeling, not Masked Language Modeling.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=id_engine_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator, # Use the robust, built-in collator
    )

    logger.info("--- FORGING THE ID-ENGINE ---")
    trainer.train()

    logger.info("Forge complete. Saving the new Id-Engine...")
    trainer.save_model(id_engine_path)
    logger.info(f"Id-Engine saved to '{id_engine_path}'. Janus has evolved.")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()