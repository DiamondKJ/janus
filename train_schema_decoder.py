# /JANUS-CORE/train_schema_decoder.py

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
from datasets import Dataset

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Training Script ---
def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # --- 1. Load Labeled Dataset ---
    dataset_path = os.path.join(project_root, 'datasets/labeled_dreams.csv')
    if not os.path.exists(dataset_path):
        logger.error(f"Labeled dataset not found at '{dataset_path}'. Please run label_corpus.py first.")
        return
        
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} labeled dreams from corpus.")

    # --- 2. Prepare Data and Labels ---
    # Create a mapping from string labels to integer IDs
    unique_labels = df['schema_label'].unique()
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    # Apply the mapping to our dataframe
    df['label'] = df['schema_label'].map(label2id)
    
    # --- 3. Load Tokenizer and Model ---
    model_name = "distilbert-base-uncased"
    logger.info(f"Loading tokenizer and model for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    # --- 4. Tokenize Dataset ---
    logger.info("Tokenizing dataset...")
    
    # Convert pandas DataFrame to Hugging Face Dataset object
    hf_dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(examples["dream_text"], padding="max_length", truncation=True, max_length=256)

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
    
    # Split into training and validation sets
    train_val_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_val_split['train']
    eval_dataset = train_val_split['test']
    
    logger.info(f"Dataset split into {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")
    
    # --- 5. Set Up Trainer ---
    output_dir = os.path.join(project_root, 'assets/modules/schema_decoder_v1.0')
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # 5 epochs should be enough for a small dataset
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        use_mps_device=torch.backends.mps.is_available() # Use MPS if available
    )
    
    # We need a compute_metrics function for evaluation
    import numpy as np
    from datasets import load_metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # --- 6. Train the Model ---
    logger.info("--- Beginning Training for the Internal Psychoanalyst (Schema Decoder) ---")
    trainer.train()
    
    # --- 7. Save the Final Model ---
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Training complete. Schema Decoder saved to '{output_dir}'.")

if __name__ == "__main__":
    # The metrics library download might cause issues with multiprocessing on macOS
    # We can set an environment variable to mitigate this.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()