# /JANUS-CORE/janus/training/train_inhibitor.py

import os
import torch
import logging
from datasets import load_dataset, ClassLabel, Features, Value # <-- Import new types
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

MODEL_ID = "distilbert-base-uncased"

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def train_new_inhibitor(project_root, training_data_path):
    output_path = os.path.join(project_root, "assets/modules/inhibitor_engine_v1.0")
    
    logger.info(f"Loading inhibitor dataset from: {training_data_path}")

    # --- THIS IS THE FIX: Define the schema for our dataset ---
    feature_schema = Features({
        'text': Value('string'),
        'label': ClassLabel(names=['CONTINUE', 'STOP']) # 0=CONTINUE, 1=STOP
    })
    # --- END OF FIX ---
    
    # Load the dataset using our new, explicit schema
    full_dataset = load_dataset("json", data_files=training_data_path, features=feature_schema, split="train")
    
    # Now stratification will work correctly
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']
    
    logger.info(f"Dataset loaded. Training with {len(train_dataset)} examples, evaluating with {len(eval_dataset)}.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    
    logger.info(f"Loaded '{MODEL_ID}' for sequence classification.")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=['text'])
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=['text'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        use_mps_device=torch.backends.mps.is_available(),
        fp16=False, bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("--- Starting Inhibitor Engine Training ---")
    trainer.train()
    logger.info("--- Training Complete ---")

    logger.info(f"Saving best model to {output_path}")
    trainer.save_model()
    
    logger.info("Inhibitor Engine 'inhibitor_engine_v1.0' has been successfully forged.")
    return output_path