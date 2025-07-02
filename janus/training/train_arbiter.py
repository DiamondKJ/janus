# /JANUS-CORE/janus/training/train_arbiter.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# --- Configuration & Hyperparameters ---
# THIS IS THE FIX: We now use the same tokenizer path as the main harness.
TOKENIZER_PATH_REL = "../assets/engines/analytical_engine_v1.0"
DATASET_PATH_REL = "../datasets/arbiter_training_data.pt"
OUTPUT_MODEL_PATH_REL = "../assets/modules/arbiter_v1.0.pt" # Overwrite the old one

# The Arbiter now needs the Mistral vocabulary size
VOCAB_SIZE = 32000 
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
OUTPUT_DIM = 1
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 4

# ... (ArbiterEngine class definition is the same) ...
class ArbiterEngine(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(ArbiterEngine, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, hidden_state = self.gru(embedded)
        output = self.output_layer(hidden_state.squeeze(0))
        return self.sigmoid(output)

class ArbiterDataset(Dataset):
    # ... (This class is the same) ...
    def __init__(self, dataset_path, tokenizer, max_prompt_len=128):
        self.data = torch.load(dataset_path)
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample['prompt']
        blending_curve = sample['blending_curve']
        tokenized_prompt = self.tokenizer(prompt, max_length=self.max_prompt_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokenized_prompt['input_ids'].squeeze(0)
        target_weight = torch.mean(blending_curve)
        return input_ids, target_weight

# --- Main Training Function ---
def train_new_arbiter(project_root, training_data_path):
    output_path = os.path.join(project_root, "assets/modules/arbiter_v1.0.pt")
    tokenizer_path = os.path.join(project_root, "assets/engines/analytical_engine_v1.0") # Use the correct path
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Arbiter training engaging on device: {device}")

    # --- THIS IS THE FIX ---
    # Load the same tokenizer as the main system
# Load the same tokenizer as the main system
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # --- THIS IS THE FINAL, CRITICAL FIX ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # --- END OF FIX ---

    full_dataset = ArbiterDataset(training_data_path, tokenizer)
    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    logger.info(f"Re-training Arbiter with unified vocabulary...")
    
    # Initialize the model with the CORRECT vocabulary size
    model = ArbiterEngine(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_eval_loss = float('inf')
    for epoch in tqdm(range(NUM_EPOCHS), desc="Re-Forging Arbiter"):
        model.train()
        for input_ids, target_weights in train_loader:
            input_ids, target_weights = input_ids.to(device), target_weights.to(device)
            optimizer.zero_grad()
            predicted_weights = model(input_ids).squeeze()
            loss = loss_function(predicted_weights, target_weights)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for input_ids, target_weights in eval_loader:
                input_ids, target_weights = input_ids.to(device), target_weights.to(device)
                predicted_weights = model(input_ids).squeeze()
                loss = loss_function(predicted_weights, target_weights)
                total_eval_loss += loss.item()
        
        avg_eval_loss = total_eval_loss / len(eval_loader)
        if (epoch + 1) % 5 == 0:
            tqdm.write(f"Epoch {epoch+1}/{NUM_EPOCHS}, Eval Loss: {avg_eval_loss:.4f}")
            
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(model.state_dict(), output_path)
            
    logger.info(f"New Arbiter re-forged with unified vocabulary. Model saved to {output_path}")
    return output_path
