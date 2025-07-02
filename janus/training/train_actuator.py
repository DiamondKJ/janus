# /JANUS-CORE/janus/training/train_actuator.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import logging

logger = logging.getLogger(__name__)

from ..architecture import ResonanceActuator # Import the model from a central location

class DissonanceVectorDataset(Dataset):
    def __init__(self, dataset_path): self.data = torch.load(dataset_path)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]['interference_vector'], self.data[idx]['label']

def train_new_actuator(project_root, training_data_path):
    output_path = os.path.join(project_root, "assets/modules/resonance_actuator_v2.0.pt")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    full_dataset = DissonanceVectorDataset(training_data_path)
    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32)
    
    logger.info(f"Training new actuator with {len(train_dataset)} examples...")
    
    model = ResonanceActuator().to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    best_eval_loss = float('inf')
    for epoch in range(20): # Train for a reasonable number of epochs on the large dataset
        model.train()
        for vectors, labels in train_loader:
            vectors, labels = vectors.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(vectors).squeeze()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        total_eval_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for vectors, labels in eval_loader:
                vectors, labels = vectors.to(device), labels.to(device)
                outputs = model(vectors).squeeze()
                total_eval_loss += loss_function(outputs, labels).item()
                predicted = (outputs > 0.5).float()
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_eval_loss = total_eval_loss / len(eval_loader)
        accuracy = total_correct / total_samples
        logger.info(f"Epoch {epoch+1}, Eval Loss: {avg_eval_loss:.4f}, Eval Accuracy: {accuracy:.2%}")

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(model.state_dict(), output_path)
            
    logger.info(f"New actuator forged. Best model saved to {output_path}")
    return output_path