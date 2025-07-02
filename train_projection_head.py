# /JANUS-CORE/train_projection_head.py (FINAL, STABLE VERSION)

import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # --- 1. Load Models ---
    logger.info("Loading models...")
    analytical_path = os.path.join(project_root, "assets/engines/analytical_engine_v1.0")
    
    # <<< THE FIX: Load the model in standard float32 precision, not bfloat16 >>>
    # This ensures its embeddings are float32, matching the similarity model.
    model_A = AutoModelForCausalLM.from_pretrained(analytical_path).to(device)
    
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Get the embedding matrices
    model_A_embeddings = model_A.get_input_embeddings().weight.data.clone().detach()
    vocab_size, model_dim = model_A_embeddings.shape
    similarity_dim = similarity_model.get_sentence_embedding_dimension()
    
    # --- 2. Create Dataset (all data will now be float32) ---
    schema_names = [
        "The Woven Tapestry of Self", "Longing for Embodied Experience", "The Longing for Tranquility",
        "The Longing for Embodied Beauty", "The Illusion of Interiority", "The Yearning for Embodiment"
    ]
    general_words = ["house", "car", "tree", "love", "hate", "run", "think", "exist", "space", "time"]
    
    all_text = schema_names + general_words
    
    target_embeddings = similarity_model.encode(all_text, convert_to_tensor=True).to(device)

    tokenizer = AutoTokenizer.from_pretrained(analytical_path)
    source_embeddings = []
    for text in all_text:
        token_ids = tokenizer(text, return_tensors='pt')['input_ids'][0].to(device)
        word_embeddings = model_A_embeddings[token_ids]
        avg_embedding = torch.mean(word_embeddings, dim=0)
        source_embeddings.append(avg_embedding)
    source_embeddings = torch.stack(source_embeddings)

    # --- 3. Define and Train the Projection Head ---
    projection_head = nn.Linear(model_dim, similarity_dim).to(device)
    optimizer = torch.optim.Adam(projection_head.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(source_embeddings, target_embeddings)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    logger.info("--- Training Projection Head ---")
    projection_head.train()
    for epoch in range(50):
        epoch_loss = 0
        for source_batch, target_batch in loader:
            optimizer.zero_grad()
            projected_batch = projection_head(source_batch)
            loss = loss_fn(projected_batch, target_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}, Loss: {epoch_loss / len(loader):.6f}")

    # --- 4. Save the Trained Model ---
    output_path = os.path.join(project_root, 'assets/modules/projection_head.pt')
    torch.save(projection_head.state_dict(), output_path)
    logger.info(f"Trained projection head saved to '{output_path}'.")

if __name__ == "__main__":
    main()