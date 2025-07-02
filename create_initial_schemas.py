# /JANUS-CORE/create_initial_schemas.py

import os
import sys
import logging
import json
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # --- 1. Load Labeled Dataset ---
    dataset_path = os.path.join(project_root, 'datasets/labeled_dreams.csv')
    if not os.path.exists(dataset_path):
        logger.error(f"Labeled dataset not found at '{dataset_path}'. Please run label_corpus.py first.")
        return
        
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} labeled dreams from corpus.")

    # --- 2. Initialize Encoder ---
    logger.info("Initializing Sentence Transformer...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # --- 3. Calculate Centroid for Each Schema ---
    schema_centroids = []
    unique_labels = df['schema_label'].unique()
    logger.info(f"Found unique schemas: {unique_labels}")

    for label in unique_labels:
        # Get all dreams associated with this label
        dreams_for_label = df[df['schema_label'] == label]['dream_text'].tolist()
        
        if not dreams_for_label:
            continue

        logger.info(f"Calculating centroid for schema '{label}' from {len(dreams_for_label)} dreams...")
        
        # Encode all dreams and calculate the mean vector (the centroid)
        with torch.no_grad():
            embeddings = encoder.encode(dreams_for_label, convert_to_tensor=True, show_progress_bar=True)
            centroid_vector = torch.mean(embeddings, dim=0)
            
        schema_centroids.append({
            "name": label,
            "vector": centroid_vector.cpu(), # Store on CPU
            "confidence": 1.0 # Initial confidence is high
        })

    # --- 4. Save the Initial Schema File ---
    output_path = os.path.join(project_root, 'assets/memory/schemas.pt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(schema_centroids, output_path)
    logger.info(f"Successfully created and saved initial schema centroids to '{output_path}'.")
    logger.info(f"Total schemas created: {len(schema_centroids)}")

if __name__ == "__main__":
    main()