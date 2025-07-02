# /JANUS-CORE/janus/data_creation/generate_training_data.py
import os
import torch
import json
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# THIS IS THE FIX: Import from the stable core, not the harness.
from ..core import generate_simple_response, consult_overseer

def create_dissonance_dataset(project_root, curriculum_path):
    # ... (The rest of this file's code is correct and does not need to change)
    output_dir = os.path.join(project_root, "datasets")
    output_path = os.path.join(output_dir, "training_vectors.pt")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    with open(curriculum_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    logger.info(f"Loaded {len(prompts)} prompts from curriculum.")
    
    analytical_path = os.path.join(project_root, "assets/engines/analytical_engine_v1.0")
    associative_path = os.path.join(project_root, "assets/engines/associative_engine_v1.0")
    
    logger.info("Loading Janus engines and similarity model...")
    tokenizer = AutoTokenizer.from_pretrained(analytical_path)
    analytical_engine = AutoModelForCausalLM.from_pretrained(analytical_path, torch_dtype=torch.bfloat16, device_map=device)
    associative_engine = AutoModelForCausalLM.from_pretrained(associative_path, torch_dtype=torch.bfloat16, device_map=device)
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    training_data = []
    
    for prompt in tqdm(prompts, desc="Generating Training Vectors"):
        analytical_thought = generate_simple_response(prompt, analytical_engine, tokenizer, device)
        associative_thought = generate_simple_response(prompt, associative_engine, tokenizer, device)
        true_label = consult_overseer(prompt, analytical_thought, associative_thought)
        
        if true_label is not None:
            with torch.no_grad():
                embeddings = similarity_model.encode([analytical_thought, associative_thought], convert_to_tensor=True, device=device)
                interference_vector = torch.abs(embeddings[0] - embeddings[1])
            
            training_data.append({
                "interference_vector": interference_vector.cpu(),
                "label": torch.tensor(true_label, dtype=torch.float32)
            })

    if not training_data:
        logger.error("No training data generated."); return None

    logger.info(f"Generated {len(training_data)} high-quality training vectors.")
    torch.save(training_data, output_path)
    return output_path