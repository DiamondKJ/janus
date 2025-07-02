# /JANUS-CORE/debug_weaver.py

import os
import sys
import torch
import torch.nn as nn
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A simple helper to print tensor stats
def print_tensor_stats(name, tensor):
    if tensor is None:
        print(f"--- {name}: is None")
        return
    print(
        f"--- {name}:\n"
        f"    Shape: {tensor.shape}\n"
        f"    DType: {tensor.dtype}\n"
        f"    Has NaN: {torch.any(torch.isnan(tensor))}\n"
        f"    Has Inf: {torch.any(torch.isinf(tensor))}\n"
        f"    Min val: {tensor.min().item():.4f}\n"
        f"    Max val: {tensor.max().item():.4f}\n"
        f"    Mean val: {tensor.mean().item():.4f}\n"
    )

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Running Debug on device: {device}")

    # --- 1. Load Minimal Required Models ---
    logger.info("Loading models...")
    analytical_path = os.path.join(project_root, "assets/engines/analytical_engine_v1.0")
    tokenizer = AutoTokenizer.from_pretrained(analytical_path)
    model_A = AutoModelForCausalLM.from_pretrained(analytical_path, torch_dtype=torch.bfloat16, device_map=device)
    model_B = AutoModelForCausalLM.from_pretrained(analytical_path.replace("analytical", "associative"), torch_dtype=torch.bfloat16, device_map=device) # Load model B
    model_A_embeddings = model_A.get_input_embeddings().weight.data

    # --- 2. Load the Projection Head and "Current Self" ---
    # We will simulate having a "Current Self" vector
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    projection_head = nn.Linear(model_A.config.hidden_size, similarity_model.get_sentence_embedding_dimension()).to(device)
    projection_head_path = os.path.join(project_root, "assets/modules/projection_head.pt")
    if os.path.exists(projection_head_path):
        projection_head.load_state_dict(torch.load(projection_head_path))
        logger.info("Loaded TRAINED Projection Head.")
    
    current_self_vector = torch.randn(similarity_model.get_sentence_embedding_dimension()).to(device)
    current_self_vector = torch.nn.functional.normalize(current_self_vector, p=2, dim=0)
    logger.info("Created a dummy 'Current Self' vector.")

    # --- 3. Run ONE step of the Weaver Loop ---
    logger.info("\n--- BEGINNING WEAVER TRACE ---")
    
    # Hardcoded parameters for this test
    user_prompt = "What should I think about today?"
    live_blend_weight = 0.5
    temperature = 0.7
    
    messages = [{"role": "user", "content": user_prompt}]
    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    current_ids = tokenizer(prompt_string, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        # --- Divergent Forward Pass ---
        outputs_A = model_A(input_ids=current_ids, use_cache=False)
        outputs_B = model_B(input_ids=current_ids, use_cache=False)
        logits_A = outputs_A.logits[:, -1, :]
        logits_B = outputs_B.logits[:, -1, :]
        print_tensor_stats("RAW LOGITS A", logits_A)
        print_tensor_stats("RAW LOGITS B", logits_B)

        # --- Logit Fusion ---
        fused_logits = (1 - live_blend_weight) * logits_A + live_blend_weight * logits_B
        print_tensor_stats("FUSED LOGITS", fused_logits)

        # --- Schema Gating ---
        k=20
        bonus_strength=0.1
        fused_logits_stable = torch.nan_to_num(fused_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        top_k_logits, top_k_indices = torch.topk(fused_logits_stable, k, dim=-1)
        top_k_embeddings = model_A_embeddings[top_k_indices]
        projected_embeddings = projection_head(top_k_embeddings.float()) # Ensure float32 for projection
        projected_embeddings = torch.nn.functional.normalize(projected_embeddings, p=2, dim=-1)
        similarities = torch.nn.functional.cosine_similarity(projected_embeddings, current_self_vector, dim=-1)
        bonus = torch.sigmoid(similarities * 5) * bonus_strength
        bonus_tensor = torch.zeros_like(fused_logits).scatter_(-1, top_k_indices, bonus.to(fused_logits.dtype))
        gated_final_logits = fused_logits + bonus_tensor
        print_tensor_stats("GATED FINAL LOGITS", gated_final_logits)

        # --- Scaling and Softmax ---
        scaled_logits = gated_final_logits / temperature
        print_tensor_stats("SCALED LOGITS", scaled_logits)
        
        probs = torch.softmax(scaled_logits, dim=-1)
        print_tensor_stats("FINAL PROBS", probs)

        # --- Final Check ---
        is_corrupted = torch.any(torch.isinf(probs)) or torch.any(torch.isnan(probs)) or torch.any(probs < 0)
        logger.info(f"\n--- FINAL VERDICT: Probability tensor is corrupted? --- {is_corrupted.item()}")

if __name__ == "__main__":
    main()