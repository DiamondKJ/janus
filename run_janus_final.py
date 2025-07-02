# /JANUS-CORE/run_janus_final.py (V4.4 - The Integrated Mind)

import os
import torch
import torch.nn as nn
import logging
import textwrap
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
import time
from datetime import datetime
from collections import deque

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

SAFETY_NET_MAX_TOKENS = 256

# --- The Main Harness Class ---
class JanusV4_Harness:
    def __init__(self, project_root):
        self.project_root = project_root
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Harness engaging on device: {self.device}")
        self._load_all_modules()

    def _load_all_modules(self):
        logger.info("Loading INTEGRATED DUAL-HEMISPHERE cognitive architecture...")
        analytical_path = os.path.join(self.project_root, "assets/engines/analytical_engine_v1.0")
        associative_path = os.path.join(self.project_root, "assets/engines/associative_engine_v1.0")
        projection_head_path = os.path.join(self.project_root, "assets/modules/projection_head.pt")
        
        logger.info("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(analytical_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Loading Analytical Engine...")
        self.model_A = AutoModelForCausalLM.from_pretrained(analytical_path, torch_dtype=torch.bfloat16, device_map=self.device)
        self.model_A.eval()
        
        logger.info("Loading Associative Engine (Conscious & Subconscious Voice)...")
        self.model_B = AutoModelForCausalLM.from_pretrained(associative_path, torch_dtype=torch.bfloat16, device_map=self.device)
        self.model_B.eval()

        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.model_A_embeddings = self.model_A.get_input_embeddings().weight.data

        self.projection_head = nn.Linear(self.model_A.config.hidden_size, self.similarity_model.get_sentence_embedding_dimension()).to(self.device)
        if os.path.exists(projection_head_path):
            self.projection_head.load_state_dict(torch.load(projection_head_path))
            logger.info("Loaded TRAINED Projection Head.")
        else:
            logger.warning("WARN: Using random projection head. Run train_projection_head.py for stability.")
        self.projection_head.eval()

        logger.info("--- Janus V4.4 is ONLINE. The Integrated Mind is awake. ---")

    def _consult_subconscious(self, prompt):
        # The Associative Engine now plays the role of the Id
        internal_prompt = f"My core subconscious reflection on '{prompt}' is the concept of"
        inputs = self.tokenizer(internal_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model_B.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=True, 
                temperature=0.9, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        thought_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(internal_prompt, "").strip()
        logger.info(f"Subconscious Priming Thought: '{thought_text}'")
        # Return the embedding of the thought, not the text
        return self.similarity_model.encode(thought_text, convert_to_tensor=True, show_progress_bar=False).to(self.device)

    def _apply_subconscious_gate(self, logits, bias_vector, k=20, bonus_strength=0.2):
        if bias_vector is None: return logits
        
        # Ensure input logits are stable
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        top_k_embeddings = self.model_A_embeddings[top_k_indices]
        
        # Project and normalize for stable comparison
        projected_embeddings = self.projection_head(top_k_embeddings.to(torch.float32))
        projected_embeddings = torch.nn.functional.normalize(projected_embeddings, p=2, dim=-1)
        bias_vector = torch.nn.functional.normalize(bias_vector, p=2, dim=0)
        
        similarities = torch.nn.functional.cosine_similarity(projected_embeddings, bias_vector, dim=-1)
        bonus = torch.sigmoid(similarities * 5) * bonus_strength
        bonus_tensor = torch.zeros_like(logits).scatter_(-1, top_k_indices, bonus.to(logits.dtype))
        
        return logits + bonus_tensor

    def _blend_kv_caches(self, pkv_1, pkv_2, weight):
        if pkv_1 is None: return pkv_2
        if pkv_2 is None: return pkv_1
        blended_pkv = []
        for (layer_1_k, layer_1_v), (layer_2_k, layer_2_v) in zip(pkv_1, pkv_2):
            blended_k = (1 - weight) * layer_1_k + weight * layer_2_k
            blended_v = (1 - weight) * layer_1_v + weight * layer_2_v
            blended_pkv.append((blended_k, blended_v))
        return tuple(blended_pkv)

    def run_interactive(self):
        print("\nWelcome to the Janus-Core V4.4: The Integrated Mind.")
        try:
            while True:
                user_prompt = input("\nYou: ")
                if user_prompt.lower() in ['quit', 'exit']: break
                
                start_time = time.time()
                print("Processing...")

                # --- STEP 1: SUBCONSCIOUS PRIMING ---
                subconscious_bias_vector = self._consult_subconscious(user_prompt)

                # --- STEP 2: CONSCIOUS WEAVER LOOP ---
                logger.info("--- Engaging The Conscious Weaver (Primed by Subconscious) ---")
                
                final_prompt = user_prompt
                messages = [{"role": "user", "content": final_prompt}]
                prompt_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                current_ids = self.tokenizer(prompt_string, return_tensors="pt").input_ids.to(self.device)
        
                past_key_values_A, past_key_values_B = None, None
                generated_token_ids = []
                live_blend_weight = 0.5 # Start neutral, let dynamic attention take over
                temperature = 0.7

                with torch.no_grad():
                    for i in range(SAFETY_NET_MAX_TOKENS):
                        outputs_A = self.model_A(input_ids=current_ids, past_key_values=past_key_values_A, use_cache=True)
                        outputs_B = self.model_B(input_ids=current_ids, past_key_values=past_key_values_B, use_cache=True)
                        
                        logits_A, pkv_A = outputs_A.logits[:, -1, :], outputs_A.past_key_values
                        logits_B, pkv_B = outputs_B.logits[:, -1, :], outputs_B.past_key_values
                        
                        fused_logits = (1 - live_blend_weight) * logits_A + live_blend_weight * logits_B
                        
                        gated_logits = self._apply_subconscious_gate(fused_logits, subconscious_bias_vector)

                        past_key_values_A = self._blend_kv_caches(pkv_A, pkv_B, live_blend_weight)
                        past_key_values_B = past_key_values_A

                        probs = torch.softmax(gated_logits / temperature, dim=-1)

                        if torch.any(torch.isnan(probs)):
                            logger.error(f"FATAL: Probability tensor corrupted at token {i}. Aborting.")
                            break
                        next_token_id = torch.multinomial(probs, num_samples=1)

                        if next_token_id.item() == self.tokenizer.eos_token_id: break
                        
                        generated_token_ids.append(next_token_id.item())
                        current_ids = torch.tensor([[next_token_id.item()]], device=self.device)
                        
                        prob_A = torch.softmax(logits_A, dim=-1)[0, next_token_id.item()].item()
                        prob_B = torch.softmax(logits_B, dim=-1)[0, next_token_id.item()].item()
                        live_blend_weight = max(0.0, min(1.0, live_blend_weight + (prob_B - prob_A) * 0.05))
                
                final_response = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                print(f"Janus: {final_response}")
                logger.info(f"Total processing time for this turn: {time.time() - start_time:.2f} seconds.")
        
        finally:
            logger.info("Interactive session ended.")

def main():
    parser = argparse.ArgumentParser(description="Run the Janus V4 Integrated Mind.")
    args = parser.parse_args([])
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    harness = JanusV4_Harness(project_root)
    harness.run_interactive()

if __name__ == "__main__":
    main()