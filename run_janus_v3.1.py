# /JANUS-CORE/run_janus_v3.1.py

import os
import torch
import torch.nn as nn
import logging
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys

# Add the janus package to the path to import the Arbiter architecture
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from janus.training.train_arbiter import ArbiterEngine

# --- 1. Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

SAFETY_NET_MAX_TOKENS = 30

# --- 2. The Final Weaver & Harness ---
class JanusV3_Harness:
    def __init__(self, project_root):
        self.project_root = project_root
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Janus v3.1 Harness engaging on device: {self.device}")
        self._load_all_modules()

    def _load_all_modules(self):
        logger.info("Loading cognitive architecture: Both Hemispheres and the new Arbiter...")
        analytical_path = os.path.join(self.project_root, "assets/engines/analytical_engine_v1.0")
        associative_path = os.path.join(self.project_root, "assets/engines/associative_engine_v1.0")
        arbiter_path = os.path.join(self.project_root, "assets/modules/arbiter_v1.0.pt")
        
        # --- Load Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(analytical_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # --- Load Both Hemispheres ---
        self.model_A = AutoModelForCausalLM.from_pretrained(analytical_path, torch_dtype=torch.bfloat16, device_map=self.device)
        self.model_A.eval()
        self.model_B = AutoModelForCausalLM.from_pretrained(associative_path, torch_dtype=torch.bfloat16, device_map=self.device)
        self.model_B.eval()
        
        # --- Load the Forged Arbiter ---
        self.arbiter = ArbiterEngine(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=256,
            hidden_dim=512,
            output_dim=1
        ).to(self.device)
        self.arbiter.load_state_dict(torch.load(arbiter_path, map_location=self.device))
        self.arbiter.eval()
        logger.info("--- Janus v3.1 is ONLINE. True Synthesis mode is active. ---")

    def generate(self, prompt, temperature=0.7):
        logger.info("--- Beginning Arbiter-guided token-by-token synthesis ---")
        
        # Tokenize the initial prompt to feed to the Arbiter
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Get the single, average blending weight from our v1 Arbiter
        with torch.no_grad():
            blending_weight = self.arbiter(prompt_ids).item()
        
        logger.info(f"Arbiter has set average blending weight for this prompt to: {blending_weight:.2f}")

        # Prepare the full prompt for the generative models
        messages = [{"role": "user", "content": prompt}]
        prompt_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output_ids = self.tokenizer(prompt_string, return_tensors="pt").input_ids.to(self.device)

        print(f"\nJanus v3.1: ", end="", flush=True)

        with torch.no_grad():
            for _ in range(SAFETY_NET_MAX_TOKENS):
                # --- The Optimized Weaver Loop ---
                # Default to None
                logits_A, logits_B = None, None
                
                # The Arbiter's weight determines which engines to run
                # Low weight -> Analytical is dominant
                if blending_weight < 0.4:
                    logits_A = self.model_A(output_ids).logits[:, -1, :]
                    final_logits = logits_A
                # High weight -> Associative is dominant
                elif blending_weight > 0.6:
                    logits_B = self.model_B(output_ids).logits[:, -1, :]
                    final_logits = logits_B
                # Middle weight -> True blending is required
                else:
                    logits_A = self.model_A(output_ids).logits[:, -1, :]
                    logits_B = self.model_B(output_ids).logits[:, -1, :]
                    final_logits = (1 - blending_weight) * logits_A + blending_weight * logits_B
                
                # --- Sampling ---
                scaled_logits = final_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                output_ids = torch.cat([output_ids, next_token_id], dim=-1)
        
        full_response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        final_answer = full_response_text.split("[/INST]")[-1].strip()

        print(textwrap.fill(final_answer, width=100))
        print("\n--- Synthesis Complete ---")
        return final_answer

# --- Main Entry Point ---
def main():
    parser = argparse.ArgumentParser(description="Run the Janus v3.1 Synthesis Engine.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to send to Janus.")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize and run the harness
    harness = JanusV3_Harness(project_root)
    harness.generate(args.prompt)

if __name__ == "__main__":
    main()