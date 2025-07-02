# /JANUS-CORE/generate_dream_corpus.py (FINAL VERSION)

import os
import sys
import logging
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the janus package to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from janus.memory import Hippocampus
from janus.somnium import SomniumEngine
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DreamCorpusGenerator:
    def __init__(self, project_root):
        self.project_root = project_root
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        logger.info("Initializing required modules...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # Load the tokenizer and the creative engine
        associative_path = os.path.join(self.project_root, "assets/engines/associative_engine_v1.0")
        self.tokenizer = AutoTokenizer.from_pretrained(associative_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_B = AutoModelForCausalLM.from_pretrained(associative_path, torch_dtype=torch.bfloat16, device_map=self.device)
        self.model_B.eval()
        
        self.api_client = None
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.api_client = genai
        except Exception: pass

        self.hippocampus = Hippocampus(self)
        self.somnium_engine = SomniumEngine(self)

    def generate_memories(self, prompts):
        logger.info(f"Generating {len(prompts)} salient memories...")
        for prompt in prompts:
            placeholder_response = "This was a foundational experience."
            high_salience_valence = {'creative_focus': 0.9, 'curiosity': 0.9}
            self.hippocampus.store_memory(prompt, placeholder_response, high_salience_valence)
        logger.info("All salient memories stored in Hippocampus.")

    # We can just borrow this function from the main harness
    def generate_dream_narrative(self, dream_prompt):
        dream_temperature = 1.2 
        messages = [{"role": "user", "content": dream_prompt}]
        full_prompt_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(full_prompt_string, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model_B.generate(**inputs, max_new_tokens=150, temperature=dream_temperature, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        inst_closing_tag = "[/INST]"
        tag_position = full_text.find(inst_closing_tag)
        return full_text[tag_position + len(inst_closing_tag):].strip() if tag_position != -1 else full_text

    def induce_dreams(self, num_dreams):
        logger.info(f"Inducing {num_dreams} dream cycles...")
        for i in range(num_dreams):
            logger.info(f"--- Dream Cycle {i+1}/{num_dreams} ---")
            self.somnium_engine._dream_phase()
        logger.info("Dream generation complete.")

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    dream_log_path = os.path.join(project_root, "reports/dreams.jsonl")
    if os.path.exists(dream_log_path):
        os.remove(dream_log_path)
        logger.info("Cleared previous dream log.")
        
    prompts_path = os.path.join(project_root, "prompts.txt")
    with open(prompts_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    generator = DreamCorpusGenerator(project_root)
    generator.generate_memories(prompts)
    
    num_dreams_to_generate = 50 
    generator.induce_dreams(num_dreams_to_generate)
    
    logger.info(f"\nSuccessfully generated {num_dreams_to_generate} dreams in '{dream_log_path}'.")

if __name__ == "__main__":
    main()