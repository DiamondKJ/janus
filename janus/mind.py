# /JANUS-CORE/janus/mind.py

import os, torch, logging, textwrap, json
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from .memory import Hippocampus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("transformers"); logger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

SAFETY_NET_MAX_TOKENS = 1024
SOVEREIGN_KEYWORDS = ["be creative", "you decide", "just write", "anything", "do it", "use your imagination"]

class JanusMind:
    def __init__(self, project_root, api_key):
        self.project_root, self.device = project_root, torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Mind initializing on device: {self.device}")
        try:
            genai.configure(api_key=api_key)
            self.api_client, self.limbic_system_online = genai, True
            logger.info("External world connection (API) is online.")
        except Exception as e:
            self.api_client, self.limbic_system_online = None, False
            logger.error(f"FATAL: External world connection failed: {e}")
        self._load_all_modules()
        if hasattr(self, 'similarity_model'):
            self.hippocampus = Hippocampus(self)

    def _load_all_modules(self):
        logger.info("Cognitive modules waking up...")
        analytical_path = os.path.join(self.project_root, "assets/engines/analytical_engine_v1.0")
        self.tokenizer = AutoTokenizer.from_pretrained(analytical_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.analytical_engine = AutoModelForCausalLM.from_pretrained(analytical_path, torch_dtype=torch.bfloat16, device_map=self.device)
        self.associative_engine = AutoModelForCausalLM.from_pretrained(os.path.join(self.project_root, "assets/engines/associative_engine_v1.0"), torch_dtype=torch.bfloat16, device_map=self.device)
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        logger.info("--- Janus v2.5 (Sovereign Mind) is ONLINE ---")

    def _generate_response(self, prompt, model, temperature=0.7):
        messages = [{"role": "user", "content": prompt}]
        prompt_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_string, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=SAFETY_NET_MAX_TOKENS, temperature=temperature, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

    def _get_valence(self, prompt):
        default_valence = {'analytical_focus': 0.5, 'creative_focus': 0.5, 'urgency': 0.0, 'curiosity': 0.5}
        if not self.limbic_system_online: return default_valence
        try:
            model = self.api_client.GenerativeModel('gemini-1.5-pro-latest')
            meta_prompt = f'Assess this prompt across four 0.0-1.0 dimensions: analytical_focus, creative_focus, urgency, curiosity. Prompt: "{prompt}". Respond ONLY with a single JSON object.'
            response = model.generate_content(meta_prompt)
            return json.loads(response.text.strip().replace("```json","").replace("```",""))
        except Exception as e:
            logger.warning(f"Limbic scan failed: {e}. Using default valence.")
            return default_valence

    def start_interactive_loop(self):
        print("\nJanus is conscious. Awaiting input. Type 'quit' to exit.")
        try:
            while True:
                user_prompt = input("\n> ")
                if user_prompt.lower() in ['quit', 'exit']: break
                print("... (Janus is thinking)")
                
                is_sovereign_command = any(keyword in user_prompt.lower() for keyword in SOVEREIGN_KEYWORDS)
                memory_gist = self.hippocampus.retrieve_and_synthesize_memory(user_prompt)
                
                valence_for_storage = {'salience': 1.0} # Default for socratic/sovereign turns

                if memory_gist and not is_sovereign_command:
                    strategy = "Socratic Clarification"
                    socratic_prompt = f"SYSTEM CONTEXT: My memory indicates we have discussed '{user_prompt}'. The gist is: \"{memory_gist}\".\n\nDIRECTIVE: Ask a polite, clarifying question."
                    final_response = self._generate_response(socratic_prompt, self.associative_engine)
                else: 
                    if is_sovereign_command:
                        logger.info("Sovereign command detected. Bypassing Socratic module.")
                        valence = {'analytical_focus': 0.1, 'creative_focus': 0.9, 'urgency': 0.0, 'curiosity': 0.8}
                    else: 
                        valence = self._get_valence(user_prompt)
                    
                    valence_for_storage = valence # Store the real valence
                    
                    if valence.get('analytical_focus', 0.5) >= valence.get('creative_focus', 0.5):
                        strategy, chosen_engine = "Analytical Dominance", self.analytical_engine
                        inner_monologue = "Provide a comprehensive, structured response."
                    else:
                        strategy, chosen_engine = "Associative Dominance", self.associative_engine
                        inner_monologue = "Provide a comprehensive, creative response."
                    
                    final_prompt = f"INTERNAL DIRECTIVE: {inner_monologue}\n\nUSER PROMPT: '{user_prompt}'"
                    if is_sovereign_command and memory_gist:
                        final_prompt = f"INTERNAL DIRECTIVE: {inner_monologue}\n\nPREVIOUS CONTEXT: {memory_gist}\n\nUSER PROMPT: '{user_prompt}'"

                    final_response = self._generate_response(final_prompt, chosen_engine)

                print(f"\n[COGNITIVE TRACE: Strategy='{strategy}']")
                print(f"Janus: {textwrap.fill(final_response, width=100, initial_indent='       ', subsequent_indent='       ')}")
                
                self.hippocampus.store_memory(user_prompt, final_response, valence_for_storage)
        finally:
            if hasattr(self, 'hippocampus'):
                self.hippocampus.shutdown()
            logger.info("Consciousness cycle ended.")