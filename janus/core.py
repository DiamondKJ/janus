# /JANUS-CORE/janus/core.py

import torch
import logging
import json
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

def generate_simple_response(prompt, model, tokenizer, device, temperature=0.1, max_tokens=64):
    """The primitive function for generating a thought from a single engine."""
    messages = [{"role": "user", "content": prompt}]
    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_string, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

def consult_overseer(prompt, thought_a, thought_b):
    """The primitive function for consulting the AI Overseer."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        meta_prompt = f"""You are a cognitive architecture analyst. Your task is to determine if two internal interpretations of a prompt represent 'Cognitive Dissonance' (1) or 'Cognitive Synergy' (0).
        Prompt: "{prompt}"
        Interpretation A: "{thought_a}"
        Interpretation B: "{thought_b}"
        Respond with ONLY a single JSON object: {{"dissonance": 1}} or {{"dissonance": 0}}."""
        response = model.generate_content(meta_prompt)
        return json.loads(response.text.strip().replace("```json", "").replace("```", ""))['dissonance']
    except Exception as e:
        logger.error(f"Overseer consultation failed: {e}"); return None