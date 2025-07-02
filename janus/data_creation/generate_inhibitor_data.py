# /JANUS-CORE/janus/data_creation/generate_inhibitor_data.py

import os
import torch
import json
import logging
import anthropic
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# Import from our core module, if needed, or define locally
def generate_long_response(prompt, model, tokenizer, device):
    """Generates a long, rambling response to be edited by the Overseer."""
    messages = [{"role": "user", "content": prompt}]
    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_string, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, # Encourage a long response
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

def get_edited_response(prompt, long_response, client):
    """Uses the Claude 'Ruthless Editor' to get the perfect concise version."""
    try:
        meta_prompt = f"""
        You are a ruthless but brilliant editor. Your only goal is to achieve maximum clarity and conciseness.
        Read the following user prompt and the AI's rambling response.
        Edit the AI's response down to the point where the user's core query is fully answered, and any further text is redundant or superfluous.
        Do not add any commentary. Return ONLY the edited, 'perfect' version of the response.

        USER PROMPT:
        ---
        {prompt}
        ---

        AI's RAMBLING RESPONSE:
        ---
        {long_response}
        ---

        YOUR EDITED, PERFECT RESPONSE:
        """
        message = client.messages.create(
            model="claude-3-haiku-20240307", # Use Haiku for speed and cost-effectiveness
            max_tokens=512,
            messages=[{"role": "user", "content": meta_prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        logger.error(f"Ruthless Editor API call failed: {e}")
        return None

def create_inhibitor_dataset(project_root, curriculum_path):
    """The main function to generate the dataset for the Inhibitor Engine."""
    output_dir = os.path.join(project_root, "datasets")
    output_path = os.path.join(output_dir, "inhibitor_training_data.jsonl")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load curriculum
    with open(curriculum_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    logger.info(f"Loaded {len(prompts)} prompts from curriculum.")

    # Initialize Claude client
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except KeyError:
        logger.error("FATAL: ANTHROPIC_API_KEY not set.")
        return None

    # Load local engine for generating rambling text
    associative_path = os.path.join(project_root, "assets/engines/associative_engine_v1.0")
    tokenizer = AutoTokenizer.from_pretrained(associative_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    associative_engine = AutoModelForCausalLM.from_pretrained(associative_path, torch_dtype=torch.bfloat16, device_map=device)

    logger.info("Starting data generation for Inhibitor Engine...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt in tqdm(prompts, desc="Distilling Inhibitor Data"):
            # 1. Generate long response locally (the "unfinished painting")
            long_response = generate_long_response(prompt, associative_engine, tokenizer, device)
            
            # 2. Get the perfect, edited version from the Overseer
            edited_response = get_edited_response(prompt, long_response, client)
            
            if not edited_response:
                logger.warning(f"Skipping prompt '{prompt[:30]}...' due to editor failure.")
                continue

            # 3. Tokenize and create labeled examples
            long_tokens = tokenizer.tokenize(edited_response + long_response[len(edited_response):])
            good_tokens_len = len(tokenizer.tokenize(edited_response))

            for i in range(1, len(long_tokens)):
                # The context is the prompt + the response so far
                context = prompt + " " + tokenizer.convert_tokens_to_string(long_tokens[:i])
                
                # The label is 0 (CONTINUE) if we are still inside the "good" part
                # and 1 (STOP) for the first token of the "bad" part.
                label = 0 if i < good_tokens_len else 1
                
                f.write(json.dumps({"text": context, "label": label}) + '\n')

                # We only need one "STOP" example per prompt
                if label == 1:
                    break
            
            time.sleep(1) # Be kind to the API

    logger.info(f"Inhibitor training data generation complete. Saved to {output_path}")
    return output_path