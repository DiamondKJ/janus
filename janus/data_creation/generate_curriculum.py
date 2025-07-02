# /JANUS-CORE/janus/data_creation/generate_curriculum.py

import os
import json
import logging
import anthropic # <-- Use the Claude library
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

def create_curriculum(project_root, total_prompts=100, chunk_size=10):
    """
    Generates a curriculum by making multiple small, reliable API calls to Claude.
    """
    output_dir = os.path.join(project_root, "datasets")
    file_path = os.path.join(output_dir, "curriculum.json")
    
    logger.info(f"Attempting to generate a curriculum of {total_prompts} prompts in chunks of {chunk_size} using Claude API...")
    
    all_prompts = []
    num_chunks = total_prompts // chunk_size
    
    try:
        # Initialize the Claude client with the correct key
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except KeyError:
        logger.error("FATAL: ANTHROPIC_API_KEY environment variable not set.")
        return None

    # Use tqdm for a visible progress bar
    for _ in tqdm(range(num_chunks), desc="Curriculum Generation Chunks"):
        try:
            # The meta-prompt remains the same, as it's a universal instruction
            meta_prompt = f"""
            Generate a diverse list of {chunk_size} prompts for an advanced AI assistant. The list should test a wide range of cognitive functions. Include the following categories:
            - Analytical & Logic Puzzles
            - Creative & Imaginative
            - Ambiguous & Philosophical
            - Hybrid & Synthesis (e.g., 'Explain X using analogy Y')
            - Urgent & Action-Oriented

            Return ONLY a single JSON object with one key, "prompts", which contains a list of the {chunk_size} generated prompt strings. Your response must begin with '{{' and end with '}}'.
            """

            # --- THIS IS THE CORRECTED CODE FOR CLAUDE ---
            message = client.messages.create(
                model="claude-3-opus-20240229", # A powerful Claude model
                max_tokens=2048, # Generous token limit for the chunk
                messages=[
                    {"role": "user", "content": meta_prompt}
                ]
            )
            
            # The response from Claude is structured differently from Gemini's
            raw_text = message.content[0].text
            # --- END OF CORRECTION ---
            
            # Use the robust parser on the smaller, more reliable response
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}') + 1
            if start_index == -1 or end_index == 0:
                logger.warning("A chunk returned invalid data and will be skipped.")
                continue

            json_string = raw_text[start_index:end_index]
            data = json.loads(json_string)
            
            all_prompts.extend(data['prompts'])
            
            # Be kind to the API and avoid potential rate limiting
            time.sleep(1) 

        except Exception as e:
            logger.error(f"Failed to process one chunk: {e}. Skipping and continuing.")
            continue
            
    if not all_prompts:
        logger.error("No prompts could be generated after all attempts. Aborting.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, indent=2)
        
    logger.info(f"Successfully generated and aggregated {len(all_prompts)} prompts. Curriculum saved to {file_path}")
    return file_path