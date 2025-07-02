
import os
import json
import logging
import anthropic
import torch
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

def generate_arbiter_dataset(project_root, curriculum_path):
    """
    Generates a dataset of "blending curves" for training the Arbiter module.
    Each item consists of a prompt and a sequence of blending weights.
    """
    output_dir = os.path.join(project_root, "datasets")
    output_path = os.path.join(output_dir, "arbiter_training_data.pt")
    
    # Load the curriculum of prompts
    with open(curriculum_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    logger.info(f"Loaded {len(prompts)} prompts from curriculum to generate Arbiter data.")
    
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except KeyError:
        logger.error("FATAL: ANTHROPIC_API_KEY environment variable not set.")
        return None

    arbiter_dataset = []
    
    for prompt in tqdm(prompts, desc="Generating Arbiter Blending Plans"):
        try:
            meta_prompt = f"""
            You are a master synthesizer, blending the outputs of two expert AIs: a hyper-logical 'Logician' and a hyper-creative 'Poet'.
            Your task is to analyze the user's prompt and generate a step-by-step "blending plan" for how you would construct the perfect response, token by token.

            USER PROMPT: "{prompt}"

            Analyze the prompt. Then, for each "phase" of the ideal response, describe which expert should lead and why.
            The 'leading_expert' can be "Logician", "Poet", or "Both".
            The 'estimated_tokens' should be a reasonable guess for the length of that phase.

            Respond ONLY with a single JSON object with a "plan" key, which is a list of these phase objects.
            Example Response:
            {{
              "plan": [
                {{ "phase": "Introduction", "leading_expert": "Logician", "justification": "Start with a precise, factual definition.", "estimated_tokens": 20 }},
                {{ "phase": "Analogy Setup", "leading_expert": "Poet", "justification": "Introduce the creative frame.", "estimated_tokens": 30 }},
                {{ "phase": "Synthesis", "leading_expert": "Both", "justification": "Weave the concepts together.", "estimated_tokens": 50 }}
              ]
            }}
            """
            
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[{"role": "user", "content": meta_prompt}]
            )
            raw_text = message.content[0].text
            
            # Robust JSON parsing
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}') + 1
            if start_index == -1 or end_index == 0:
                logger.warning(f"Skipping prompt due to invalid JSON response: {prompt}")
                continue
            
            json_string = raw_text[start_index:end_index]
            plan_data = json.loads(json_string)['plan']

            # --- Convert the plan into a sequence of blending weights ---
            blending_curve = []
            weight_map = {"Logician": 0.1, "Poet": 0.9, "Both": 0.5}
            
            for phase in plan_data:
                weight = weight_map.get(phase['leading_expert'], 0.5) # Default to 0.5 if key is weird
                num_tokens = int(phase['estimated_tokens'])
                blending_curve.extend([weight] * num_tokens)
            
            # Ensure the curve isn't empty
            if not blending_curve:
                continue

            arbiter_dataset.append({
                "prompt": prompt,
                "blending_curve": torch.tensor(blending_curve, dtype=torch.float32)
            })
            
            time.sleep(1) # Be kind to the API

        except Exception as e:
            logger.error(f"Failed to process prompt '{prompt}': {e}. Skipping.")
            continue
            
    if not arbiter_dataset:
        logger.error("Could not generate any Arbiter training data. Aborting.")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    torch.save(arbiter_dataset, output_path)
    
    logger.info(f"Successfully generated {len(arbiter_dataset)} blending curves. Data saved to {output_path}")
    return output_path
