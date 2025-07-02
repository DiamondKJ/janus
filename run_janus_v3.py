# /JANUS-CORE/run_janus.py

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from janus.mind import run_interactive_session
from janus.data_creation.generate_curriculum import create_curriculum
from janus.data_creation.generate_arbiter_data import generate_arbiter_dataset
from janus.training.train_arbiter import train_new_arbiter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="JANUS-CORE: A Neuromorphic Cognitive Architecture.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--interactive', action='store_true', help='Run the latest stable version of Janus.')
    parser.add_argument('--full-pipeline', action='store_true', help='Execute the autonomous training pipeline.')
    
    args = parser.parse_args()

    if args.interactive:
        logger.info("Engaging INTERACTIVE MODE...")
        run_interactive_session(PROJECT_ROOT)
        
    elif args.full_pipeline:
        logger.info("Engaging FULL AUTONOMOUS TRAINING PIPELINE for Arbiter-Engine...")
        
        # --- Define paths for our check-and-resume protocol ---
        curriculum_path = os.path.join(PROJECT_ROOT, "datasets/curriculum.json")
        arbiter_data_path = os.path.join(PROJECT_ROOT, "datasets/arbiter_training_data.pt")

        # --- THIS IS THE FIX: The "Check-and-Resume" Logic ---
        
        # Stage 1: Generate Curriculum
        if not os.path.exists(curriculum_path):
            logger.info("[PIPELINE STAGE 1/3] Curriculum not found. Generating...")
            if not create_curriculum(PROJECT_ROOT, total_prompts=100, chunk_size=10):
                logger.error("Failed to create curriculum. Aborting pipeline.")
                return
            logger.info(f"Curriculum successfully created at: {curriculum_path}")
        else:
            logger.info(f"[PIPELINE STAGE 1/3] Found existing curriculum. Skipping generation.")

        # Stage 2: Generate Arbiter Data
        if not os.path.exists(arbiter_data_path):
            logger.info("[PIPELINE STAGE 2/3] Arbiter training data not found. Generating...")
            if not generate_arbiter_dataset(PROJECT_ROOT, curriculum_path):
                logger.error("Failed to create Arbiter training data. Aborting pipeline.")
                return
            logger.info(f"Arbiter training data successfully created at: {arbiter_data_path}")
        else:
            logger.info(f"[PIPELINE STAGE 2/3] Found existing Arbiter data. Skipping generation.")

        # Stage 3: Train the Arbiter
        logger.info("[PIPELINE STAGE 3/3] Forging the Arbiter-Engine v1.0...")
        new_arbiter_path = train_new_arbiter(PROJECT_ROOT, arbiter_data_path)
        if new_arbiter_path:
            logger.info("PIPELINE COMPLETE. A new Arbiter has been forged and saved.")
        else:
            logger.error("Failed to forge the new Arbiter. The mind has not evolved.")

    else:
        print("No operation specified."); parser.print_help()

if __name__ == "__main__":
    main()
