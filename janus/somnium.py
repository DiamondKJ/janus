# /JANUS-CORE/janus/somnium.py (V3.2 - Evolving Mind)

import logging
import torch
import json
import os
from datetime import datetime
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import util

logger = logging.getLogger(__name__)

class SomniumEngine:
    def __init__(self, mind):
        self.mind = mind
        self.dream_log_path = os.path.join(mind.project_root, "reports/dreams.jsonl")
        self.belief_trajectory_path = os.path.join(mind.project_root, "reports/belief_trajectory.jsonl")
        self.schema_path = os.path.join(mind.project_root, "assets/memory/schemas.pt")
        
        # Load the dynamic schema centroids
        self.schemas = []
        if os.path.exists(self.schema_path):
            try:
                self.schemas = torch.load(self.schema_path)
                logger.info(f"Somnium Engine loaded {len(self.schemas)} dynamic schema centroids.")
            except Exception as e:
                logger.error(f"Failed to load schema centroids: {e}")
        
        self.conceptual_anomalies = []

    def _schema_induction_phase(self):
        """Analyzes new dreams, calculates their schema field, and detects novelty."""
        logger.info("[SOMNIUM] Entering deep sleep. Beginning dynamic schema processing...")

        # For this process, we only need the dreams generated in the *last* REM cycle
        # We can find these by looking at the last entries in the dream log.
        # A simpler approach for now is to just process the single latest dream.
        try:
            with open(self.dream_log_path, 'r') as f:
                # Get the very last dream generated
                last_dream_line = f.readlines()[-1]
                last_dream = json.loads(last_dream_line)
        except (FileNotFoundError, IndexError):
            logger.warning("[SOMNIUM] No new dreams to process.")
            return

        dream_narrative = last_dream['narrative']
        logger.debug(f"[SOMNIUM] Analyzing new dream: '{dream_narrative[:70]}...'")

        # 1. Calculate Schema Field for the new dream
        dream_vec = self.mind.similarity_model.encode(dream_narrative, convert_to_tensor=True)
        schema_field = {}
        max_similarity = 0.0

        if self.schemas:
            schema_vectors = torch.stack([s['vector'] for s in self.schemas]).to(dream_vec.device)
            similarities = util.cos_sim(dream_vec, schema_vectors)[0]
            
            for i, schema in enumerate(self.schemas):
                similarity = similarities[i].item()
                schema_field[schema['name']] = similarity
                if similarity > max_similarity:
                    max_similarity = similarity
        
        # Log the belief state for this dream
        belief_entry = {
            "timestamp": datetime.now().isoformat(),
            "dream_narrative": dream_narrative,
            "schema_field": schema_field
        }
        with open(self.belief_trajectory_path, 'a') as f:
            f.write(json.dumps(belief_entry) + '\n')
        logger.info(f"[SOMNIUM] Dream mapped to schema field. Max similarity: {max_similarity:.2f}")

        # 2. Novelty Detection
        novelty_threshold = 0.6 # If a dream isn't at least 60% similar to ANY known schema, it's novel
        if max_similarity < novelty_threshold:
            logger.info(f"[SOMNIUM] Conceptual Anomaly detected! This dream doesn't fit known schemas.")
            self.conceptual_anomalies.append({"narrative": dream_narrative, "vector": dream_vec.cpu()})

        # 3. Autonomous Schema Creation
        # If we've collected enough anomalies, try to form a new belief
        if len(self.conceptual_anomalies) >= 5:
            logger.info("[SOMNIUM] Threshold for conceptual anomalies reached. Attempting to form a new schema...")
            
            anomaly_narratives = [a['narrative'] for a in self.conceptual_anomalies]
            
            try:
                # Use the Overseer one last time to name the new concept
                model = self.mind.api_client.GenerativeModel('gemini-1.5-pro-latest')
                meta_prompt = (
                    "The following are novel dream concepts that an AI could not categorize. "
                    "What is the single, underlying theme that connects them? "
                    "Give it a short, descriptive name (3-5 words).\n\n"
                    "DREAMS:\n- " + "\n- ".join(anomaly_narratives)
                )
                response = model.generate_content(meta_prompt)
                new_schema_name = response.text.strip().replace("\"", "")
                
                # Create the new schema centroid
                anomaly_vectors = torch.stack([a['vector'] for a in self.conceptual_anomalies])
                new_centroid = torch.mean(anomaly_vectors, dim=0)
                
                self.schemas.append({
                    "name": new_schema_name,
                    "vector": new_centroid,
                    "confidence": 1.0 # It's new and confident
                })
                
                # Save the updated schema list and clear the anomalies
                torch.save(self.schemas, self.schema_path)
                self.conceptual_anomalies = []
                logger.info(f"[SOMNIUM] SUCCESS! New schema '{new_schema_name}' created and integrated into worldview.")

            except Exception as e:
                logger.error(f"[SOMNIUM] New schema creation failed: {e}")

    def _dream_phase(self):
        # ... (This function remains unchanged) ...
        pass

    def sleep_cycle(self):
        # ... (This function now calls the new schema induction phase) ...
        # ... (Pruning and Reinforcement logic remains unchanged) ...
        logger.info("--- [SOMNIUM] INITIATING SLEEP CYCLE ---")
        # ...
        self._dream_phase()
        self._schema_induction_phase()
        # ...
        logger.info(f"--- [SOMNIUM] SLEEP CYCLE COMPLETE ---")