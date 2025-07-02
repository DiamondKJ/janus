# /JANUS-CORE/janus/memory.py
import os, torch, logging, json
from datetime import datetime, timezone
from sentence_transformers import util

logger = logging.getLogger(__name__)

class Hippocampus:
    def __init__(self, mind):
        self.mind = mind
        self.db_path = os.path.join(mind.project_root, "assets/memory/hippocampus_db.pt")
        self.encoder = mind.similarity_model
        self.decay_rate = 0.008 # Not currently used, but good for future implementation
        self._load_from_disk()

    def _load_from_disk(self):
        try:
            if os.path.exists(self.db_path):
                self.memory_bank = torch.load(self.db_path, weights_only=False) # Allow loading dicts
                # Ensure all memories have the new fields for backward compatibility
                for mem in self.memory_bank:
                    mem.setdefault('valence', {})
                    mem.setdefault('access_count', 0)
                    mem.setdefault('reinforcement_score', 1.0)
                logger.info(f"Loaded {len(self.memory_bank)} memories from persistent storage.")
            else: self.memory_bank = []
        except Exception as e:
            logger.warning(f"Could not load memory, starting fresh. Error: {e}")
            self.memory_bank = []

    def _save_to_disk(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        torch.save(self.memory_bank, self.db_path)

    def store_memory(self, prompt, response, valence):
        salience_score = valence.get('urgency', 0) + valence.get('curiosity', 0) + valence.get('creative_focus', 0)
        if salience_score < 0.9: 
            logger.debug("Event not salient enough for long-term storage.")
            return
        
        logger.info(f"Salient event detected. Storing to long-term memory...")
        memory_text = f"User: '{prompt}' | Janus: '{response}'"
        embedding = self.encoder.encode(memory_text, convert_to_tensor=True)
        
        # <<< UPGRADED MEMORY STRUCTURE
        self.memory_bank.append({
            "text": memory_text, 
            "embedding": embedding.cpu(), 
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "valence": valence,
            "access_count": 0,
            "reinforcement_score": 1.0
        })
        self._save_to_disk()
        logger.info(f"Hippocampus state with {len(self.memory_bank)} memories persisted.")
    
    def check_memory_familiarity(self, prompt, n_results=3):
        if not self.memory_bank:
            return {"familiarity_score": 0.0, "fragments": []}

        query_embedding = self.encoder.encode(prompt, convert_to_tensor=True)
        memory_embeddings = torch.stack([mem["embedding"] for mem in self.memory_bank]).to(query_embedding.device)
        similarities = util.cos_sim(query_embedding, memory_embeddings)[0]
        
        best_similarity, best_index = torch.max(similarities, 0)
        best_similarity = best_similarity.item()
        best_index = best_index.item()
        
        # <<< INCREMENT ACCESS COUNT ON RETRIEVAL
        if best_similarity > 0.6: # Only count as "accessed" if reasonably similar
            self.memory_bank[best_index]['access_count'] = self.memory_bank[best_index].get('access_count', 0) + 1
            logger.debug(f"Accessed memory '{self.memory_bank[best_index]['text'][:30]}...' (Accesses: {self.memory_bank[best_index]['access_count']})")

        top_indices = torch.topk(similarities, k=min(n_results, len(similarities))).indices
        retrieved_fragments = [self.memory_bank[i.item()]["text"] for i in top_indices]
        
        return {"familiarity_score": best_similarity, "fragments": retrieved_fragments}

    def synthesize_gist_from_fragments(self, prompt, fragments):
        # This function remains the same
        logger.info("High familiarity detected. Synthesizing memory gist via Overseer...")
        try:
            model = self.mind.api_client.GenerativeModel('gemini-1.5-pro-latest')
            meta_prompt = f"""You are a memory consolidation module... (prompt unchanged)"""
            response = model.generate_content(meta_prompt)
            gist = response.text.strip()
            logger.info(f"Memory Gist Synthesized: '{gist}'")
            return gist
        except Exception as e:
            logger.error(f"Memory Gist Synthesis failed: {e}")
            return "We've discussed this topic before."

    def shutdown(self):
        self._save_to_disk()
        logger.info("Hippocampus shutdown complete.")