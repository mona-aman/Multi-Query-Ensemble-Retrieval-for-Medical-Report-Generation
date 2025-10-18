"""
Baseline X-REM implementation
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


class BaselineXREM:
    """
    Baseline X-REM: Cosine Similarity + NLI Filter
    (Simplified - no ITM scoring for speed)
    """
    
    def __init__(
        self,
        image_encoder,
        text_encoder,
        nli_model,
        top_i: int = 50,
        top_k: int = 2
    ):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.nli_model = nli_model
        self.top_i = top_i
        self.top_k = top_k
        
        # Cache for corpus embeddings
        self.corpus_embeddings = None
        self.corpus_reports = None
    
    def index_corpus(self, reports: List[str]):
        """Pre-compute embeddings for retrieval corpus"""
        print("🔧 Indexing corpus...")
        self.corpus_reports = reports
        
        # Encode in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(reports), batch_size)):
            batch = reports[i:i+batch_size]
            batch_emb = self.text_encoder.encode(batch)
            embeddings.append(batch_emb)
        
        self.corpus_embeddings = torch.cat(embeddings, dim=0)
        print(f"✅ Indexed {len(reports)} reports")
    
    def retrieve(
        self,
        image: torch.Tensor,
        use_nli_filter: bool = True
    ) -> List[str]:
        """
        Retrieve reports for a given image
        
        Args:
            image: Input image
            use_nli_filter: Whether to use NLI filtering
            
        Returns:
            List of retrieved reports
        """
        if self.corpus_embeddings is None:
            raise ValueError("Corpus not indexed! Call index_corpus() first.")
        
        # Step 1: Encode query image
        img_emb = self.image_encoder.encode([image])
        
        # Step 2: Compute cosine similarities
        similarities = F.cosine_similarity(
            img_emb.unsqueeze(1),
            self.corpus_embeddings.unsqueeze(0),
            dim=-1
        ).squeeze()
        
        # Step 3: Get top-i candidates
        top_indices = torch.topk(similarities, k=min(self.top_i, len(similarities))).indices
        candidates = [self.corpus_reports[idx] for idx in top_indices.cpu().numpy()]
        
        # Step 4: NLI filtering (optional)
        if use_nli_filter:
            selected = self._nli_filter(candidates, self.top_k)
        else:
            selected = candidates[:self.top_k]
        
        return selected
    
    def _nli_filter(self, candidates: List[str], k: int) -> List[str]:
        """
        Filter candidates using NLI to avoid redundancy
        """
        selected = []
        
        for candidate in candidates:
            if len(selected) == 0:
                selected.append(candidate)
                continue
            
            # Check if candidate is redundant
            is_redundant = False
            for prev_report in selected:
                relation = self.nli_model.predict(prev_report, candidate)
                if relation in ['entailment', 'contradiction']:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append(candidate)
            
            if len(selected) >= k:
                break
        
        # If not enough selected, just take top-k
        if len(selected) < k:
            selected.extend(candidates[len(selected):k])
        
        return selected[:k]
    
    def batch_retrieve(
        self,
        images: List,
        use_nli_filter: bool = True
    ) -> List[List[str]]:
        """Retrieve for multiple images"""
        results = []
        for img in tqdm(images, desc="Retrieving"):
            retrieved = self.retrieve(img, use_nli_filter)
            results.append(retrieved)
        return results


if __name__ == "__main__":
    print("Testing Baseline X-REM...")
    
    # This will be tested in run_baseline.py
    pass
