"""
Novel X-REM Method: Query Expansion
Enhances retrieval by generating multiple query variants
"""
import torch
import torch.nn.functional as F
from typing import List
import numpy as np
from tqdm import tqdm
import os


class QueryExpansionXREM:
    """
    Novel X-REM with query expansion using LLM or fallback method
    """
    
    def __init__(
        self,
        image_encoder,
        text_encoder,
        nli_model,
        llm_provider: str = "fallback",
        llm_model: str = "gpt-3.5-turbo",
        num_expansions: int = 3,
        top_i: int = 50,
        top_k: int = 2
    ):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.nli_model = nli_model
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.num_expansions = num_expansions
        self.top_i = top_i
        self.top_k = top_k
        
        # Cache for corpus embeddings
        self.corpus_embeddings = None
        self.corpus_reports = None
        
        # Initialize LLM client if needed
        if llm_provider == "openai":
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                print("✅ OpenAI client initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize OpenAI: {e}")
                print("   Falling back to template-based expansion")
                self.llm_provider = "fallback"
    
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
    
    def expand_query_with_llm(self, initial_query: str) -> List[str]:
        """Generate query expansions using LLM"""
        if self.llm_provider != "openai":
            return self.expand_query_fallback(initial_query)
        
        try:
            prompt = f"""You are a medical expert. Given an initial medical report query, generate {self.num_expansions} alternative phrasings that maintain the same clinical meaning but use different medical terminology.

Initial query: "{initial_query}"

Generate {self.num_expansions} alternative medical phrasings, one per line:"""
            
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a medical terminology expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            expansions = response.choices[0].message.content.strip().split('\n')
            expansions = [e.strip('- ').strip() for e in expansions if e.strip()]
            
            return expansions[:self.num_expansions]
            
        except Exception as e:
            print(f"⚠️  LLM expansion failed: {e}")
            return self.expand_query_fallback(initial_query)
    
    def expand_query_fallback(self, initial_query: str) -> List[str]:
        """
        Template-based query expansion without LLM
        Uses medical terminology variations
        """
        # Common medical term substitutions
        substitutions = {
            'normal': ['unremarkable', 'within normal limits', 'no abnormality'],
            'abnormal': ['pathologic', 'atypical', 'unusual'],
            'chest': ['thorax', 'thoracic', 'chest wall'],
            'lungs': ['pulmonary', 'lung fields', 'pulmonary parenchyma'],
            'heart': ['cardiac', 'cardiovascular', 'cardiac silhouette'],
            'clear': ['lucent', 'transparent', 'no opacity'],
            'opacity': ['density', 'infiltrate', 'consolidation'],
            'enlarged': ['increased', 'expanded', 'prominent'],
            'small': ['diminished', 'reduced', 'decreased'],
        }
        
        expansions = []
        words = initial_query.lower().split()
        
        # Generate variations by substituting terms
        for key, alternatives in substitutions.items():
            if key in words:
                for alt in alternatives[:self.num_expansions]:
                    expanded = initial_query.lower().replace(key, alt)
                    if expanded != initial_query.lower():
                        expansions.append(expanded)
        
        # If no substitutions found, create semantic variations
        if not expansions:
            templates = [
                f"findings consistent with {initial_query}",
                f"radiographic appearance of {initial_query}",
                f"impression: {initial_query}",
            ]
            expansions = templates[:self.num_expansions]
        
        return expansions[:self.num_expansions]
    
    def retrieve(
        self,
        image: torch.Tensor,
        use_nli_filter: bool = False
    ) -> List[str]:
        """
        Retrieve reports using query expansion
        
        Args:
            image: Input image
            use_nli_filter: Whether to use NLI filtering (optional)
            
        Returns:
            List of retrieved reports
        """
        if self.corpus_embeddings is None:
            raise ValueError("Corpus not indexed! Call index_corpus() first.")
        
        # Step 1: Encode query image
        img_emb = self.image_encoder.encode([image])
        
        # Step 2: Compute initial similarities
        similarities = F.cosine_similarity(
            img_emb.unsqueeze(1),
            self.corpus_embeddings.unsqueeze(0),
            dim=-1
        ).squeeze()
        
        # Step 3: Get top candidate for expansion
        top_idx = torch.argmax(similarities).item()
        initial_report = self.corpus_reports[top_idx]
        
        # Step 4: Generate query expansions
        expanded_queries = self.expand_query_with_llm(initial_report)
        
        # Step 5: Encode expanded queries
        if expanded_queries:
            expanded_embeddings = self.text_encoder.encode(expanded_queries)
            
            # Combine image and text query embeddings
            # Average image embedding with expanded text embeddings
            # combined_emb = torch.cat([img_emb, expanded_embeddings], dim=0).mean(dim=0, keepdim=True)
            img_weight = 0.85
            text_weight = 0.15 / len(expanded_embeddings)  # Split text weight among expansions

            combined_emb = img_weight * img_emb
            for exp_emb in expanded_embeddings:
                combined_emb += text_weight * exp_emb
        else:
            combined_emb = img_emb
        
        # Step 6: Re-compute similarities with expanded query
        final_similarities = F.cosine_similarity(
            combined_emb.unsqueeze(1),
            self.corpus_embeddings.unsqueeze(0),
            dim=-1
        ).squeeze()
        
        # Step 7: Get top-i candidates
        top_indices = torch.topk(final_similarities, k=min(self.top_i, len(final_similarities))).indices
        candidates = [self.corpus_reports[idx] for idx in top_indices.cpu().numpy()]
        
        # Step 8: NLI filtering (optional, same as baseline)
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
        use_nli_filter: bool = False
    ) -> List[List[str]]:
        """Retrieve for multiple images"""
        results = []
        for img in tqdm(images, desc="Retrieving with expansion"):
            retrieved = self.retrieve(img, use_nli_filter)
            results.append(retrieved)
        return results


if __name__ == "__main__":
    print("Testing Query Expansion X-REM...")
    
    # This will be tested in run_novel.py
    pass