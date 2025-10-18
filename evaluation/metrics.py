"""
Evaluation metrics for report generation
"""
import evaluate
from typing import List, Dict
import numpy as np


class ReportMetrics:
    """Wrapper for all evaluation metrics"""
    
    def __init__(self):
        print("📥 Loading evaluation metrics...")
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        self.meteor = evaluate.load("meteor")
        print("✅ Metrics loaded")
    
    def compute_all(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute all metrics
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            Dictionary of metric scores
        """
        print("\n📊 Computing metrics...")
        
        # Ensure lists
        if isinstance(predictions[0], list):
            predictions = [' '.join(p) for p in predictions]
        if isinstance(references[0], list):
            references = [' '.join(r) for r in references]
        
        results = {}
        
        # BLEU
        print("  - Computing BLEU...")
        bleu_results = self.bleu.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )
        results['bleu_1'] = bleu_results.get('precisions', [0])[0] if bleu_results.get('precisions') else 0
        results['bleu_2'] = bleu_results.get('precisions', [0, 0])[1] if len(bleu_results.get('precisions', [])) > 1 else 0
        results['bleu_4'] = bleu_results.get('bleu', 0)
        
        # ROUGE
        print("  - Computing ROUGE...")
        rouge_results = self.rouge.compute(
            predictions=predictions,
            references=references
        )
        results['rouge1'] = rouge_results['rouge1']
        results['rouge2'] = rouge_results['rouge2']
        results['rougeL'] = rouge_results['rougeL']
        
        # BERTScore
        print("  - Computing BERTScore...")
        bertscore_results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang='en',
            model_type='distilbert-base-uncased'
        )
        results['bertscore_precision'] = np.mean(bertscore_results['precision'])
        results['bertscore_recall'] = np.mean(bertscore_results['recall'])
        results['bertscore_f1'] = np.mean(bertscore_results['f1'])
        
        # METEOR
        print("  - Computing METEOR...")
        meteor_results = self.meteor.compute(
            predictions=predictions,
            references=references
        )
        results['meteor'] = meteor_results['meteor']
        
        print("✅ Metrics computed\n")
        return results
    
    def print_results(self, results: Dict[str, float]):
        """Pretty print results"""
        print("\n" + "="*50)
        print(" EVALUATION RESULTS")
        print("="*50)
        
        print("\n📝 BLEU Scores:")
        print(f"  BLEU-1: {results['bleu_1']:.4f}")
        print(f"  BLEU-2: {results['bleu_2']:.4f}")
        print(f"  BLEU-4: {results['bleu_4']:.4f}")
        
        print("\n📝 ROUGE Scores:")
        print(f"  ROUGE-1: {results['rouge1']:.4f}")
        print(f"  ROUGE-2: {results['rouge2']:.4f}")
        print(f"  ROUGE-L: {results['rougeL']:.4f}")
        
        print("\n📝 BERTScore:")
        print(f"  Precision: {results['bertscore_precision']:.4f}")
        print(f"  Recall: {results['bertscore_recall']:.4f}")
        print(f"  F1: {results['bertscore_f1']:.4f}")
        
        print("\n📝 METEOR:")
        print(f"  Score: {results['meteor']:.4f}")
        
        print("\n" + "="*50 + "\n")
    
    def compare_methods(
        self,
        baseline_results: Dict[str, float],
        novel_results: Dict[str, float]
    ):
        """Compare two methods"""
        print("\n" + "="*50)
        print(" METHOD COMPARISON")
        print("="*50)
        
        for metric in baseline_results.keys():
            baseline = baseline_results[metric]
            novel = novel_results[metric]
            improvement = ((novel - baseline) / baseline) * 100
            
            symbol = "📈" if improvement > 0 else "📉"
            print(f"\n{metric}:")
            print(f"  Baseline: {baseline:.4f}")
            print(f"  Novel:    {novel:.4f}")
            print(f"  Change:   {symbol} {improvement:+.2f}%")
        
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    # Test metrics
    metrics = ReportMetrics()
    
    preds = ["Normal chest X-ray.", "Mild cardiomegaly observed."]
    refs = ["Chest X-ray is normal.", "Cardiomegaly is present."]
    
    results = metrics.compute_all(preds, refs)
    metrics.print_results(results)