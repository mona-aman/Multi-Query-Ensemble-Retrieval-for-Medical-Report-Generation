"""
Run Novel Method: Multi-Query Ensemble Retrieval
Day 2: Test novel approach on full dataset
"""
import torch
from pathlib import Path
import yaml
from tqdm import tqdm
import json
from datetime import datetime
import sys
import os
import argparse

# Add project root to path
sys.path.append('/ocean/projects/cis250085p/shared/X-REM')

from data.dataloader import create_dataloader, get_corpus_reports
from models.encoders import load_encoders
from models.novel_method import EnsembleXREM
from evaluation.metrics import ReportMetrics
from evaluation.visualize import Visualizer
from utils.helpers import set_seed, save_results, get_device, Timer


def main(args=None):
    print("="*70)
    print(" NOVEL METHOD: MULTI-QUERY ENSEMBLE RETRIEVAL - FULL DATASET")
    print("="*70)
    
    # Load config
    config_path = '/ocean/projects/cis250085p/shared/X-REM/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths
    config['data']['data_dir'] = '/ocean/projects/cis250085p/shared/X-REM/data'
    
    # Optionally override sizes via CLI
    if args and args.train_size is not None:
        config['data']['train_size'] = args.train_size
    if args and args.test_size is not None:
        config['data']['test_size'] = args.test_size
    
    # Set seed
    set_seed(config['random_seed'])
    
    # Setup device
    device = get_device()
    config['models']['device'] = device
    
    # Create output directory
    output_dir = Path(config['evaluation']['output_dir']) / 'novel'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(" STEP 1: DATA PREPARATION")
    print("="*70)
    
    print(f"\n📂 Data directory: {config['data']['data_dir']}")
    train_size_cfg = config['data'].get('train_size')
    test_size_cfg = config['data'].get('test_size')
    print(f"📊 Train corpus size: {train_size_cfg if train_size_cfg else 'all available'}")
    print(f"📊 Test samples: {test_size_cfg if test_size_cfg else 'all available'}")
    
    # Load corpus reports for retrieval (train split)
    print("\n📚 Loading corpus reports (train split)...")
    corpus_reports = get_corpus_reports(
        data_dir=config['data']['data_dir'],
        split='train',
        skip_corrupted=True
    )
    
    # Limit corpus size if specified in config
    train_size = config['data'].get('train_size')
    if train_size and train_size < len(corpus_reports):
        corpus_reports = corpus_reports[:train_size]
        print(f"✅ Using {len(corpus_reports)} reports for corpus (limited from config)")
    else:
        print(f"✅ Using all {len(corpus_reports)} available corpus reports")
    
    # Load test dataloader
    print("\n📥 Loading test dataloader...")
    test_dataloader = create_dataloader(
        data_dir=config['data']['data_dir'],
        split='test',
        batch_size=1,  # Process one at a time for retrieval
        max_samples=config['data'].get('test_size'),
        shuffle=False,
        skip_corrupted=True
    )
    
    print(f"✅ Test samples: {len(test_dataloader.dataset)}")
    
    print("\n" + "="*70)
    print(" STEP 2: LOAD MODELS")
    print("="*70)
    
    # Load encoders
    with Timer("Loading models"):
        image_encoder, text_encoder, nli_model = load_encoders(config)
    
    print("\n" + "="*70)
    print(" STEP 3: INITIALIZE MULTI-QUERY ENSEMBLE X-REM")
    print("="*70)
    
    # Get ensemble config (with defaults if not in config)
    ensemble_config = config.get('ensemble', {})
    num_queries = ensemble_config.get('num_queries', 5)
    ensemble_method = ensemble_config.get('ensemble_method', 'vote')
    
    print(f"\n💡 Novel Method Configuration:")
    print(f"   Method: Multi-Query Ensemble Retrieval")
    print(f"   Num Query Variations: {num_queries}")
    print(f"   Ensemble Strategy: {ensemble_method}")
    print(f"   Top-i: {config['retrieval']['top_i']}")
    print(f"   Top-k: {config['retrieval']['top_k']}")
    
    # Create novel retrieval model
    retrieval_model = EnsembleXREM(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        nli_model=nli_model,
        num_queries=num_queries,
        ensemble_method=ensemble_method,
        top_i=config['retrieval']['top_i'],
        top_k=config['retrieval']['top_k']
    )
    
    # Index corpus
    print(f"\n📚 Building retrieval corpus with {len(corpus_reports)} reports...")
    with Timer("Indexing corpus"):
        retrieval_model.index_corpus(corpus_reports)
    
    print("\n" + "="*70)
    print(" STEP 4: RUN ENSEMBLE RETRIEVAL ON TEST SET")
    print("="*70)
    
    predictions = []
    references = []
    
    print(f"\n🔍 Running ensemble retrieval on {len(test_dataloader.dataset)} test samples...")
    print("💡 This uses multiple query variations for robust retrieval")
    print(f"⚙️  Settings: top_i={config['retrieval']['top_i']}, top_k={config['retrieval']['top_k']}")
    print(f"⚙️  Query variations per image: {num_queries}")
    print(f"⚙️  Ensemble method: {ensemble_method}")
    
    for batch in tqdm(test_dataloader, desc="Ensemble retrieval"):
        # Get sample from batch
        image = batch['image'][0]  # First (and only) item in batch
        report = batch['report'][0]
        
        # Retrieve with query expansion
        retrieved = retrieval_model.retrieve(image=image)
        
        # Join retrieved reports
        prediction = ' '.join(retrieved)
        predictions.append(prediction)
        references.append(report)
    
    # Save predictions
    print("\n💾 Saving predictions...")
    results_data = {
        'predictions': predictions,
        'references': references,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'vision_encoder': config['models']['vision_encoder'],
            'text_encoder': config['models']['text_encoder'],
            'nli_model': config['models']['nli_model'],
            'method': 'multi_query_ensemble',
            'num_queries': num_queries,
            'ensemble_method': ensemble_method,
            'top_i': config['retrieval']['top_i'],
            'top_k': config['retrieval']['top_k'],
            'corpus_size': len(corpus_reports),
            'test_size': len(predictions)
        }
    }
    
    save_results(results_data, output_dir / 'predictions.json')
    
    print("\n" + "="*70)
    print(" STEP 5: EVALUATE")
    print("="*70)
    
    # Compute metrics
    print("📥 Loading evaluation metrics...")
    metrics_calculator = ReportMetrics()
    print("✅ Metrics loaded\n")
    
    results = metrics_calculator.compute_all(predictions, references)
    print("✅ Metrics computed\n")
    
    metrics_calculator.print_results(results)
    
    # Save metrics
    save_results(results, output_dir / 'metrics.json')
    
    print("\n" + "="*70)
    print(" STEP 6: VISUALIZE")
    print("="*70)
    
    # Create visualizations
    viz = Visualizer(output_dir / 'figures')
    
    # Plot examples
    print("\n📊 Creating visualization examples...")
    examples = []
    num_examples = min(5, len(test_dataloader.dataset))  # Show 5 examples
    
    for i in range(num_examples):
        # Get sample again
        sample = test_dataloader.dataset[i]
        
        examples.append({
            'image': sample['image'],
            'ground_truth': references[i],
            'baseline': 'See baseline results',
            'novel': predictions[i]
        })
    
    viz.plot_retrieval_examples(examples, 'novel_examples.png')
    
    print("\n" + "="*70)
    print(" NOVEL METHOD COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"\n📊 Summary:")
    print(f"  Corpus size: {len(corpus_reports)}")
    print(f"  Test samples: {len(predictions)}")
    print(f"  BLEU-1: {results.get('bleu_1', 0):.4f}")
    print(f"  BLEU-2: {results.get('bleu_2', 0):.4f}")
    print(f"  BLEU-4: {results.get('bleu_4', 0):.4f}")
    print(f"  ROUGE-1: {results.get('rouge1', 0):.4f}")
    print(f"  ROUGE-L: {results.get('rougeL', 0):.4f}")
    print(f"  BERTScore F1: {results.get('bertscore_f1', 0):.4f}")
    print(f"  METEOR: {results.get('meteor_score', 0):.4f}")
    
    # Check for empty references
    empty_refs = sum(1 for ref in references if len(ref.strip()) == 0)
    if empty_refs > 0:
        print(f"\n⚠️  Note: {empty_refs} test samples had empty reference reports")
    
    print("\n✅ Run comparison script to compare with baseline!")
    
    # Load baseline results for comparison
    baseline_path = Path(config['evaluation']['output_dir']) / 'baseline' / 'metrics.json'
    if baseline_path.exists():
        print("\n" + "="*70)
        print(" COMPARISON WITH BASELINE")
        print("="*70)
        
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
        
        print("\n📊 Improvement over Baseline:")
        metrics_to_compare = ['bleu_1', 'bleu_2', 'rouge1', 'rougeL', 'bertscore_f1']
        
        for metric in metrics_to_compare:
            baseline_val = baseline_results.get(metric, 0)
            novel_val = results.get(metric, 0)
            
            if baseline_val > 0:
                improvement = ((novel_val - baseline_val) / baseline_val) * 100
                symbol = "↑" if improvement > 0 else "↓"
                print(f"  {metric.upper()}: {baseline_val:.4f} → {novel_val:.4f} ({symbol} {abs(improvement):.1f}%)")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run X-REM novel method")
    parser.add_argument("--train-size", type=int, default=None,
                        help="Limit number of corpus (train) reports; default: all")
    parser.add_argument("--test-size", type=int, default=None,
                        help="Limit number of test samples; default: all")
    cli_args = parser.parse_args()
    
    try:
        results = main(cli_args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()