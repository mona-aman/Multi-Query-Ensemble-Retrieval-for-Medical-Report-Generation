"""
Run Baseline X-REM Experiment
Day 1: Get baseline numbers
"""
import torch
from pathlib import Path
import yaml
from tqdm import tqdm
import json
from datetime import datetime
import sys

# Add project root to path
sys.path.append('/ocean/projects/cis250085p/shared/X-REM')

from data.dataloader import create_dataloader, get_corpus_reports
from models.encoders import load_encoders
from models.baseline_xrem import BaselineXREM
from evaluation.metrics import ReportMetrics
from evaluation.visualize import Visualizer
from utils.helpers import set_seed, save_results, get_device, Timer


def main():
    print("="*70)
    print(" BASELINE X-REM EXPERIMENT")
    print("="*70)
    
    # Load config
    config_path = '/ocean/projects/cis250085p/shared/X-REM/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths
    config['data']['data_dir'] = '/ocean/projects/cis250085p/shared/X-REM/data'
    
    # Set seed
    set_seed(config['random_seed'])
    
    # Setup device
    device = get_device()
    config['models']['device'] = device
    
    # Create output directory
    output_dir = Path(config['evaluation']['output_dir']) / 'baseline'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(" STEP 1: DATA PREPARATION")
    print("="*70)
    
    print(f"\n📂 Data directory: {config['data']['data_dir']}")
    print(f"📊 Train samples: {config['data'].get('train_size', 'all')}")
    print(f"📊 Test samples: {config['data'].get('test_size', 'all')}")
    
    # Load corpus reports for retrieval
    print("\n📚 Loading corpus reports (train split)...")
    corpus_reports = get_corpus_reports(
        data_dir=config['data']['data_dir'],
        split='train',
        skip_corrupted=True
    )
    
    # Limit corpus size if specified
    train_size = config['data'].get('train_size', len(corpus_reports))
    if train_size and train_size < len(corpus_reports):
        corpus_reports = corpus_reports[:train_size]
    
    print(f"✅ Loaded {len(corpus_reports)} corpus reports")
    
    # Load test data
    print("\n📥 Loading test dataloader...")
    test_dataloader = create_dataloader(
        data_dir=config['data']['data_dir'],
        split='test',
        batch_size=1,  # Process one at a time for retrieval
        max_samples=config['data'].get('test_size', None),
        shuffle=False,
        skip_corrupted=True
    )
    
    print(f"✅ Test samples: {len(test_dataloader.dataset)}")
    
    print("\n" + "="*70)
    print(" STEP 2: LOAD MODELS")
    print("="*70)
    
    # Load encoders
    with Timer("Loading encoders"):
        image_encoder, text_encoder, nli_model = load_encoders(config)
    
    print("\n" + "="*70)
    print(" STEP 3: INITIALIZE BASELINE X-REM")
    print("="*70)
    
    # Create retrieval model
    retrieval_model = BaselineXREM(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        nli_model=nli_model,
        top_i=config['retrieval']['top_i'],
        top_k=config['retrieval']['top_k']
    )
    
    # Index corpus
    print("\n📚 Building retrieval corpus...")
    with Timer("Indexing corpus"):
        retrieval_model.index_corpus(corpus_reports)
    
    print("\n" + "="*70)
    print(" STEP 4: RUN RETRIEVAL ON TEST SET")
    print("="*70)
    
    # Retrieve for test set
    predictions = []
    references = []
    
    print(f"\n🔍 Running retrieval on {len(test_dataloader.dataset)} test samples...")
    
    for batch in tqdm(test_dataloader, desc="Retrieving"):
        # Get sample from batch
        image = batch['image'][0]  # First (and only) item in batch
        report = batch['report'][0]
        
        # Retrieve reports
        retrieved = retrieval_model.retrieve(
            image=image,
            use_nli_filter=config['retrieval'].get('use_nli_filter', True)
        )
        
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
    metrics_calculator = ReportMetrics()
    results = metrics_calculator.compute_all(predictions, references)
    metrics_calculator.print_results(results)
    
    # Save metrics
    save_results(results, output_dir / 'metrics.json')
    
    print("\n" + "="*70)
    print(" STEP 6: VISUALIZE")
    print("="*70)
    
    # Create visualizations
    viz = Visualizer(output_dir / 'figures')
    
    # Plot examples (first 3 samples)
    print("\n📊 Creating visualization examples...")
    examples = []
    num_examples = min(3, len(test_dataloader.dataset))
    
    for i in range(num_examples):
        # Get sample again
        sample = test_dataloader.dataset[i]
        
        examples.append({
            'image': sample['image'],
            'ground_truth': references[i],
            'baseline': predictions[i],
            'novel': predictions[i]  # Same as baseline for now
        })
    
    viz.plot_retrieval_examples(examples, 'baseline_examples.png')
    
    print("\n" + "="*70)
    print(" BASELINE EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {output_dir}")
    print("\n📊 Summary:")
    print(f"  Corpus size: {len(corpus_reports)}")
    print(f"  Test samples: {len(predictions)}")
    print(f"  BLEU-1: {results.get('bleu_1', 0):.4f}")
    print(f"  BLEU-2: {results.get('bleu_2', 0):.4f}")
    print(f"  ROUGE-L: {results.get('rougeL', 0):.4f}")
    print(f"  BERTScore F1: {results.get('bertscore_f1', 0):.4f}")
    print("\n✅ Ready for Day 2: Run novel method!")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()