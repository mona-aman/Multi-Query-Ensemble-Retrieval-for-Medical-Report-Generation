"""
Compare Baseline vs Novel Method
Generate final results and visualizations
"""
import json
from pathlib import Path
import yaml

from evaluation.metrics import ReportMetrics
from evaluation.visualize import Visualizer
from utils.helpers import save_results


def main():
    print("="*60)
    print(" COMPARING BASELINE VS NOVEL METHOD")
    print("="*60)
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    results_dir = Path(config['evaluation']['output_dir'])
    
    # Check if results exist
    baseline_dir = results_dir / 'baseline'
    novel_dir = results_dir / 'novel'
    
    if not baseline_dir.exists() or not novel_dir.exists():
        print("\n❌ Error: Run baseline and novel experiments first!")
        print(f"  Missing: {baseline_dir if not baseline_dir.exists() else novel_dir}")
        return
    
    print("\n📁 Loading results...")
    
    # Load baseline results
    with open(baseline_dir / 'metrics.json', 'r') as f:
        baseline_results = json.load(f)
    
    with open(baseline_dir / 'predictions.json', 'r') as f:
        baseline_data = json.load(f)
    
    # Load novel results
    with open(novel_dir / 'metrics.json', 'r') as f:
        novel_results = json.load(f)
    
    with open(novel_dir / 'predictions.json', 'r') as f:
        novel_data = json.load(f)
    
    print("✅ Results loaded")
    
    print("\n" + "="*60)
    print(" COMPARISON RESULTS")
    print("="*60)
    
    # Compare using metrics class
    metrics = ReportMetrics()
    metrics.compare_methods(baseline_results, novel_results)
    
    # Create comparison directory
    comparison_dir = results_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(" GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create visualizer
    viz = Visualizer(comparison_dir / 'figures')
    
    # 1. Metric comparison bar chart
    print("\n📊 Creating metric comparison...")
    viz.plot_metric_comparison(
        baseline_results,
        novel_results,
        'metric_comparison.png'
    )
    
    # 2. Improvement heatmap
    print("📊 Creating improvement heatmap...")
    viz.plot_improvement_heatmap(
        baseline_results,
        novel_results,
        'improvement_heatmap.png'
    )
    
    # 3. Create LaTeX table
    print("📊 Creating results table...")
    viz.create_results_table(
        baseline_results,
        novel_results,
        'results_table.txt'
    )
    
    # 4. Side-by-side examples
    print("📊 Creating example comparisons...")
    from data.dataloader import MIMICCXRDataset
    
    test_dataset = MIMICCXRDataset(
        data_dir=config['data']['data_dir'],
        split='test',
        max_samples=3
    )
    
    examples = []
    for i in range(min(3, len(test_dataset))):
        sample = test_dataset[i]
        examples.append({
            'image': sample['image'],
            'ground_truth': baseline_data['references'][i],
            'baseline': baseline_data['predictions'][i],
            'novel': novel_data['predictions'][i]
        })
    
    viz.plot_retrieval_examples(examples, 'comparison_examples.png')
    
    # Save combined results
    comparison_results = {
        'baseline': baseline_results,
        'novel': novel_results,
        'improvements': {}
    }
    
    for metric in baseline_results.keys():
        baseline = baseline_results[metric]
        novel = novel_results[metric]
        improvement = ((novel - baseline) / baseline) * 100
        comparison_results['improvements'][metric] = improvement
    
    save_results(comparison_results, comparison_dir / 'comparison_metrics.json')
    
    print("\n" + "="*60)
    print(" FINAL SUMMARY")
    print("="*60)
    
    print("\n📈 Key Improvements:")
    for metric, improvement in comparison_results['improvements'].items():
        symbol = "✅" if improvement > 0 else "⚠️"
        print(f"  {symbol} {metric}: {improvement:+.2f}%")
    
    print(f"\n📁 All results saved to: {comparison_dir}")
    print("\n🎉 Comparison complete!")
    print("\n📄 Next steps:")
    print("  1. Review visualizations in:", comparison_dir / 'figures')
    print("  2. Check LaTeX table in:", comparison_dir / 'figures' / 'results_table.txt')
    print("  3. Use these results for your paper/poster!")


if __name__ == "__main__":
    main()