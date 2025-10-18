"""
Visualization utilities for X-REM
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import textwrap


class Visualizer:
    """Create visualizations for retrieval and generation results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 9
    
    def tensor_to_image(self, tensor):
        """Convert tensor to displayable image"""
        if isinstance(tensor, torch.Tensor):
            # Handle batch dimension
            if tensor.dim() == 4:
                tensor = tensor[0]
            
            # Denormalize if needed
            if tensor.min() < 0 or tensor.max() > 1:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = tensor * std + mean
            
            # Clamp to valid range
            tensor = torch.clamp(tensor, 0, 1)
            
            # Move to CPU and convert to numpy
            img_array = tensor.cpu().numpy()
            
            # Convert from (C, H, W) to (H, W, C) for matplotlib
            if img_array.shape[0] == 3 or img_array.shape[0] == 1:  # Channel first
                img_array = np.transpose(img_array, (1, 2, 0))
            
            # If single channel, squeeze it
            if img_array.shape[2] == 1:
                img_array = img_array.squeeze(2)
            
            return img_array
        elif isinstance(tensor, Image.Image):
            # Convert PIL to numpy
            return np.array(tensor)
        else:
            # Already numpy array
            return tensor
    
    def wrap_text(self, text: str, width: int = 60) -> str:
        """Wrap text to specified width"""
        return '\n'.join(textwrap.wrap(text, width=width))
    
    def plot_retrieval_examples(self, examples: list, filename: str):
        """
        Plot retrieval examples with images and reports
        
        Args:
            examples: List of dicts with keys:
                - image: Image tensor or PIL Image
                - ground_truth: Ground truth report
                - baseline: Baseline retrieved report
                - novel: Novel method retrieved report (optional)
        """
        n_examples = len(examples)
        fig, axes = plt.subplots(n_examples, 2, figsize=(14, 5 * n_examples))
        
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i, example in enumerate(examples):
            # Plot image
            ax_img = axes[i, 0]
            img = self.tensor_to_image(example['image'])
            
            # Determine if grayscale or RGB
            if img.ndim == 2:  # Grayscale
                ax_img.imshow(img, cmap='gray')
            else:  # RGB
                ax_img.imshow(img)
            
            ax_img.set_title(f'Example {i+1}', fontweight='bold', fontsize=12)
            ax_img.axis('off')
            
            # Plot reports
            ax_text = axes[i, 1]
            ax_text.axis('off')
            
            # Format reports
            gt_text = self.wrap_text(example['ground_truth'], width=70)
            baseline_text = self.wrap_text(example['baseline'], width=70)
            
            report_text = (
                f"Ground Truth:\n{gt_text}\n\n"
                f"Baseline Retrieved:\n{baseline_text}"
            )
            
            if 'novel' in example and example['novel'] != example['baseline']:
                novel_text = self.wrap_text(example['novel'], width=70)
                report_text += f"\n\nNovel Retrieved:\n{novel_text}"
            
            ax_text.text(
                0.05, 0.95, report_text,
                transform=ax_text.transAxes,
                fontsize=9,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            )
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved visualization to {save_path}")
    
    def plot_metrics_comparison(self, baseline_metrics: dict, novel_metrics: dict, filename: str):
        """Plot comparison of metrics between baseline and novel method"""
        metrics_to_plot = ['bleu_1', 'bleu_2', 'rouge1', 'rougeL', 'bertscore_f1']
        
        # Filter available metrics
        available_metrics = [m for m in metrics_to_plot if m in baseline_metrics and m in novel_metrics]
        
        if not available_metrics:
            print("⚠️  No metrics to plot")
            return
        
        x = np.arange(len(available_metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        baseline_scores = [baseline_metrics[m] for m in available_metrics]
        novel_scores = [novel_metrics[m] for m in available_metrics]
        
        bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', alpha=0.8)
        bars2 = ax.bar(x + width/2, novel_scores, width, label='Novel', alpha=0.8)
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Baseline vs Novel Method Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in available_metrics], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved metrics comparison to {save_path}")
    
    def plot_metric_distribution(self, scores: list, metric_name: str, filename: str):
        """Plot distribution of a metric across samples"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(scores, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(scores), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
        ax.axvline(np.median(scores), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(scores):.3f}')
        
        ax.set_xlabel(f'{metric_name} Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{metric_name} Score Distribution', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved distribution plot to {save_path}")
    
    def plot_retrieval_heatmap(self, similarities: np.ndarray, filename: str):
        """Plot similarity heatmap for retrieved reports"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(similarities, cmap='viridis', aspect='auto')
        
        ax.set_xlabel('Corpus Reports', fontweight='bold')
        ax.set_ylabel('Query Images', fontweight='bold')
        ax.set_title('Image-Report Similarity Heatmap', fontweight='bold', fontsize=14)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved heatmap to {save_path}")


def create_sample_visualization():
    """Create a sample visualization for testing"""
    from PIL import Image
    
    viz = Visualizer(Path('output/figures'))
    
    # Create dummy example
    dummy_img = Image.new('RGB', (224, 224), color='gray')
    example = {
        'image': dummy_img,
        'ground_truth': 'Normal chest radiograph. No acute cardiopulmonary process.',
        'baseline': 'The heart size is normal. The lungs are clear.',
        'novel': 'Normal cardiac silhouette. Clear lung fields bilaterally.'
    }
    
    viz.plot_retrieval_examples([example], 'test_example.png')
    print("✅ Sample visualization created!")


if __name__ == "__main__":
    create_sample_visualization()