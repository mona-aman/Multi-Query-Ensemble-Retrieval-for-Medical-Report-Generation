"""
Helper utilities
"""
import torch
import numpy as np
import random
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import logging


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🎲 Random seed set to {seed}")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: Dict, output_path: str):
    """Save results to JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to {output_path}")


def setup_logging(log_dir: str = "./logs", experiment_name: str = "experiment"):
    """Setup logging"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("💻 Using CPU")
    return device


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"⏱️  Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"✅ Completed: {self.name} in {format_time(elapsed)}")


import time

if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    config = load_config()
    print(f"Config loaded: {config['project_name']}")
    
    device = get_device()
    print(f"Device: {device}")