"""
Test if MIMIC-CXR data is loading correctly
"""
from data.dataloader import MIMICCXRDataset
from pathlib import Path

def test_data_loading():
    print("="*60)
    print(" TESTING MIMIC-CXR DATA LOADING")
    print("="*60)
    
    data_dir = "./data/mimic_cxr"
    
    # Check if directory exists
    if not Path(data_dir).exists():
        print(f"\n❌ Directory not found: {data_dir}")
        print("Please create it and add MIMIC-CXR data")
        return
    
    # Check for required files
    required_files = [
        "mimic-cxr-2.0.0-metadata.csv.gz",
        "mimic-cxr-2.0.0-split.csv.gz",
        "mimic-cxr-2.0.0-chexpert.csv.gz"
    ]
    
    print("\n📋 Checking for required files...")
    for filename in required_files:
        filepath = Path(data_dir) / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} - NOT FOUND")
    
    # Check for preprocessed metadata
    print("\n📋 Checking for preprocessed metadata...")
    for split in ['train', 'val', 'test']:
        metadata_file = Path(data_dir) / f"{split}_metadata.json"
        if metadata_file.exists():
            print(f"  ✅ {split}_metadata.json")
        else:
            print(f"  ⚠️  {split}_metadata.json - Run preprocessing first")
    
    # Try loading dataset
    print("\n🧪 Testing dataset loading...")
    try:
        dataset = MIMICCXRDataset(
            data_dir=data_dir,
            split="train",
            max_samples=5
        )
        
        print(f"✅ Successfully loaded {len(dataset)} samples")
        
        # Test first sample
        print("\n🔍 Testing first sample...")
        sample = dataset[0]
        print(f"  Study ID: {sample['study_id']}")
        print(f"  DICOM ID: {sample['dicom_id']}")
        print(f"  Image shape: {sample['image'].size}")
        print(f"  Report: {sample['report'][:100]}...")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\nPlease run: python data/preprocessing.py")

if __name__ == "__main__":
    test_data_loading()