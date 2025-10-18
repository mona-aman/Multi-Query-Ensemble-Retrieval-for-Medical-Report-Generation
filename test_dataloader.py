"""
Test script to verify dataloader functionality
"""
import sys
import os
sys.path.append('/ocean/projects/cis250085p/shared/X-REM')

from data.dataloader import create_dataloader, MIMICCXRDataset
import torch
from torchvision import transforms

def test_dataloader():
    """Test the dataloader with our preprocessed data"""
    
    print("🧪 Testing MIMIC-CXR Dataloader...")
    
    # Test parameters
    data_dir = "/ocean/projects/cis250085p/shared/X-REM/data"
    batch_size = 4
    max_samples = 10  # Test with small sample first
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print(f"\n📂 Data directory: {data_dir}")
    print(f"🔢 Batch size: {batch_size}")
    print(f"📊 Max samples for test: {max_samples}")
    
    # Test each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*50}")
        print(f"Testing {split.upper()} split")
        print('='*50)
        
        try:
            # Create dataloader
            dataloader = create_dataloader(
                data_dir=data_dir,
                split=split,
                batch_size=batch_size,
                transform=transform,
                max_samples=max_samples,
                num_workers=0,  # Use 0 for debugging
                shuffle=False,
                skip_corrupted=True
            )
            
            print(f"✅ Dataloader created successfully")
            print(f"📊 Dataset size: {len(dataloader.dataset)}")
            print(f"🔢 Number of batches: {len(dataloader)}")
            
            # Test loading a batch
            print(f"\n🔄 Loading first batch...")
            
            for i, batch in enumerate(dataloader):
                print(f"Batch {i+1}:")
                print(f"  Image shape: {batch['image'].shape}")
                print(f"  Image dtype: {batch['image'].dtype}")
                print(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
                print(f"  Batch size: {len(batch['report'])}")
                
                # Print sample report info
                for j in range(min(2, len(batch['report']))):
                    report = batch['report'][j]
                    impression = batch['impression'][j]
                    study_id = batch['study_id'][j]
                    
                    print(f"\n  Sample {j+1}:")
                    print(f"    Study ID: {study_id}")
                    print(f"    Report length: {len(report)} chars")
                    print(f"    Impression length: {len(impression)} chars")
                    print(f"    Report preview: {report[:100]}...")
                    print(f"    Impression preview: {impression[:100]}...")
                
                # Only test first batch
                break
                
        except Exception as e:
            print(f"❌ Error testing {split} split: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("🧪 Dataloader test completed!")
    print('='*50)

if __name__ == "__main__":
    test_dataloader()