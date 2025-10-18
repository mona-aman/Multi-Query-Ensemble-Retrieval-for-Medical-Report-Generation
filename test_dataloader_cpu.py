"""
Test dataloader with corrupted file handling
"""
from data.dataloader import MIMICCXRDataset, create_dataloader
from torchvision import transforms

DATA_ROOT = "/ocean/projects/cis250085p/shared/X-REM/data/mimic_cxr"

def test_robust_loading():
    print("="*60)
    print(" TESTING ROBUST DATALOADER")
    print("="*60)
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Test loading with corruption filtering
    print("\n1️⃣  Loading dataset (will filter corrupted)...")
    dataset = MIMICCXRDataset(
        data_dir=DATA_ROOT,
        split='train',
        transform=transform,
        max_samples=100,  # Test with 100
        skip_corrupted=True
    )
    
    print(f"\n✅ Dataset created with {len(dataset)} valid samples")
    
    if len(dataset) == 0:
        print("\n❌ No valid samples found!")
        print("   All files are corrupted - you MUST re-download")
        return False
    
    # Test getting a sample
    print("\n2️⃣  Loading first sample...")
    sample = dataset[0]
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Study ID: {sample['study_id']}")
    
    # Test dataloader
    print("\n3️⃣  Creating dataloader...")
    dataloader = create_dataloader(
        data_dir=DATA_ROOT,
        split='train',
        batch_size=4,
        max_samples=20,
        num_workers=0,
        skip_corrupted=True
    )
    
    print(f"   Batches: {len(dataloader)}")
    
    # Load a batch
    print("\n4️⃣  Loading first batch...")
    batch = next(iter(dataloader))
    print(f"   ✅ Batch loaded!")
    print(f"   Images: {batch['image'].shape}")
    print(f"   Reports: {len(batch['report'])}")
    
    print("\n" + "="*60)
    print("✅ DATALOADER WORKING!")
    print("="*60)
    
    if len(dataset) < 50:
        print("\n⚠️  WARNING: Very few valid samples!")
        print(f"   Only {len(dataset)} valid samples found")
        print("   You should re-download the data properly")
    
    return True

if __name__ == "__main__":
    try:
        success = test_robust_loading()
        if not success:
            print("\n❌ Test failed - need to fix data download")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()