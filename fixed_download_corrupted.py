"""
Full scan of ALL files (not just first 50)
"""
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

DATA_ROOT = "/ocean/projects/cis250085p/shared/X-REM/data/mimic_cxr"

def check_file_validity(filepath):
    """Check if file is valid"""
    if not filepath.exists():
        return 'missing'
    
    try:
        size = filepath.stat().st_size
        if size < 1000:
            return 'too_small'
        
        # Check if HTML error
        with open(filepath, 'rb') as f:
            header = f.read(100)
            if b'<!DOCTYPE html>' in header or b'<html' in header:
                return 'html_error'
            if b'403' in header or b'forbidden' in header.lower():
                return 'html_error'
        
        # For images, try to open
        if filepath.suffix == '.jpg':
            try:
                with Image.open(filepath) as img:
                    img.verify()
                return 'valid'
            except:
                return 'corrupt_image'
        
        return 'valid'
        
    except Exception as e:
        return 'error'

def full_scan():
    """Scan ALL files, not just first 50"""
    print("🔍 FULL DATASET SCAN (checking ALL files)")
    print("="*60)
    
    data_path = Path(DATA_ROOT)
    
    results = {
        'train': {'total': 0, 'valid_images': 0, 'corrupted_images': 0, 'missing_images': 0},
        'val': {'total': 0, 'valid_images': 0, 'corrupted_images': 0, 'missing_images': 0},
        'test': {'total': 0, 'valid_images': 0, 'corrupted_images': 0, 'missing_images': 0}
    }
    
    valid_samples = {'train': [], 'val': [], 'test': []}
    
    for split in ['train', 'val', 'test']:
        metadata_file = data_path / f'{split}_metadata.json'
        
        if not metadata_file.exists():
            print(f"❌ {split}_metadata.json not found!")
            continue
        
        with open(metadata_file, 'r') as f:
            samples = json.load(f)
        
        results[split]['total'] = len(samples)
        
        print(f"\n📋 Scanning {split} split - ALL {len(samples)} samples...")
        
        for sample in tqdm(samples, desc=f"  {split}"):
            img_path = data_path / sample['image_path']
            status = check_file_validity(img_path)
            
            if status == 'valid':
                results[split]['valid_images'] += 1
                valid_samples[split].append(sample)
            elif status == 'missing':
                results[split]['missing_images'] += 1
            else:
                results[split]['corrupted_images'] += 1
    
    # Print results
    print("\n" + "="*60)
    print(" FULL SCAN RESULTS")
    print("="*60)
    
    total_valid = 0
    total_samples = 0
    
    for split in ['train', 'val', 'test']:
        data = results[split]
        total = data['total']
        valid = data['valid_images']
        corrupted = data['corrupted_images']
        missing = data['missing_images']
        
        total_valid += valid
        total_samples += total
        
        print(f"\n{split.upper()}:")
        print(f"  Total:     {total:,}")
        print(f"  ✅ Valid:    {valid:,} ({valid/total*100:.1f}%)")
        print(f"  ❌ Corrupt:  {corrupted:,} ({corrupted/total*100:.1f}%)")
        print(f"  ⚠️  Missing:  {missing:,} ({missing/total*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"TOTAL VALID: {total_valid:,} / {total_samples:,} ({total_valid/total_samples*100:.1f}%)")
    print(f"{'='*60}")
    
    if total_valid > 0:
        print(f"\n✅ GOOD NEWS: {total_valid:,} valid files found!")
        print(f"   You can work with these for now")
        
        # Save valid samples
        for split in ['train', 'val', 'test']:
            if len(valid_samples[split]) > 0:
                output_file = data_path / f'{split}_metadata_valid_only.json'
                with open(output_file, 'w') as f:
                    json.dump(valid_samples[split], f, indent=2)
                print(f"   Saved {len(valid_samples[split])} valid {split} samples to:")
                print(f"     {output_file}")
    else:
        print(f"\n❌ NO VALID FILES FOUND")
        print(f"   Must re-download everything")
    
    return results, valid_samples

if __name__ == "__main__":
    results, valid = full_scan()