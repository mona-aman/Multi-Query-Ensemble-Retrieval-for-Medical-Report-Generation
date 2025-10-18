"""
Check for corrupted downloads and identify what needs to be re-downloaded
"""
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

DATA_ROOT = "/ocean/projects/cis250085p/shared/X-REM/data/mimic_cxr"

def check_file_validity(filepath):
    """Check if file is valid (not HTML error page)"""
    if not filepath.exists():
        return 'missing'
    
    try:
        # Check file size
        size = filepath.stat().st_size
        if size < 1000:  # Suspiciously small
            return 'too_small'
        
        # Check if it's HTML (error page)
        with open(filepath, 'rb') as f:
            header = f.read(100)
            if b'<!DOCTYPE html>' in header or b'<html' in header:
                return 'html_error'
        
        # For images, try to open
        if filepath.suffix == '.jpg':
            try:
                with Image.open(filepath) as img:
                    img.verify()
                return 'valid'
            except:
                return 'corrupt_image'
        
        # For text files
        if filepath.suffix == '.txt':
            with open(filepath, 'r') as f:
                text = f.read(200)
                if '<html' in text.lower() or '403' in text or 'forbidden' in text.lower():
                    return 'html_error'
            return 'valid'
        
        return 'valid'
        
    except Exception as e:
        return f'error_{str(e)[:20]}'

def scan_dataset():
    """Scan all files and report status"""
    print("🔍 Scanning dataset for corrupted files...")
    
    data_path = Path(DATA_ROOT)
    
    # Load metadata
    stats = {
        'train': {'total': 0, 'issues': {}},
        'val': {'total': 0, 'issues': {}},
        'test': {'total': 0, 'issues': {}}
    }
    
    for split in ['train', 'val', 'test']:
        metadata_file = data_path / f'{split}_metadata.json'
        
        if not metadata_file.exists():
            print(f"❌ {split}_metadata.json not found!")
            continue
        
        with open(metadata_file, 'r') as f:
            samples = json.load(f)
        
        stats[split]['total'] = len(samples)
        
        print(f"\n📋 Checking {split} split ({len(samples)} samples)...")
        
        for sample in tqdm(samples[:50], desc=f"  {split}"):  # Check first 50
            # Check image
            img_path = data_path / sample['image_path']
            img_status = check_file_validity(img_path)
            
            if img_status != 'valid':
                if img_status not in stats[split]['issues']:
                    stats[split]['issues'][img_status] = []
                stats[split]['issues'][img_status].append(sample['study_id'])
            
            # Check report
            report_path = data_path / sample['report_path']
            report_status = check_file_validity(report_path)
            
            if report_status != 'valid':
                key = f'report_{report_status}'
                if key not in stats[split]['issues']:
                    stats[split]['issues'][key] = []
                stats[split]['issues'][key].append(sample['study_id'])
    
    # Print summary
    print("\n" + "="*60)
    print(" CORRUPTION SCAN RESULTS")
    print("="*60)
    
    total_issues = 0
    for split, data in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Total samples: {data['total']}")
        
        if not data['issues']:
            print(f"  ✅ No issues found (checked first 50)")
        else:
            for issue_type, study_ids in data['issues'].items():
                print(f"  ❌ {issue_type}: {len(study_ids)} files")
                total_issues += len(study_ids)
                # Show first few
                print(f"     Sample IDs: {study_ids[:3]}...")
    
    print(f"\n{'='*60}")
    print(f"Total issues found: {total_issues}")
    
    if total_issues > 0:
        print("\n⚠️  CRITICAL: Download failed!")
        print("\n💡 Likely causes:")
        print("   1. PhysioNet credentials expired")
        print("   2. Access permissions changed")
        print("   3. Rate limiting kicked in")
        print("   4. Download script needs authentication fix")
        
        print("\n🔧 Solutions:")
        print("   1. Re-download with fixed authentication")
        print("   2. Use wget with correct credentials")
        print("   3. Download from PhysioNet web interface")
    
    return stats

if __name__ == "__main__":
    scan_dataset()