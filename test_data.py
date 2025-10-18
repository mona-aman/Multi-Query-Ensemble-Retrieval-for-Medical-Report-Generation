import os
from pathlib import Path
from PIL import Image

# Check images
print("=" * 60)
print("CHECKING IMAGES")
print("=" * 60)

valid_images = 0
html_images = 0

image_files = list(Path("physionet.org/files/mimic-cxr-jpg").rglob("*.jpg"))
print(f"Found {len(image_files)} .jpg files\n")

for img_file in image_files[:100]:  # Check first 100
    try:
        # Check if HTML
        with open(img_file, 'rb') as f:
            header = f.read(200)
            if b'<!DOCTYPE' in header or b'<html' in header:
                html_images += 1
                continue
        
        # Try to open as image
        with Image.open(img_file) as img:
            img.verify()
        valid_images += 1
    except:
        pass

print(f"Results (first 100 checked):")
print(f"  ✅ Valid images: {valid_images}/100")
print(f"  ❌ HTML errors: {html_images}/100")

# Check reports
print("\n" + "=" * 60)
print("CHECKING REPORTS")
print("=" * 60)

valid_reports = 0
html_reports = 0

report_files = list(Path("physionet.org/files/mimic-cxr").rglob("*.txt"))
print(f"Found {len(report_files)} .txt files\n")

for rpt_file in report_files[:100]:
    try:
        with open(rpt_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(200)
            if '<!DOCTYPE' in content or '<html' in content:
                html_reports += 1
            else:
                valid_reports += 1
    except:
        pass

print(f"Results (first 100 checked):")
print(f"  ✅ Valid reports: {valid_reports}/100")
print(f"  ❌ HTML errors: {html_reports}/100")

print("\n" + "=" * 60)
if valid_images > 90 and valid_reports > 90:
    print("✅✅✅ EXCELLENT! Files are valid!")
    print("You can proceed to training!")
elif valid_images > 50:
    print("⚠️  MIXED - Some valid, some corrupt")
else:
    print("❌ MOSTLY CORRUPT")
print("=" * 60)