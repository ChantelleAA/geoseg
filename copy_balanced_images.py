"""Copy selected balanced images from combined dataset to training set."""

import json
import shutil
from pathlib import Path

def copy_balanced_images(selection_file, source_root, dest_root):
    """Copy selected balanced images to training set."""
    
    # Load selection
    with open(selection_file, 'r') as f:
        data = json.load(f)
    
    image_ids = data['images']
    
    source_images = Path(source_root) / 'images'
    source_masks = Path(source_root) / 'masks'
    dest_images = Path(dest_root) / 'images_png'
    dest_masks = Path(dest_root) / 'masks_png'
    
    copied_count = 0
    failed_count = 0
    
    for img_id in image_ids:
        # Try different extensions
        src_img = None
        src_mask = None
        
        # Check for image
        for ext in ['.png', '.tif', '.tiff']:
            candidate = source_images / f"{img_id}{ext}"
            if candidate.exists():
                src_img = candidate
                break
        
        # Check for mask
        for ext in ['.png', '.tif', '.tiff']:
            candidate = source_masks / f"{img_id}{ext}"
            if candidate.exists():
                src_mask = candidate
                break
        
        if src_img and src_mask:
            # Copy as PNG to destination (with _bal suffix to mark as balanced addition)
            dest_img = dest_images / f"{img_id}_bal.png"
            dest_mask_file = dest_masks / f"{img_id}_bal.png"
            
            shutil.copy(src_img, dest_img)
            shutil.copy(src_mask, dest_mask_file)
            copied_count += 1
        else:
            print(f"⚠️  Missing files for {img_id}")
            failed_count += 1
    
    print(f"\n✓ Successfully copied {copied_count} balanced images")
    print(f"✗ Failed to copy {failed_count} images")
    
    # Save log
    log = {
        'source': str(source_root),
        'destination': str(dest_root),
        'images_copied': copied_count,
        'images_failed': failed_count,
        'total_requested': len(image_ids)
    }
    
    with open('balanced_copy_log.json', 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"✓ Saved log to: balanced_copy_log.json")

if __name__ == '__main__':
    copy_balanced_images(
        selection_file='combined_balanced_images.json',
        source_root='data/Biodiversity_tiff/Train',
        dest_root='data/biodiversity/Train/Rural'
    )
