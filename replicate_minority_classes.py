"""Replicate images with high Settlement/SemiNatural presence to balance the dataset."""

import json
import shutil
from pathlib import Path

def replicate_images(data_root, augmentation_list_file, replications=1):
    """
    Replicate images and their masks.
    
    Args:
        data_root: Path to Train directory
        augmentation_list_file: JSON file with list of images to replicate
        replications: Number of times to replicate each image (default=1 means image appears 2x total)
    """
    
    # Load augmentation list
    with open(augmentation_list_file, 'r') as f:
        aug_list = json.load(f)
    
    settlement_images = set(aug_list['settlement_images'])
    seminatural_images = set(aug_list['seminatural_images'])
    
    # Combine both lists (some images might have both classes)
    all_images_to_replicate = settlement_images.union(seminatural_images)
    
    print(f"Total unique images to replicate: {len(all_images_to_replicate)}")
    print(f"  - Only Settlement: {len(settlement_images - seminatural_images)}")
    print(f"  - Only SemiNatural: {len(seminatural_images - settlement_images)}")
    print(f"  - Both classes: {len(settlement_images.intersection(seminatural_images))}")
    print(f"Replications per image: {replications}")
    print()
    
    # Define paths
    rural_dir = Path(data_root) / 'Rural'
    images_dir = rural_dir / 'images_png'
    masks_dir = rural_dir / 'masks_png'
    
    # Check directories exist
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist!")
        return
    if not masks_dir.exists():
        print(f"ERROR: {masks_dir} does not exist!")
        return
    
    # Replicate images
    replicated_count = 0
    
    for img_id in sorted(all_images_to_replicate):
        image_path = images_dir / f"{img_id}.png"
        mask_path = masks_dir / f"{img_id}.png"
        
        if not image_path.exists():
            print(f"WARNING: Image not found: {image_path}")
            continue
        if not mask_path.exists():
            print(f"WARNING: Mask not found: {mask_path}")
            continue
        
        # Create replications
        for rep_num in range(1, replications + 1):
            # New filenames with suffix
            new_image_path = images_dir / f"{img_id}_rep{rep_num}.png"
            new_mask_path = masks_dir / f"{img_id}_rep{rep_num}.png"
            
            # Copy files
            shutil.copy2(image_path, new_image_path)
            shutil.copy2(mask_path, new_mask_path)
        
        replicated_count += 1
        
        if replicated_count % 100 == 0:
            print(f"Replicated {replicated_count}/{len(all_images_to_replicate)} images...")
    
    print(f"\n✓ Successfully replicated {replicated_count} images")
    print(f"  Original training set: 2307 images")
    print(f"  Added replications: {replicated_count * replications}")
    print(f"  New training set size: {2307 + replicated_count * replications} images")
    
    # Save replication log
    log_data = {
        'original_size': 2307,
        'replicated_images': replicated_count,
        'replications_per_image': replications,
        'total_added': replicated_count * replications,
        'new_size': 2307 + replicated_count * replications,
        'settlement_only': len(settlement_images - seminatural_images),
        'seminatural_only': len(seminatural_images - settlement_images),
        'both_classes': len(settlement_images.intersection(seminatural_images))
    }
    
    with open('replication_log.json', 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n✓ Saved replication log to: replication_log.json")

if __name__ == '__main__':
    replicate_images(
        data_root='data/biodiversity/Train',
        augmentation_list_file='train_augmentation_list.json',
        replications=1  # Each image will be replicated 1x (appear 2x total)
    )
