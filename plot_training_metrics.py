import pandas as pd
import matplotlib.pyplot as plt

# Read the metrics CSV
df = pd.read_csv('lightning_logs/biodiversity/ftunetformer-kd-512-crop-ms-augmented/version_4/metrics.csv')

# Get validation metrics (they appear at the end of each epoch)
val_metrics = df[df['val_mIoU'].notna()].copy()
val_metrics = val_metrics.reset_index(drop=True)

# Get training metrics (aggregate by epoch)
train_metrics = df.groupby('epoch').agg({
    'train_mIoU': 'last',
    'train_F1': 'last',
    'train_OA': 'last',
    'train_loss_epoch': 'last'
}).reset_index()

print("Current Epoch:", val_metrics['epoch'].max())
print("\nValidation Metrics:")
print(val_metrics[['epoch', 'val_mIoU', 'val_F1', 'val_OA']].tail(10))

print("\n\nTraining Metrics:")
print(train_metrics[['epoch', 'train_mIoU', 'train_F1', 'train_OA']].tail(10))

# Plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# mIoU
axes[0, 0].plot(val_metrics['epoch'], val_metrics['val_mIoU'], 'o-', label='Validation', linewidth=2)
axes[0, 0].plot(train_metrics['epoch'], train_metrics['train_mIoU'], 's-', label='Training', linewidth=2, alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('mIoU')
axes[0, 0].set_title('Mean Intersection over Union')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# F1
axes[0, 1].plot(val_metrics['epoch'], val_metrics['val_F1'], 'o-', label='Validation', linewidth=2)
axes[0, 1].plot(train_metrics['epoch'], train_metrics['train_F1'], 's-', label='Training', linewidth=2, alpha=0.7)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].set_title('F1 Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# OA
axes[1, 0].plot(val_metrics['epoch'], val_metrics['val_OA'], 'o-', label='Validation', linewidth=2)
axes[1, 0].plot(train_metrics['epoch'], train_metrics['train_OA'], 's-', label='Training', linewidth=2, alpha=0.7)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Overall Accuracy')
axes[1, 0].set_title('Overall Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Loss
axes[1, 1].plot(val_metrics['epoch'], val_metrics['val_loss'], 'o-', label='Validation', linewidth=2)
axes[1, 1].plot(train_metrics['epoch'], train_metrics['train_loss_epoch'], 's-', label='Training', linewidth=2, alpha=0.7)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Training Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
print("\n\nPlot saved as 'training_progress.png'")
plt.show()
