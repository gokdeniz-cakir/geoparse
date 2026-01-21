"""Test keypoint model on real textbook images."""

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path

from parser.keypoint_model import KeypointDetector


def test_on_image(image_path: str, model_path: str = "experiments/keypoint_v1/best_model.pt"):
    """Test keypoint detection on a single image."""
    device = "cpu"  # Use CPU for inference
    
    # Load model
    model = KeypointDetector(num_keypoints=4, heatmap_size=56)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from {model_path} (val_loss={ckpt['val_loss']:.6f})")
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        heatmaps = model(img_tensor)  # [1, 4, 56, 56]
        coords = model.predict_keypoints(img_tensor)  # [1, 4, 2]
    
    coords = coords[0].numpy()  # [4, 2] in range [0, 1]
    heatmaps = heatmaps[0].numpy()  # [4, 56, 56]
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original image with predicted keypoints
    ax = axes[0, 0]
    ax.imshow(img_pil)
    # Scale coords to image size
    h, w = img_pil.size[1], img_pil.size[0]
    for i, (x, y) in enumerate(coords[:4]):
        px, py = x * w, y * h
        ax.scatter(px, py, c='red', s=100, marker='o')
        ax.annotate(f'P{i+1}', (px, py), fontsize=12, color='red')
    ax.set_title("Predicted Keypoints")
    ax.axis("off")
    
    # Individual heatmaps
    for i in range(4):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row, col]
        ax.imshow(heatmaps[i], cmap='hot')
        ax.set_title(f"P{i+1} Heatmap")
        ax.axis("off")
    
    # Combined heatmap
    axes[1, 2].imshow(heatmaps.max(axis=0), cmap='hot')
    axes[1, 2].set_title("Combined")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    output_path = "keypoint_test_result.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")
    
    # Print coordinates
    print(f"\nPredicted Coordinates (normalized 0-1):")
    for i, (x, y) in enumerate(coords):
        print(f"  P{i+1}: ({x:.3f}, {y:.3f})")
    
    return coords


if __name__ == "__main__":
    # Test on user's textbook screenshot
    test_path = "data/testbookpdf/templates/Screenshot 2026-01-20 at 21.22.56.png"
    
    if Path(test_path).exists():
        print(f"Testing on: {test_path}")
        coords = test_on_image(test_path)
    else:
        # Fallback to synthetic
        test_path = "data/dataset/images/angle_sum_alpha_0_0.png"
        print(f"Using synthetic: {test_path}")
        coords = test_on_image(test_path)
