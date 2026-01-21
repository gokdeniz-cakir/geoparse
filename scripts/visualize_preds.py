
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from torchvision import transforms
from parser.model import DiagramParser

def visualize_prediction(model_path, image_path, output_path):
    device = "cpu"  # Use CPU for inference to avoid interfering with training
    
    # Load model
    model = DiagramParser(pretrained=False, output_dim=8)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load and transform image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        
    # Denormalize predictions (raw tensor output)
    # Output is [Ax, Ay, Bx, By, Cx, Cy, Dx, Dy] normalized by 10.0
    # But coordinate space is abstract (0-10 or so).
    # We need to map this back to the image pixels.
    # The generator produces images where (0,0) is bottom-left in plot coords,
    # but (0,0) top-left in image.
    # Actually, let's just assume the model learned the mapping to the 224x224 space implicitly via the transform?
    # No, the model targets are normalized Cartesian coords.
    # We need to trust the model learned the visual features.
    # The coordinate system might be tricky. The training targets were raw cartesian coords / 10.
    
    # Plotting: Side-by-side comparison
    pred = output[0].cpu().numpy() * 20.0  # Normalized by 20.0 in dataset

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Original Image
    # Show the resized tensor image (what model sees)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_display = inv_normalize(img_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    
    axes[0].imshow(img_display)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Right: Reconstructed Geometry
    # Plot using the raw predicted coordinates
    ax = axes[1]
    
    # Extract points
    points = []
    # Explicitly parse as P1, P2, P3, P4
    # Check padding: if P4 is close to P3, it's padding
    p_coords = []
    for i in range(4):
        p_coords.append((pred[2*i], pred[2*i+1]))
        
    # P1, P2, P3 are always valid (assuming convex hull property from sorting)
    # The sort was Top-Down. P1=Top, P2=Left, P3=Right (roughly).
    # P4 is padding if it duplicates P3.
    
    valid_points = p_coords[:3]
    p3 = p_coords[2]
    p4 = p_coords[3]
    
    # Distance check for P4
    dist_p4_p3 = ((p4[0]-p3[0])**2 + (p4[1]-p3[1])**2)**0.5
    if dist_p4_p3 > 1.0: # Threshold for distinct point
        valid_points.append(p4)
        
    for i, (x, y) in enumerate(valid_points):
        ax.scatter(x, y, c='red', s=100)
        ax.text(x+0.2, y+0.2, f"P{i+1}", fontsize=12)
    
    # Connect dots
    if(len(valid_points) >= 3):
        # Draw triangle P1-P2-P3-P1
        poly = list(valid_points[:3])
        poly.append(valid_points[0]) 
        px, py = zip(*poly)
        ax.plot(px, py, 'b-', linewidth=2)
        
        # If P4 exists, connect it. Usually altitude or cevian.
        # It's hard to know topology without type, but usually connect to P1 (Top)
        if len(valid_points) == 4:
            ax.plot([valid_points[0][0], valid_points[3][0]], 
                   [valid_points[0][1], valid_points[3][1]], 'g--', linewidth=2)

    ax.set_title("Model Reconstruction\n(Predicted Coords)")
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Set limits based on normalization range [-20, 20] roughly
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    # Test on a few images from the dataset
    data_dir = Path("data/dataset/images")
    output_dir = Path("data/preds")
    output_dir.mkdir(exist_ok=True)
    
    # Pick random images
    import random
    all_imgs = list(data_dir.glob("*.png"))
    
    test_imgs = all_imgs[:3] # First 3
    if len(all_imgs) > 3:
        test_imgs = random.sample(all_imgs, 3)
        
    for i, img_path in enumerate(test_imgs):
        visualize_prediction(
            "experiments/parser_v4/best_model.pt",
            img_path,
            output_dir / f"pred_{i}.png"
        )
