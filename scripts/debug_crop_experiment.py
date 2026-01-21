
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms
from parser.model import DiagramParser

def debug_crop_experiment(image_path, output_path="debug_crop_vis.png"):
    print(f"Running Crop Experiment on: {image_path}")
    
    # 1. Load Image
    img_orig = Image.open(image_path).convert("RGB")
    
    # 2. Auto-Crop Logic
    # Invert to find bounding box of black pixels (content)
    img_inv = ImageOps.invert(img_orig)
    bbox = img_inv.getbbox()
    
    if bbox:
        # Add some padding
        left, upper, right, lower = bbox
        w, h = right - left, lower - upper
        pad = max(w, h) * 0.1
        
        left = max(0, left - pad)
        upper = max(0, upper - pad)
        right = min(img_orig.width, right + pad)
        lower = min(img_orig.height, lower + pad)
        
        img_cropped = img_orig.crop((left, upper, right, lower))
        print(f"Cropped from {img_orig.size} to {img_cropped.size}")
    else:
        print("No content found to crop!")
        img_cropped = img_orig

    # 3. Load Model
    device = "cpu"
    parser = DiagramParser(output_dim=8)
    parser.load_state_dict(torch.load("experiments/parser_v4/best_model.pt", map_location=device)["model_state_dict"])
    parser.eval()
    
    # 4. Predict on Cropped
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        coords = parser(transform(img_cropped).unsqueeze(0))[0]
        
    # 5. Visualize
    p = coords.numpy() * 20.0
    px = p[0::2]
    py = p[1::2]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left: Cropped Input
    axes[0].imshow(img_cropped)
    axes[0].set_title("Auto-Cropped Input")
    axes[0].axis("off")
    
    # Right: Prediction
    axes[1].scatter(px, py, c='red', s=50, label='Pred Vertices')
    
    # Draw simple loop
    points = list(zip(px, py))
    # Filter overlapping (P3/P4)
    if ((px[2]-px[3])**2 + (py[2]-py[3])**2)**0.5 < 0.5:
        points = points[:3]
    points.append(points[0])
    
    xs, ys = zip(*points)
    axes[1].plot(xs, ys, 'b-')
    
    # Add labels
    for i, (x, y) in enumerate(zip(px[:3], py[:3])):
         axes[1].annotate(f"P{i+1}", (x, y))
         
    axes[1].set_title("Reconstructed Shape")
    axes[1].grid(True)
    axes[1].set_aspect('equal')
    axes[1].set_xlim(-15, 15)
    axes[1].set_ylim(-15, 15)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    img_path = "/Users/gkden/Desktop/geoparse/data/testbookpdf/templates/Screenshot 2026-01-20 at 21.22.56.png"
    debug_crop_experiment(img_path)
