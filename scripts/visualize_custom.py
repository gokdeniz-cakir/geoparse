
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from parser.model import DiagramParser

def visualize_custom(image_path, output_path="debug_custom_vis.png"):
    device = "cpu"
    print(f"Visualizing: {image_path}")
    
    # 1. Load Parser
    parser = DiagramParser(output_dim=8)
    parser_path = "experiments/parser_v4/best_model.pt"
    try:
        ckpt = torch.load(parser_path, map_location=device)
        parser.load_state_dict(ckpt["model_state_dict"])
        parser.eval()
    except Exception as e:
        print(f"Failed to load parser: {e}")
        return

    # 2. Process Image
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open image: {e}")
        return
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0)
    
    # 3. Predict
    with torch.no_grad():
        coords = parser(input_tensor)[0] # [8]
        
    # Denormalize (x * 20)
    # Note: Training data normalized / 20.0 centered at 0.
    # To plot on 224x224 image, we need to map [-scale, scale] -> [0, 224].
    # But wait, looking at `visualize_preds.py`, how did we plot?
    # We plotted using matplotlib's coordinate system, which matched the generator's coord system.
    # To plot OVER the image, we need to map the predicted geometer-coords back to pixel-coords.
    # This is tricky because the ResNet sees a resized 224x224 image, but the coords are in "Geometry Space" [-10, 10].
    # There isn't a strict linear mapping unless we know the camera scale used during generation.
    #
    # HOWEVER, `visualize_preds.py` plotted the RECONSTRUCTED geometry side-by-side, not overlaid.
    # Let's do that. Side-by-side is safer and shows what the model "thinks" the shape is.
    
    p = coords.numpy() * 20.0
    px = p[0::2]
    py = p[1::2]
    
    # 4. Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left: Original Image
    axes[0].imshow(img_pil)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    # Right: Reconstructed Geometry
    # Invert Y to match screen coords if needed, but generator uses standard math coords (Y up).
    # Matplotlib default is Y up.
    
    # Vertices
    axes[1].scatter(px, py, c='red', s=50, label='Pred Vertices')
    
    # Edges (P1-P2-P3-P1)
    # Filter out P4 if close to P3
    points = []
    for i in range(4):
        points.append((px[i], py[i]))
        
    # Check simple distance P3-P4
    dist_p3_p4 = ((px[2]-px[3])**2 + (py[2]-py[3])**2)**0.5
    valid_count = 4
    if dist_p3_p4 < 0.5: # Threshold
        valid_count = 3
        
    polygon = points[:valid_count] + [points[0]] # Close loop
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    
    axes[1].plot(xs, ys, 'b-', linewidth=2)
    
    # Annotate labels
    labels = ['P1', 'P2', 'P3', 'P4']
    for i in range(valid_count):
        axes[1].annotate(labels[i], (px[i], py[i]), 
                         xytext=(5, 5), textcoords='offset points')
        
    axes[1].set_title("Reconstructed Shape")
    axes[1].grid(True)
    axes[1].set_aspect('equal')
    
    # Set limit to standard range to see shift
    axes[1].set_xlim(-15, 15)
    axes[1].set_ylim(-15, 15)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison saved to {output_path}")

if __name__ == "__main__":
    img_path = "/Users/gkden/Desktop/geoparse/data/testbookpdf/templates/Screenshot 2026-01-20 at 21.22.56.png"
    visualize_custom(img_path)
