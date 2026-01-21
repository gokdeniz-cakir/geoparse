
import json
from pathlib import Path
import numpy as np

def check_bounds():
    data_dir = Path("data/dataset")
    meta_path = data_dir / "metadata.jsonl"
    
    max_x, max_y = -float('inf'), -float('inf')
    min_x, min_y = float('inf'), float('inf')
    
    count = 0
    with open(meta_path, "r") as f:
        for line in f:
            item = json.loads(line)
            vertices = item["structure"]["vertices"]
            
            for v, p in vertices.items():
                x, y = p
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
            
            count += 1
            if count > 5000: break # Check 5000 samples
            
    print(f"Checked {count} samples")
    print(f"X Range: [{min_x:.2f}, {max_x:.2f}]")
    print(f"Y Range: [{min_y:.2f}, {max_y:.2f}]")

if __name__ == "__main__":
    check_bounds()
