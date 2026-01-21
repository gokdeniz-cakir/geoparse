
import subprocess
import sys
from pathlib import Path

def run_training():
    cmd = [
        sys.executable,
        "-m", "parser.train",
        "--data_dir", "data/dataset",
        "--output_dir", "experiments/parser_v4",
        "--epochs", "20",
        "--batch_size", "64",
        "--lr", "1e-3",
        "--num_workers", "8"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_training()
