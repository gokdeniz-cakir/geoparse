
import os
import json
import time
import argparse
from pathlib import Path
from multiprocessing import Pool
import random
from tqdm import tqdm
from functools import partial

from generator import GeometryProblemGenerator, DiagramRenderer, RightTriangleWithAltitude, TriangleWithCevian
from generator.sketchy_renderer import SketchyRenderer

# Configuration
OUTPUT_DIR = Path("data/dataset")
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_DIR = OUTPUT_DIR / "metadata"
NUM_SAMPLES_PER_TYPE = 2000
CHUNK_SIZE = 50
SKETCHY_PROBABILITY = 0.5  # 50% chance of using sketchy style

PROBLEM_TYPES = [
    # Right Triangle (Basic)
    "find_hypotenuse", "find_leg", "find_area",
    # Right Triangle (Trig)
    "find_angle", "find_side_with_angle",
    # Isosceles
    "find_isosceles_base", "find_isosceles_side",
    # Equilateral
    "find_equilateral_side", "find_equilateral_area",
    # Altitude
    "find_altitude", "find_BD", "find_DC",
    # Angle Sum
    "angle_sum_alpha", "angle_sum_two_known",
    # External Angle
    "external_angle_find_interior", "external_angle_find_exterior",
    # Law of Cosines
    "law_of_cosines_find_side", "law_of_cosines_find_angle",
    # Nested
    "cevian_angle_chase"
]

def generate_chunk(args):
    """
    Generate a chunk of geometry problems.
    args: (problem_type, chunk_id, seed_offset)
    """
    problem_type, chunk_id, seed_offset = args
    
    # Initialize generator with unique seed for this chunk
    seed = seed_offset + (chunk_id * 1000)
    gen = GeometryProblemGenerator(seed=seed, randomize_labels=True)
    
    # Create both renderers
    clean_renderer = DiagramRenderer()
    sketchy_renderer = SketchyRenderer(randomize=True)
    
    metadata = []
    
    for i in range(CHUNK_SIZE):
        try:
            # Generate problem
            # include_angles=True covers all bases; specific types ignore it if not needed
            problem = gen.generate_problem(problem_type=problem_type, include_angles=True)
            
            # Filename: type_chunk_index.png
            filename = f"{problem_type}_{chunk_id}_{i}.png"
            image_path = IMAGES_DIR / filename
            
            # Randomly select renderer style
            use_sketchy = random.random() < SKETCHY_PROBABILITY
            renderer = sketchy_renderer if use_sketchy else clean_renderer
            
            # Render based on triangle type
            if isinstance(problem.triangle, RightTriangleWithAltitude):
                renderer.render_altitude_triangle(
                    problem.triangle, 
                    hidden_sides=problem.structure.get("hidden", []),
                    save_path=image_path
                )
            elif isinstance(problem.triangle, TriangleWithCevian):
                 renderer.render_cevian_triangle(
                    problem.triangle,
                    angles_to_show=problem.structure.get("angles_shown", {}),
                    save_path=image_path
                 )
            elif "external_angle" in problem.structure:
                ext_data = problem.structure["external_angle"]
                ext_at = list(ext_data.keys())[0]
                ext_val = ext_data[ext_at]
                renderer.render_external_angle(
                    problem.triangle,
                    extend_at="B", # Default for now
                    extend_from="C",
                    external_angle=ext_val,
                    interior_angles=problem.structure.get("angles_shown", {}),
                    save_path=image_path
                )
            else:
                renderer.render_triangle(
                    problem.triangle,
                    show_side_lengths=True,
                    angles_to_show=problem.structure.get("angles_shown", {}),
                    save_path=image_path
                )
            
            # Record metadata
            meta = {
                "id": f"{problem_type}_{chunk_id}_{i}",
                "filename": filename,
                "problem_type": problem_type,
                "goal": problem.goal,
                "answer": problem.answer,
                "structure": problem.structure
            }
            metadata.append(meta)
            
        except Exception as e:
            print(f"Error generating {problem_type} #{i}: {e}")
            continue
            
    # Save chunk metadata
    meta_path = METADATA_DIR / f"{problem_type}_{chunk_id}.jsonl"
    with open(meta_path, "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
            
    return len(metadata)

def main():
    print(f"Starting dataset generation...")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Types: {len(PROBLEM_TYPES)}")
    print(f"Samples per type: {NUM_SAMPLES_PER_TYPE}")
    print(f"Total target: {len(PROBLEM_TYPES) * NUM_SAMPLES_PER_TYPE}")
    
    # Setup directories
    if OUTPUT_DIR.exists():
        import shutil
        print("Cleaning existing directory...")
        shutil.rmtree(OUTPUT_DIR)
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare tasks
    tasks = []
    num_chunks = NUM_SAMPLES_PER_TYPE // CHUNK_SIZE
    
    for ptype in PROBLEM_TYPES:
        for i in range(num_chunks):
            # args: (problem_type, chunk_id, seed_offset)
            tasks.append((ptype, i, random.randint(0, 100000)))
            
    print(f"Created {len(tasks)} tasks (chunks of {CHUNK_SIZE})")
    
    # Run parallel generation
    # Use fewer processes to avoid overwhelming the system/disk
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Using {num_workers} workers")
    
    total_generated = 0
    start_time = time.time()
    
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(generate_chunk, tasks), total=len(tasks)))
        total_generated = sum(results)
        
    duration = time.time() - start_time
    print(f"\nGeneration complete in {duration:.2f}s")
    print(f"Generated {total_generated} samples")
    
    # Merge metadata
    print("Merging metadata...")
    all_metadata = []
    for meta_file in METADATA_DIR.glob("*.jsonl"):
        with open(meta_file, "r") as f:
            for line in f:
                all_metadata.append(json.loads(line))
    
    final_meta_path = OUTPUT_DIR / "metadata.jsonl"
    with open(final_meta_path, "w") as f:
        for item in all_metadata:
            f.write(json.dumps(item) + "\n")
            
    print(f"Merged metadata saved to {final_meta_path}")
    print("Done!")

if __name__ == "__main__":
    main()
