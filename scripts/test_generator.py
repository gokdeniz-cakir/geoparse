"""
Quick test of the synthetic diagram generator.
"""

from pathlib import Path
from generator import RightTriangleGenerator

# Create output directory
output_dir = Path("data/synthetic/test")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize generator with fixed seed for reproducibility
gen = RightTriangleGenerator(seed=42)

# Generate a few sample problems
print("Generating sample problems...")
for i in range(3):
    problem = gen.generate_problem()
    
    # Save the image
    image_path = output_dir / f"sample_{i}.png"
    gen.renderer.render_right_triangle(
        problem.triangle,
        show_labels=True,
        show_right_angle=True,
        show_side_lengths=True,
        save_path=image_path
    )
    
    print(f"\nProblem {i + 1}:")
    print(f"  Type: {problem.structure['problem_type']}")
    print(f"  Goal: {problem.goal}")
    print(f"  Answer: {problem.answer}")
    print(f"  Image: {image_path}")

print(f"\nSamples saved to {output_dir}")
