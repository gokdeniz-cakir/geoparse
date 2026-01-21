"""Quick test to verify sketchy renderer integration in dataset generation."""

import random
from pathlib import Path
from generator import GeometryProblemGenerator, DiagramRenderer
from generator.sketchy_renderer import SketchyRenderer

def test_render_styles():
    """Generate 6 samples alternating clean/sketchy to verify integration."""
    output_dir = Path("test_sketchy_samples")
    output_dir.mkdir(exist_ok=True)
    
    gen = GeometryProblemGenerator(seed=42, randomize_labels=True)
    clean_renderer = DiagramRenderer()
    sketchy_renderer = SketchyRenderer(randomize=True)
    
    problem_types = ["find_hypotenuse", "angle_sum_alpha", "find_isosceles_base"]
    
    for i, ptype in enumerate(problem_types):
        problem = gen.generate_problem(problem_type=ptype, include_angles=True)
        
        # Clean version
        clean_path = output_dir / f"{i}_clean_{ptype}.png"
        clean_renderer.render_triangle(
            problem.triangle,
            show_side_lengths=True,
            angles_to_show=problem.structure.get("angles_shown", {}),
            save_path=clean_path
        )
        
        # Sketchy version
        sketchy_path = output_dir / f"{i}_sketchy_{ptype}.png"
        sketchy_renderer.render_triangle(
            problem.triangle,
            show_side_lengths=True,
            angles_to_show=problem.structure.get("angles_shown", {}),
            save_path=sketchy_path
        )
        
        print(f"Generated: {clean_path.name} and {sketchy_path.name}")
        
    print(f"\nSamples saved to {output_dir}/")

if __name__ == "__main__":
    test_render_styles()
