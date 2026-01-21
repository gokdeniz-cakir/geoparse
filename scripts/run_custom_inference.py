
import sys
from pathlib import Path
from pipeline.solve import GeometrySolver

def run_inference(image_path):
    print(f"Running inference on: {image_path}")
    
    # Initialize Solver
    # Assuming running from project root
    solver = GeometrySolver(
        parser_path="experiments/parser_v4/best_model.pt",
        solver_path="experiments/solver_baseline/best_solver.pt"
    )
    
    # Solve
    # We don't know the problem type, so we pass 'unknown'. 
    # Refinement will mostly enforce coordinate fidelity.
    solution = solver.solve(image_path, problem_type="unknown")
    
    print("\n--- Solution ---")
    print(f"Area: {solution.area:.2f}")
    
    print("\nGeometric Properties:")
    print(f"  Angles: {[round(a, 1) for a in solution.angles]}")
    print(f"  Sides : {[round(s, 2) for s in solution.sides]}")
    
    print("\nCoordinates (Refined):")
    for i in range(0, 8, 2):
        x = solution.refined_coords[i]
        y = solution.refined_coords[i+1]
        print(f"  P{i//2 + 1}: ({x:.2f}, {y:.2f})")

if __name__ == "__main__":
    # Use the path provided by user
    img_path = "/Users/gkden/Desktop/geoparse/data/testbookpdf/templates/Screenshot 2026-01-20 at 21.22.56.png"
    run_inference(img_path)
