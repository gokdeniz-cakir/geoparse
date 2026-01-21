
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os
from generator import GeometryProblemGenerator, DiagramRenderer, RightTriangleWithAltitude, TriangleWithCevian

def generate_grid():
    gen = GeometryProblemGenerator(seed=42, randomize_labels=True)
    renderer = DiagramRenderer()
    
    output_dir = Path("data/grid_temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select 16 distinct types
    types = [
        "find_hypotenuse", "find_angle", "find_leg", "find_area",
        "find_isosceles_base", "find_isosceles_side", "find_equilateral_area", "find_equilateral_side",
        "find_altitude", "find_BD", "angle_sum_alpha", "angle_sum_two_known",
        "external_angle_find_interior", "external_angle_find_exterior", 
        "law_of_cosines_find_side", "cevian_angle_chase"
    ]
    
    image_paths = []
    titles = []
    
    print(f"Generating {len(types)} examples...")
    
    for i, ptype in enumerate(types):
        print(f"  {i+1}. {ptype}")
        # Generate problem
        # Need include_angles=True for right triangle angle problems
        problem = gen.generate_problem(problem_type=ptype, include_angles=True)
        
        fname = output_dir / f"ex_{i}.png"
        
        # Select appropriate render method
        if isinstance(problem.triangle, RightTriangleWithAltitude):
            renderer.render_altitude_triangle(
                problem.triangle, 
                hidden_sides=problem.structure.get("hidden", []),
                save_path=fname
            )
        elif isinstance(problem.triangle, TriangleWithCevian):
             renderer.render_cevian_triangle(
                problem.triangle,
                angles_to_show=problem.structure.get("angles_shown", {}),
                save_path=fname
             )
        elif "external_angle" in problem.structure:
            ext_data = problem.structure["external_angle"]
            # Detect which vertex has external angle
            ext_at = list(ext_data.keys())[0]
            ext_val = ext_data[ext_at]
            # Need to find "from" vertex (usually the one connecting to base)
            # Default logic: if at B, extend from C (as implemented in problem gen logic)
            # My current impl always extends B from C
            renderer.render_external_angle(
                problem.triangle,
                extend_at="B",
                extend_from="C",
                external_angle=ext_val,
                interior_angles=problem.structure.get("angles_shown", {}),
                save_path=fname
            )
        else:
            # Standard triangle
            renderer.render_triangle(
                problem.triangle,
                show_side_lengths=True,
                angles_to_show=problem.structure.get("angles_shown", {}),
                save_path=fname
            )
            
        image_paths.append(fname)
        titles.append(ptype)

    # Create grid
    print("Creating grid...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(image_paths):
            img = mpimg.imread(str(image_paths[i]))
            ax.imshow(img)
            ax.set_title(titles[i], fontsize=9)
            ax.axis('off')
            
            # Print goal for context
            # print(f"{titles[i]}: {problems[i].goal}")
            
    plt.tight_layout()
    plt.savefig("data/grid_16_examples.png", dpi=150)
    print("Saved data/grid_16_examples.png")
    
    # Cleanup
    for p in image_paths:
        try:
            os.remove(p)
        except:
            pass
    try:
        os.rmdir(output_dir)
    except:
        pass

if __name__ == "__main__":
    generate_grid()
