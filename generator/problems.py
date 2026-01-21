"""
Problem generators for synthetic geometry diagrams.
"""

import random
import math
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .primitives import Point, RightTriangle, IsoscelesTriangle, EquilateralTriangle, RightTriangleWithAltitude, TriangleWithCevian
from .renderer import DiagramRenderer
from .randomization import LabelRandomizer


@dataclass
class GeneratedProblem:
    """A generated geometry problem with image and structure."""
    triangle: RightTriangle
    structure: dict
    goal: str
    answer: float
    image_path: Optional[Path] = None


class GeometryProblemGenerator:
    """
    Generates random geometry problems.
    
    Problem types:
    - Right Triangle: hypotenuse, leg, area, angle
    - Isosceles Triangle: base, leg, height, base_angle
    - Equilateral Triangle: side, area, height
    """
    
    def __init__(
        self,
        min_side: float = 2,
        max_side: float = 10,
        seed: Optional[int] = None,
        randomize_labels: bool = False
    ):
        self.min_side = min_side
        self.max_side = max_side
        self.renderer = DiagramRenderer()
        self.randomize_labels = randomize_labels
        self.label_randomizer = LabelRandomizer(seed)
        
        if seed is not None:
            random.seed(seed)
    
    def generate_triangle(
        self,
        leg_a: Optional[float] = None,
        leg_b: Optional[float] = None,
        rotation: Optional[float] = None
    ) -> RightTriangle:
        """
        Generate a right triangle with given or random leg lengths.
        
        The right angle is at C, positioned at origin.
        A is along one leg, B along the other.
        
        Args:
            leg_a: Length of leg BC (if None, random)
            leg_b: Length of leg AC (if None, random)
            rotation: Rotation angle in degrees (if None, random)
        """
        # Generate random sides if not provided
        if leg_a is None:
            leg_a = random.uniform(self.min_side, self.max_side)
        if leg_b is None:
            leg_b = random.uniform(self.min_side, self.max_side)
        
        # Calculate hypotenuse
        hypotenuse = math.sqrt(leg_a**2 + leg_b**2)
        
        # Initial positions (right angle at origin, legs along axes)
        C = Point(0, 0, "C")
        A = Point(leg_b, 0, "A")  # leg_b = AC
        B = Point(0, leg_a, "B")  # leg_a = BC
        
        # Apply random rotation
        if rotation is None:
            rotation = random.uniform(0, 360)
        
        angle_rad = math.radians(rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        def rotate(p: Point) -> Point:
            new_x = p.x * cos_a - p.y * sin_a
            new_y = p.x * sin_a + p.y * cos_a
            return Point(new_x, new_y, p.label)
        
        A, B, C = rotate(A), rotate(B), rotate(C)
        
        # Round side lengths for cleaner problems
        leg_a = round(leg_a, 1)
        leg_b = round(leg_b, 1)
        hypotenuse = round(hypotenuse, 2)
        
        return RightTriangle(
            A=A, B=B, C=C,
            side_a=leg_a,
            side_b=leg_b,
            side_c=hypotenuse
        )

    def generate_isosceles_triangle(
        self,
        side_equal: Optional[float] = None,
        side_base: Optional[float] = None,
        rotation: Optional[float] = None
    ) -> IsoscelesTriangle:
        """
        Generate an isosceles triangle with AB = AC.
        """
        if side_equal is None:
            side_equal = random.uniform(self.min_side, self.max_side)
        if side_base is None:
            # Base must be less than 2*side_equal for a valid triangle
            side_base = random.uniform(1.0, 1.8 * side_equal)
            if side_base < self.min_side: side_base = self.min_side
            
        # Height h^2 + (base/2)^2 = equal^2
        height = math.sqrt(side_equal**2 - (side_base/2)**2)
        
        # Initial positions: A at (0, height), B at (-base/2, 0), C at (base/2, 0)
        A = Point(0, height, "A")
        B = Point(-side_base/2, 0, "B")
        C = Point(side_base/2, 0, "C")
        
        # Centroid at (0, height/3)
        centroid = Point(0, height/3)
        # Shift to origin centroid-ish
        A = Point(A.x - centroid.x, A.y - centroid.y, A.label)
        B = Point(B.x - centroid.x, B.y - centroid.y, B.label)
        C = Point(C.x - centroid.x, C.y - centroid.y, C.label)
        
        # Apply random rotation
        if rotation is None:
            rotation = random.uniform(0, 360)
        
        angle_rad = math.radians(rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        def rotate(p: Point) -> Point:
            new_x = p.x * cos_a - p.y * sin_a
            new_y = p.x * sin_a + p.y * cos_a
            return Point(new_x, new_y, p.label)
        
        A, B, C = rotate(A), rotate(B), rotate(C)
        
        return IsoscelesTriangle(
            A=A, B=B, C=C,
            side_equal=round(side_equal, 1),
            side_base=round(side_base, 1)
        )

    def generate_equilateral_triangle(
        self,
        side_length: Optional[float] = None,
        rotation: Optional[float] = None
    ) -> EquilateralTriangle:
        """
        Generate an equilateral triangle AB = BC = CA.
        """
        if side_length is None:
            side_length = random.uniform(self.min_side, self.max_side)
            
        height = math.sqrt(3)/2 * side_length
        
        A = Point(0, height, "A")
        B = Point(-side_length/2, 0, "B")
        C = Point(side_length/2, 0, "C")
        
        centroid = Point(0, height/3)
        A = Point(A.x - centroid.x, A.y - centroid.y, A.label)
        B = Point(B.x - centroid.x, B.y - centroid.y, B.label)
        C = Point(C.x - centroid.x, C.y - centroid.y, C.label)
        
        if rotation is None:
            rotation = random.uniform(0, 360)
            
        angle_rad = math.radians(rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        def rotate(p: Point) -> Point:
            new_x = p.x * cos_a - p.y * sin_a
            new_y = p.x * sin_a + p.y * cos_a
            return Point(new_x, new_y, p.label)
            
        A, B, C = rotate(A), rotate(B), rotate(C)
        
        return EquilateralTriangle(
            A=A, B=B, C=C,
            side_length=round(side_length, 1)
        )

    def generate_altitude_triangle(
        self,
        leg_AB: Optional[float] = None,
        leg_AC: Optional[float] = None,
        rotation: Optional[float] = None
    ) -> RightTriangleWithAltitude:
        """
        Generate a right triangle with altitude to the hypotenuse.
        Right angle at A, altitude foot D on BC.
        """
        if leg_AB is None:
            leg_AB = random.uniform(self.min_side, self.max_side)
        if leg_AC is None:
            leg_AC = random.uniform(self.min_side, self.max_side)
        if rotation is None:
            rotation = random.uniform(0, 360)
            
        return RightTriangleWithAltitude.from_legs(leg_AB, leg_AC, rotation)
    
    def generate_problem(
        self,
        problem_type: Optional[str] = None,
        include_angles: bool = False
    ) -> GeneratedProblem:
        """
        Generate a complete problem with triangle, goal, and answer.
        """
        right_types = ["find_hypotenuse", "find_leg", "find_area"]
        if include_angles:
            right_types.extend(["find_angle", "find_side_with_angle"])
            
        iso_types = ["find_isosceles_base", "find_isosceles_side"]
        eq_types = ["find_equilateral_side", "find_equilateral_area"]
        altitude_types = ["find_altitude", "find_BD", "find_DC"]
        angle_sum_types = ["angle_sum_alpha", "angle_sum_two_known"]
        external_types = ["external_angle_find_interior", "external_angle_find_exterior"]
        cosine_types = ["law_of_cosines_find_side", "law_of_cosines_find_angle"]
        cevian_types = ["cevian_angle_chase"]
        
        all_types = right_types + iso_types + eq_types + altitude_types + angle_sum_types + external_types + cosine_types + cevian_types
        
        if problem_type is None:
            problem_type = random.choice(all_types)
        
        # Default: no angle arcs to display
        angles_to_show = None
        
        if problem_type in right_types:
            triangle = self.generate_triangle()
            # Calculate actual angles at A and B
            angle_A = round(math.degrees(math.atan(triangle.side_a / triangle.side_b)), 1)
            angle_B = round(90 - angle_A, 1)
            
            if problem_type == "find_hypotenuse":
                goal = f"Given BC = {triangle.side_a} and CA = {triangle.side_b}, find AB"
                answer = triangle.side_c
                given = ["BC", "CA"]
                hidden = ["AB"]
            elif problem_type == "find_leg":
                if random.random() < 0.5:
                    goal = f"Given AB = {triangle.side_c} and BC = {triangle.side_a}, find CA"
                    answer = triangle.side_b
                    given = ["AB", "BC"]
                    hidden = ["CA"]
                else:
                    goal = f"Given AB = {triangle.side_c} and CA = {triangle.side_b}, find BC"
                    answer = triangle.side_a
                    given = ["AB", "CA"]
                    hidden = ["BC"]
            elif problem_type == "find_area":
                goal = f"Given BC = {triangle.side_a} and CA = {triangle.side_b}, find the area"
                answer = round(0.5 * triangle.side_a * triangle.side_b, 2)
                given = ["BC", "CA"]
                hidden = ["area"]
            elif problem_type == "find_angle":
                if random.random() < 0.5:
                    goal = f"Given BC = {triangle.side_a} and CA = {triangle.side_b}, find angle A"
                    answer = angle_A
                    given = ["BC", "CA"]
                    hidden = ["angle_A"]
                    angles_to_show = {"B": angle_B}
                else:
                    goal = f"Given BC = {triangle.side_a} and CA = {triangle.side_b}, find angle B"
                    answer = angle_B
                    given = ["BC", "CA"]
                    hidden = ["angle_B"]
                    angles_to_show = {"A": angle_A}
            elif problem_type == "find_side_with_angle":
                if random.random() < 0.5:
                    goal = f"Given angle A = {angle_A}° and BC = {triangle.side_a}, find CA"
                    answer = triangle.side_b
                    given = ["angle_A", "BC"]
                    hidden = ["CA"]
                    angles_to_show = {"A": angle_A}
                else:
                    goal = f"Given angle B = {angle_B}° and CA = {triangle.side_b}, find BC"
                    answer = triangle.side_a
                    given = ["angle_B", "CA"]
                    hidden = ["BC"]
                    angles_to_show = {"B": angle_B}
            
            structure = triangle.to_dict()
            structure["angles"] = {"A": angle_A, "B": angle_B, "C": 90}
            if angles_to_show:
                structure["angles_shown"] = angles_to_show

        elif problem_type in iso_types:
            triangle = self.generate_isosceles_triangle()
            # Calculate top angle A (angle between AB and AC)
            # sin(A/2) = (base/2) / side_equal
            angle_A = round(2 * math.degrees(math.asin((triangle.side_base/2) / triangle.side_equal)), 1)
            angle_BC = round((180 - angle_A) / 2, 1)
            
            if problem_type == "find_isosceles_base":
                goal = f"Given equal sides = {triangle.side_equal} and top angle = {angle_A}°, find BC"
                answer = triangle.side_base
                given = ["AB", "AC", "angle_A"]
                hidden = ["BC"]
                angles_to_show = {"A": angle_A}
            elif problem_type == "find_isosceles_side":
                goal = f"Given BC = {triangle.side_base} and top angle = {angle_A}°, find equal sides"
                answer = triangle.side_equal
                given = ["BC", "angle_A"]
                hidden = ["AB", "AC"]
                angles_to_show = {"A": angle_A}
            
            structure = triangle.to_dict()
            structure["angles"] = {"A": angle_A, "B": angle_BC, "C": angle_BC}
            if angles_to_show:
                structure["angles_shown"] = angles_to_show

        elif problem_type in eq_types:
            triangle = self.generate_equilateral_triangle()
            if problem_type == "find_equilateral_side":
                # Find side given area? No, let's keep it simple: "What is the side length?"
                # Actually, maybe "Given height = ..., find side"
                height = round(math.sqrt(3)/2 * triangle.side_length, 2)
                goal = f"Given height = {height}, find the side length"
                answer = triangle.side_length
                given = ["height"]
                hidden = ["side"]
            elif problem_type == "find_equilateral_area":
                goal = f"Given side = {triangle.side_length}, find the area"
                answer = round(math.sqrt(3)/4 * triangle.side_length**2, 2)
                given = ["side"]
                hidden = ["area"]
            
            structure = triangle.to_dict()
            structure["angles"] = {"A": 60.0, "B": 60.0, "C": 60.0}

        elif problem_type in altitude_types:
            triangle = self.generate_altitude_triangle()
            
            if problem_type == "find_altitude":
                # Given AB and AC, find AD (altitude)
                goal = f"Given AB = {triangle.AB} and AC = {triangle.AC}, find AD"
                answer = triangle.AD
                given = ["AB", "AC"]
                hidden = ["AD", "BD", "DC"]
            elif problem_type == "find_BD":
                # Given DC and AC, use geometric mean: AC² = DC × BC
                goal = f"Given DC = {triangle.DC} and AC = {triangle.AC}, find BD"
                answer = triangle.BD
                given = ["DC", "AC"]
                hidden = ["AB", "AD", "BD"]
            elif problem_type == "find_DC":
                # Given BD and AB, use geometric mean: AB² = BD × BC  
                goal = f"Given BD = {triangle.BD} and AB = {triangle.AB}, find DC"
                answer = triangle.DC
                given = ["BD", "AB"]
                hidden = ["AC", "AD", "DC"]
            
            structure = triangle.to_dict()
            structure["angles"] = {"A": 90.0}

        elif problem_type in angle_sum_types:
            # Use isosceles triangle for realistic angle problems
            triangle = self.generate_isosceles_triangle()
            
            # Calculate actual angles
            # For isosceles: base angles are equal, apex angle is different
            # Using law of cosines to get apex angle
            if triangle.side_base and triangle.side_equal:
                cos_A = 1 - (triangle.side_base**2) / (2 * triangle.side_equal**2)
                cos_A = max(-1, min(1, cos_A))  # Clamp for numerical stability
                angle_A = round(math.degrees(math.acos(cos_A)), 1)
            else:
                angle_A = round(random.uniform(30, 100), 1)
            angle_BC = round((180 - angle_A) / 2, 1)
            
            if problem_type == "angle_sum_alpha":
                # Given two angles, find α (third angle)
                goal = f"Given ∠B = {angle_BC}° and ∠C = {angle_BC}°, find α (∠A)"
                answer = angle_A
                given = ["angle_B", "angle_C"]
                hidden = ["angle_A"]
                angles_to_show = {"B": angle_BC, "C": angle_BC}
            elif problem_type == "angle_sum_two_known":
                # Given apex angle, find base angles
                goal = f"Given ∠A = {angle_A}°, find α (base angles)"
                answer = angle_BC
                given = ["angle_A"]
                hidden = ["angle_B", "angle_C"]
                angles_to_show = {"A": angle_A}
            
            structure = triangle.to_dict()
            structure["angles"] = {"A": angle_A, "B": angle_BC, "C": angle_BC}
            if angles_to_show:
                structure["angles_shown"] = angles_to_show

        elif problem_type in external_types:
            # Use isosceles for consistent angle relationships
            triangle = self.generate_isosceles_triangle()
            
            # Calculate angles
            if triangle.side_base and triangle.side_equal:
                cos_A = 1 - (triangle.side_base**2) / (2 * triangle.side_equal**2)
                cos_A = max(-1, min(1, cos_A))
                angle_A = round(math.degrees(math.acos(cos_A)), 1)
            else:
                angle_A = round(random.uniform(30, 100), 1)
            angle_B = round((180 - angle_A) / 2, 1)
            angle_C = angle_B
            
            # External angle at B = A + C (sum of non-adjacent interior angles)
            external_B = round(angle_A + angle_C, 1)
            
            if problem_type == "external_angle_find_interior":
                # Given external angle and one interior, find the other
                goal = f"Given external ∠B = {external_B}° and ∠C = {angle_C}°, find ∠A"
                answer = angle_A
                given = ["external_B", "angle_C"]
                hidden = ["angle_A"]
                angles_to_show = {"C": angle_C}
            elif problem_type == "external_angle_find_exterior":
                # Given two interior angles, find external
                goal = f"Given ∠A = {angle_A}° and ∠C = {angle_C}°, find external ∠B"
                answer = external_B
                given = ["angle_A", "angle_C"]
                hidden = ["external_B"]
                angles_to_show = {"A": angle_A, "C": angle_C}
            
            structure = triangle.to_dict()
            structure["angles"] = {"A": angle_A, "B": angle_B, "C": angle_C}
            structure["external_angle"] = {"B": external_B}
            if angles_to_show:
                structure["angles_shown"] = angles_to_show

        elif problem_type in cosine_types:
            # Generate an obtuse triangle for Law of Cosines
            # Use integer side lengths that produce nice results
            side_a = random.randint(3, 8)
            side_b = random.randint(3, 8)
            # Use an angle between 60-120 degrees for interesting results
            angle_C = random.choice([60, 75, 90, 100, 120])
            
            # Law of Cosines: c² = a² + b² - 2ab·cos(C)
            angle_C_rad = math.radians(angle_C)
            c_squared = side_a**2 + side_b**2 - 2 * side_a * side_b * math.cos(angle_C_rad)
            side_c = math.sqrt(c_squared)
            
            # Calculate other angles using Law of Cosines
            cos_A = (side_b**2 + side_c**2 - side_a**2) / (2 * side_b * side_c)
            cos_A = max(-1, min(1, cos_A))
            angle_A = round(math.degrees(math.acos(cos_A)), 1)
            angle_B = round(180 - angle_A - angle_C, 1)
            
            # Create isosceles triangle geometry (positions don't matter for answer)
            triangle = self.generate_isosceles_triangle(side_equal=side_a, side_base=side_b)
            
            if problem_type == "law_of_cosines_find_side":
                # Format answer as √N if it's irrational
                if abs(side_c - round(side_c)) < 0.01:
                    answer_str = str(round(side_c))
                    answer = round(side_c, 2)
                else:
                    answer = round(side_c, 2)
                    # Check if c² is a nice integer for display
                    if abs(c_squared - round(c_squared)) < 0.1:
                        answer_str = f"√{round(c_squared)}"
                    else:
                        answer_str = f"{answer}"
                
                goal = f"Given a = {side_a}, b = {side_b}, ∠C = {angle_C}°, find c"
                given = ["a", "b", "angle_C"]
                hidden = ["c"]
                angles_to_show = {"C": angle_C}
            elif problem_type == "law_of_cosines_find_angle":
                # Given three sides, find an angle
                goal = f"Given a = {side_a}, b = {side_b}, c = {round(side_c, 2)}, find ∠A"
                answer = angle_A
                given = ["a", "b", "c"]
                hidden = ["angle_A"]
                angles_to_show = {}
            
            structure = triangle.to_dict()
            structure["angles"] = {"A": angle_A, "B": angle_B, "C": angle_C}
            structure["law_of_cosines"] = True
            if angles_to_show:
                structure["angles_shown"] = angles_to_show

        elif problem_type in cevian_types:
            # Generate random triangle angles
            angle_B = round(random.uniform(40, 80), 1)
            angle_C = round(random.uniform(40, 80), 1)
            # Cevian somewhere in the middle
            ratio = random.uniform(0.3, 0.7)
            
            triangle = TriangleWithCevian.from_angles(
                angle_B=angle_B, 
                angle_C=angle_C, 
                cevian_ratio=ratio
            )
            
            # Problem: given B and BAD, find ADB (exterior to ABD, interior to ADC)
            # Or other combinations
            
            if problem_type == "cevian_angle_chase":
                # Option 1: find ADB given B and BAD
                # angle_ADB = 180 - angle_B - angle_BAD
                goal = f"Given ∠B = {triangle.angle_ABD}°, ∠BAD = {triangle.angle_BAD}°, find ∠ADB"
                answer = triangle.angle_ADB
                given = ["angle_B", "angle_BAD"]
                hidden = ["angle_ADB"]
                angles_to_show = {
                    "B": triangle.angle_ABD,
                    "BAD": triangle.angle_BAD
                }
            
            structure = triangle.to_dict()
            if angles_to_show:
                structure["angles_shown"] = angles_to_show

        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        structure["problem_type"] = problem_type
        structure["goal"] = goal
        structure["answer"] = answer
        structure["given"] = given
        structure["hidden"] = hidden
        
        # Apply domain randomization if enabled
        # Apply domain randomization if enabled
        if self.randomize_labels:
            style = random.choice(["exotic", "random"])
            
            if isinstance(triangle, (RightTriangleWithAltitude, TriangleWithCevian)):
                # Need 4 unique labels
                available = list(self.label_randomizer.VERTEX_LABELS_COMMON)
                random.shuffle(available)
                labels = available[:4]  # A, B, C, D
                
                # Relabel using a custom approach since relabel_triangle takes 3
                # Create a mapping
                mapping = {"A": labels[0], "B": labels[1], "C": labels[2], "D": labels[3]}
                
                # Update primitive
                triangle.A.label = labels[0]
                triangle.B.label = labels[1]
                triangle.C.label = labels[2]
                triangle.D.label = labels[3]
                
                # Update structure
                structure["vertices"] = {
                    labels[0]: structure["vertices"]["A"],
                    labels[1]: structure["vertices"]["B"],
                    labels[2]: structure["vertices"]["C"],
                    labels[3]: structure["vertices"]["D"]
                }
                
                # Update goal string
                for old, new in mapping.items():
                    goal = goal.replace(old, new)
                    # Handle angles like ∠BAD -> ∠L1L2L4
                    # This simple replace might be risky for substring matches, but labels are usually capitalized
                    # Better to be specific
                    
                # Update structure angles
                if "angles" in structure:
                    new_angles = {}
                    for k, v in structure["angles"].items():
                        new_key = k
                        for char in k:
                            if char in mapping:
                                new_key = new_key.replace(char, mapping[char])
                        new_angles[new_key] = v
                    structure["angles"] = new_angles
                
                structure["goal"] = goal
                
            else:
                # Standard 3-point triangle
                labels = self.label_randomizer.get_triangle_labels(style)
                
                # Relabel the triangle primitives
                triangle = self.label_randomizer.relabel_triangle(triangle, labels)
                
                # Update the goal string
                mapping = {"A": labels[0], "B": labels[1], "C": labels[2]}
                for old, new in mapping.items():
                    goal = goal.replace(old, new)
                    goal = goal.replace(f"angle_{old}", f"angle_{new}")
                
                # Update structure
                structure["vertices"] = {
                    labels[0]: structure["vertices"]["A"],
                    labels[1]: structure["vertices"]["B"],
                    labels[2]: structure["vertices"]["C"]
                }
                
                # Update structure angles keys
                if "angles" in structure:
                    new_angles = {}
                    for k, v in structure["angles"].items():
                        new_k = mapping.get(k, k)
                        new_angles[new_k] = v
                    structure["angles"] = new_angles
                
                structure["goal"] = goal
        
        return GeneratedProblem(
            triangle=triangle,
            structure=structure,
            goal=goal,
            answer=answer
        )
    
    def generate_dataset(
        self,
        n_samples: int,
        output_dir: Path,
        problem_types: Optional[list[str]] = None,
        n_workers: int = 1
    ) -> list[dict]:
        """
        Generate a dataset of (image, structure) pairs.
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        structures_dir = output_dir / "structures"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        structures_dir.mkdir(parents=True, exist_ok=True)
        
        if problem_types is None:
            problem_types = [
                "find_hypotenuse", "find_leg", "find_area", 
                "find_angle", "find_side_with_angle",
                "find_isosceles_base", "find_isosceles_side",
                "find_equilateral_side", "find_equilateral_area"
            ]
        
        if n_workers > 1:
            # Parallel generation
            from multiprocessing import Pool
            from functools import partial
            
            worker_fn = partial(
                _generate_single_sample,
                images_dir=images_dir,
                structures_dir=structures_dir,
                problem_types=problem_types,
                min_side=self.min_side,
                max_side=self.max_side
            )
            
            with Pool(n_workers) as pool:
                metadata = []
                for i, result in enumerate(pool.imap(worker_fn, range(n_samples))):
                    metadata.append(result)
                    if (i + 1) % 500 == 0:
                        print(f"Generated {i + 1}/{n_samples} samples")
        else:
            # Sequential generation
            include_angles = any(pt in problem_types for pt in ["find_angle", "find_side_with_angle", "find_isosceles_base", "find_isosceles_side"])
            metadata = []
            for i in range(n_samples):
                problem_type = random.choice(problem_types)
                problem = self.generate_problem(problem_type, include_angles=include_angles)
                
                angles_to_show = problem.structure.get("angles_shown", None)
                
                # Save image
                image_path = images_dir / f"{i:05d}.png"
                self.renderer.render_triangle(
                    problem.triangle,
                    show_labels=True,
                    show_side_lengths=True,
                    angles_to_show=angles_to_show,
                    save_path=image_path
                )
                
                # Save structure
                structure_path = structures_dir / f"{i:05d}.json"
                with open(structure_path, "w") as f:
                    json.dump(problem.structure, f, indent=2)
                
                metadata.append({
                    "id": i,
                    "image": str(image_path),
                    "structure": str(structure_path),
                    "problem_type": problem_type,
                    "goal": problem.goal,
                    "answer": problem.answer
                })
                
                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{n_samples} samples")
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        return metadata


def _generate_single_sample(
    idx: int,
    images_dir: Path,
    structures_dir: Path,
    problem_types: list[str],
    min_side: float,
    max_side: float
) -> dict:
    """Worker function for parallel generation."""
    # Each worker creates its own generator and renderer
    from .renderer import DiagramRenderer
    
    generator = GeometryProblemGenerator(min_side=min_side, max_side=max_side)
    renderer = DiagramRenderer()
    
    include_angles = any(pt in problem_types for pt in ["find_angle", "find_side_with_angle", "find_isosceles_base", "find_isosceles_side"])
    
    problem_type = random.choice(problem_types)
    problem = generator.generate_problem(problem_type, include_angles=include_angles)
    
    angles_to_show = problem.structure.get("angles_shown", None)
    
    # Save image
    image_path = images_dir / f"{idx:05d}.png"
    renderer.render_triangle(
        problem.triangle,
        show_labels=True,
        show_side_lengths=True,
        angles_to_show=angles_to_show,
        save_path=image_path
    )
    
    # Save structure
    structure_path = structures_dir / f"{idx:05d}.json"
    with open(structure_path, "w") as f:
        json.dump(problem.structure, f, indent=2)
    
    return {
        "id": idx,
        "image": str(image_path),
        "structure": str(structure_path),
        "problem_type": problem_type,
        "goal": problem.goal,
        "answer": problem.answer
    }
