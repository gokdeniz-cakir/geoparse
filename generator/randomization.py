"""
Domain randomization utilities for realistic geometry problem generation.
"""

import random
from typing import Optional, List, Tuple

# Vertex label pools
VERTEX_LABELS_STANDARD = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VERTEX_LABELS_COMMON = ["A", "B", "C", "D", "E", "P", "Q", "R", "K", "L", "M", "N"]

# Greek symbols for unknown angles
GREEK_ANGLE_SYMBOLS = ["α", "β", "γ", "θ", "φ", "x", "y"]

# Variable symbols for unknown sides
VARIABLE_SIDE_SYMBOLS = ["x", "y", "a", "b", "c", "m", "n"]


class LabelRandomizer:
    """
    Generates random but consistent labels for geometric shapes.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.VERTEX_LABELS_COMMON = VERTEX_LABELS_COMMON
        if seed is not None:
            random.seed(seed)
    
    def get_triangle_labels(self, style: str = "random") -> Tuple[str, str, str]:
        """
        Get three vertex labels for a triangle.
        
        Args:
            style: "standard" (A,B,C), "random" (random 3 letters), "exotic" (K,L,M etc)
        """
        if style == "standard":
            return ("A", "B", "C")
        elif style == "exotic":
            # Use less common letter sets
            sets = [
                ("K", "L", "M"),
                ("P", "Q", "R"),
                ("D", "E", "F"),
                ("X", "Y", "Z"),
                ("M", "N", "O"),
            ]
            return random.choice(sets)
        else:  # random
            if random.random() < 0.5:
                return ("A", "B", "C")  # 50% chance of standard
            else:
                available = list(VERTEX_LABELS_COMMON)
                random.shuffle(available)
                return (available[0], available[1], available[2])
    
    def get_angle_symbol(self) -> str:
        """Get a random symbol for an unknown angle."""
        return random.choice(GREEK_ANGLE_SYMBOLS)
    
    def get_side_variable(self) -> str:
        """Get a random variable for an unknown side."""
        return random.choice(VARIABLE_SIDE_SYMBOLS)
    
    def relabel_triangle(self, triangle, labels: Tuple[str, str, str]):
        """
        Create a copy of a triangle with new vertex labels.
        """
        from .primitives import Point
        
        old_A, old_B, old_C = triangle.A, triangle.B, triangle.C
        new_A = Point(old_A.x, old_A.y, labels[0])
        new_B = Point(old_B.x, old_B.y, labels[1])
        new_C = Point(old_C.x, old_C.y, labels[2])
        
        # Return new triangle of same type with new points
        triangle_type = type(triangle)
        
        # Get all attributes except A, B, C
        kwargs = {k: v for k, v in triangle.__dict__.items() 
                  if k not in ['A', 'B', 'C']}
        
        return triangle_type(A=new_A, B=new_B, C=new_C, **kwargs)


def format_sqrt(value: float) -> str:
    """
    Format a number as a simplified square root if appropriate.
    E.g., 7.071... -> "5√2" or "√50"
    """
    import math
    
    # If it's close to an integer, return as integer
    if abs(value - round(value)) < 0.01:
        return str(int(round(value)))
    
    # Check if value² is close to an integer (for √N form)
    value_sq = value ** 2
    if abs(value_sq - round(value_sq)) < 0.1:
        return f"√{int(round(value_sq))}"
    
    # Otherwise return decimal
    return f"{value:.2f}"
