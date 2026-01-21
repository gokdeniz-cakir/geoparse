"""
Geometric primitives for diagram generation.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import math


@dataclass
class Point:
    """A 2D point with an optional label."""
    x: float
    y: float
    label: Optional[str] = None
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def distance_to(self, other: "Point") -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Segment:
    """A line segment between two points."""
    start: Point
    end: Point
    length: Optional[float] = None  # Labeled length (may differ from actual for display)
    show_length: bool = False
    
    def actual_length(self) -> float:
        return self.start.distance_to(self.end)
    
    def midpoint(self) -> Point:
        return Point(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2
        )


@dataclass
class Angle:
    """An angle at a vertex, defined by three points."""
    vertex: Point
    point1: Point  # One arm endpoint
    point2: Point  # Other arm endpoint
    measure: Optional[float] = None  # Degrees
    show_measure: bool = False
    is_right: bool = False  # Draw right angle marker


@dataclass
class RightTriangle:
    """
    A right triangle with the right angle at vertex C.
    
    Vertices:
      A - one end of hypotenuse
      B - other end of hypotenuse  
      C - right angle vertex
    """
    A: Point
    B: Point
    C: Point
    
    # Side lengths (can be set for labeling)
    side_a: Optional[float] = None  # BC (opposite to A)
    side_b: Optional[float] = None  # AC (opposite to B)
    side_c: Optional[float] = None  # AB (hypotenuse, opposite to C)
    
    def get_segments(self) -> list[Segment]:
        """Return the three sides as segments."""
        return [
            Segment(self.A, self.B, self.side_c),  # Hypotenuse
            Segment(self.B, self.C, self.side_a),  # Side a
            Segment(self.C, self.A, self.side_b),  # Side b
        ]
    
    def get_right_angle(self) -> Angle:
        """Return the right angle at C."""
        return Angle(
            vertex=self.C,
            point1=self.A,
            point2=self.B,
            measure=90,
            is_right=True
        )
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable structure."""
        return {
            "type": "right_triangle",
            "vertices": {
                "A": [round(self.A.x, 4), round(self.A.y, 4)],
                "B": [round(self.B.x, 4), round(self.B.y, 4)],
                "C": [round(self.C.x, 4), round(self.C.y, 4)],
            },
            "sides": {
                "AB": self.side_c,  # Hypotenuse
                "BC": self.side_a,
                "CA": self.side_b,
            },
            "right_angle_at": "C"
        }


@dataclass
class RightTriangleWithAltitude:
    """
    A right triangle with the right angle at vertex A, 
    and an altitude from A to hypotenuse BC.
    
    Vertices:
      A - right angle vertex (top)
      B - left base vertex
      C - right base vertex
      D - foot of altitude on BC
      
    Geometric Mean Relationships:
      AD² = BD × DC
      AB² = BD × BC
      AC² = DC × BC
    """
    A: Point
    B: Point
    C: Point
    D: Point  # Altitude foot on BC
    
    # Segment lengths
    AB: Optional[float] = None
    AC: Optional[float] = None
    BC: Optional[float] = None  # Hypotenuse
    AD: Optional[float] = None  # Altitude
    BD: Optional[float] = None
    DC: Optional[float] = None
    
    @classmethod
    def from_legs(cls, leg_AB: float, leg_AC: float, rotation: float = 0) -> "RightTriangleWithAltitude":
        """
        Create a right triangle with altitude from leg lengths.
        Right angle at A (top), B at left, C at right.
        """
        import math
        
        # Hypotenuse
        BC = math.sqrt(leg_AB**2 + leg_AC**2)
        
        # Altitude from A to BC using: Area = (1/2)*BC*AD = (1/2)*AB*AC
        AD = (leg_AB * leg_AC) / BC
        
        # BD and DC using similar triangles
        BD = (leg_AB**2) / BC
        DC = (leg_AC**2) / BC
        
        # Position vertices: B at origin, C on x-axis
        B = Point(0, 0, "B")
        C = Point(BC, 0, "C")
        D = Point(BD, 0, "D")
        A = Point(BD, AD, "A")
        
        # Apply rotation around center
        center_x, center_y = BC / 2, AD / 3
        angle_rad = math.radians(rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        def rotate(p: Point) -> Point:
            dx, dy = p.x - center_x, p.y - center_y
            new_x = center_x + dx * cos_a - dy * sin_a
            new_y = center_y + dx * sin_a + dy * cos_a
            return Point(new_x, new_y, p.label)
        
        A, B, C, D = rotate(A), rotate(B), rotate(C), rotate(D)
        
        return cls(
            A=A, B=B, C=C, D=D,
            AB=round(leg_AB, 2),
            AC=round(leg_AC, 2),
            BC=round(BC, 2),
            AD=round(AD, 2),
            BD=round(BD, 2),
            DC=round(DC, 2)
        )
    
    def get_segments(self) -> list[Segment]:
        """Return all segments including altitude."""
        return [
            Segment(self.A, self.B, self.AB),
            Segment(self.B, self.C, self.BC),
            Segment(self.C, self.A, self.AC),
            Segment(self.A, self.D, self.AD),  # Altitude
        ]
    
    def to_dict(self) -> dict:
        return {
            "type": "right_triangle_with_altitude",
            "vertices": {
                "A": [round(self.A.x, 4), round(self.A.y, 4)],
                "B": [round(self.B.x, 4), round(self.B.y, 4)],
                "C": [round(self.C.x, 4), round(self.C.y, 4)],
                "D": [round(self.D.x, 4), round(self.D.y, 4)],
            },
            "sides": {
                "AB": self.AB,
                "AC": self.AC,
                "BC": self.BC,
                "AD": self.AD,
                "BD": self.BD,
                "DC": self.DC,
            },
            "right_angle_at": "A",
            "altitude_foot": "D"
        }


@dataclass
class IsoscelesTriangle:
    """
    An isosceles triangle where AB = AC.
    Base is BC.
    """
    A: Point  # Top vertex
    B: Point  # Base vertex 1
    C: Point  # Base vertex 2
    
    side_equal: Optional[float] = None  # Length of AB and AC
    side_base: Optional[float] = None   # Length of BC (base)
    
    def get_segments(self) -> list[Segment]:
        return [
            Segment(self.A, self.B, self.side_equal),
            Segment(self.B, self.C, self.side_base),
            Segment(self.C, self.A, self.side_equal),
        ]
        
    def to_dict(self) -> dict:
        return {
            "type": "isosceles_triangle",
            "vertices": {
                "A": [round(self.A.x, 4), round(self.A.y, 4)],
                "B": [round(self.B.x, 4), round(self.B.y, 4)],
                "C": [round(self.C.x, 4), round(self.C.y, 4)],
            },
            "sides": {
                "AB": self.side_equal,
                "BC": self.side_base,
                "CA": self.side_equal,
            },
            "equal_sides": ["AB", "CA"]
        }


@dataclass
class EquilateralTriangle:
    """
    An equilateral triangle where AB = BC = CA.
    """
    A: Point
    B: Point
    C: Point
    
    side_length: Optional[float] = None
    
    def get_segments(self) -> list[Segment]:
        return [
            Segment(self.A, self.B, self.side_length),
            Segment(self.B, self.C, self.side_length),
            Segment(self.C, self.A, self.side_length),
        ]
        
    def to_dict(self) -> dict:
        return {
            "type": "equilateral_triangle",
            "vertices": {
                "A": [round(self.A.x, 4), round(self.A.y, 4)],
                "B": [round(self.B.x, 4), round(self.B.y, 4)],
                "C": [round(self.C.x, 4), round(self.C.y, 4)],
            },
            "sides": {
                "AB": self.side_length,
                "BC": self.side_length,
                "CA": self.side_length,
            },
            "equal_sides": ["AB", "BC", "CA"]
        }


@dataclass
class TriangleWithCevian:
    """
    A triangle with a cevian (line from vertex A to point D on BC).
    Creates sub-triangles ABD and ACD.
    
    Vertices:
      A - apex
      B - base left
      C - base right
      D - point on BC (cevian foot)
    """
    A: Point
    B: Point
    C: Point
    D: Point  # On segment BC
    
    # Angles in degrees
    angle_BAD: Optional[float] = None  # Angle at A in triangle ABD
    angle_DAC: Optional[float] = None  # Angle at A in triangle ACD
    angle_ABD: Optional[float] = None  # Angle at B
    angle_ACD: Optional[float] = None  # Angle at C
    angle_ADB: Optional[float] = None  # Angle at D (in ABD)
    angle_ADC: Optional[float] = None  # Angle at D (in ACD) = 180 - angle_ADB
    
    @classmethod
    def from_angles(cls, angle_B: float, angle_C: float, cevian_ratio: float = 0.5, rotation: float = 0):
        """
        Create triangle with cevian from apex through D on base.
        
        Args:
            angle_B: Angle at B in degrees
            angle_C: Angle at C in degrees
            cevian_ratio: Position of D on BC (0=B, 1=C)
            rotation: Rotation of whole figure
        """
        import math
        
        angle_A = 180 - angle_B - angle_C
        
        # Place B at origin, C on x-axis
        BC = 5  # Arbitrary base length
        B = Point(0, 0, "B")
        C = Point(BC, 0, "C")
        
        # D is on BC
        D = Point(BC * cevian_ratio, 0, "D")
        
        # Find A using angles
        # From B, A is at angle (180 - angle_B) from x-axis
        angle_from_B = math.radians(180 - angle_B)
        # From C, A is at angle angle_C from negative x-axis
        angle_from_C = math.radians(180 - angle_C)
        
        # Solve for A intersection
        # Line from B: y = x * tan(angle_from_B)
        # Line from C: y = (x - BC) * tan(pi - angle_from_C)
        
        tan_B = math.tan(angle_from_B)
        tan_C = math.tan(math.pi - angle_from_C)
        
        # x * tan_B = (x - BC) * tan_C
        # x * tan_B = x * tan_C - BC * tan_C
        # x * (tan_B - tan_C) = -BC * tan_C
        if abs(tan_B - tan_C) > 0.001:
            A_x = -BC * tan_C / (tan_B - tan_C)
            A_y = A_x * tan_B
        else:
            A_x, A_y = BC / 2, BC  # Fallback
        
        A = Point(A_x, A_y, "A")
        
        # Calculate sub-angles
        # angle_BAD and angle_DAC from A to D
        def angle_between(p1, vertex, p2):
            v1 = (p1.x - vertex.x, p1.y - vertex.y)
            v2 = (p2.x - vertex.x, p2.y - vertex.y)
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if mag1 * mag2 == 0:
                return 0
            cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
            return math.degrees(math.acos(cos_angle))
        
        angle_BAD = round(angle_between(B, A, D), 1)
        angle_DAC = round(angle_between(D, A, C), 1)
        angle_ADB = round(angle_between(A, D, B), 1)
        angle_ADC = round(180 - angle_ADB, 1)
        
        # Apply rotation
        if rotation != 0:
            center = Point((A.x + B.x + C.x)/3, (A.y + B.y + C.y)/3)
            angle_rad = math.radians(rotation)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            
            def rotate(p):
                dx, dy = p.x - center.x, p.y - center.y
                new_x = center.x + dx * cos_a - dy * sin_a
                new_y = center.y + dx * sin_a + dy * cos_a
                return Point(new_x, new_y, p.label)
            
            A, B, C, D = rotate(A), rotate(B), rotate(C), rotate(D)
        
        return cls(
            A=A, B=B, C=C, D=D,
            angle_BAD=angle_BAD,
            angle_DAC=angle_DAC,
            angle_ABD=round(angle_B, 1),
            angle_ACD=round(angle_C, 1),
            angle_ADB=angle_ADB,
            angle_ADC=angle_ADC
        )
    
    def to_dict(self) -> dict:
        return {
            "type": "triangle_with_cevian",
            "vertices": {
                "A": [round(self.A.x, 4), round(self.A.y, 4)],
                "B": [round(self.B.x, 4), round(self.B.y, 4)],
                "C": [round(self.C.x, 4), round(self.C.y, 4)],
                "D": [round(self.D.x, 4), round(self.D.y, 4)],
            },
            "angles": {
                "BAD": self.angle_BAD,
                "DAC": self.angle_DAC,
                "ABD": self.angle_ABD,
                "ACD": self.angle_ACD,
                "ADB": self.angle_ADB,
                "ADC": self.angle_ADC,
            },
            "cevian": ["A", "D"]
        }

