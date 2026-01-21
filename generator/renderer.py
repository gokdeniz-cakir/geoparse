"""
Matplotlib-based diagram renderer.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Optional
import numpy as np

from .primitives import Point, Segment, Angle, RightTriangle, IsoscelesTriangle, EquilateralTriangle, RightTriangleWithAltitude, TriangleWithCevian


class DiagramRenderer:
    """Renders geometry diagrams using matplotlib."""
    
    def __init__(
        self,
        figsize: tuple[float, float] = (6, 6),
        dpi: int = 100,
        style: str = "clean"
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Style settings
        self.line_color = "black"
        self.line_width = 2
        self.label_fontsize = 14
        self.label_offset = 0.15
        self.right_angle_size = 0.3
        
    def render_right_triangle(
        self,
        triangle: RightTriangle,
        show_labels: bool = True,
        show_right_angle: bool = True,
        show_side_lengths: bool = False,
        angles_to_show: Optional[dict] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Render a right triangle diagram.
        
        Args:
            triangle: The RightTriangle to render
            show_labels: Whether to show vertex labels (A, B, C)
            show_right_angle: Whether to show the right angle marker at C
            show_side_lengths: Whether to show side length labels
            angles_to_show: Dict mapping vertex names to angles, e.g. {"A": 30, "B": 60}
            save_path: If provided, save the figure to this path
            
        Returns:
            The matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Get vertices
        A, B, C = triangle.A, triangle.B, triangle.C
        vertices = {"A": A, "B": B, "C": C}
        
        # Draw triangle edges
        xs = [A.x, B.x, C.x, A.x]
        ys = [A.y, B.y, C.y, A.y]
        ax.plot(xs, ys, color=self.line_color, linewidth=self.line_width)
        
        # Draw right angle marker at C
        if show_right_angle:
            self._draw_right_angle(ax, C, A, B)
        
        # Draw angle arcs at other vertices
        if angles_to_show:
            for vertex_name, angle_deg in angles_to_show.items():
                v = vertices[vertex_name]
                # Get the two other vertices to form the angle arms
                other_vertices = [vertices[n] for n in ["A", "B", "C"] if n != vertex_name]
                self._draw_angle_arc(ax, v, other_vertices[0], other_vertices[1], angle_deg)
        
        # Draw vertex labels
        if show_labels:
            self._draw_vertex_label(ax, A, [B, C])
            self._draw_vertex_label(ax, B, [A, C])
            self._draw_vertex_label(ax, C, [A, B])
        
        # Draw side lengths (positioned outside the triangle)
        if show_side_lengths:
            # Calculate centroid for outward positioning
            centroid = Point(
                (A.x + B.x + C.x) / 3,
                (A.y + B.y + C.y) / 3
            )
            segments = triangle.get_segments()
            for seg in segments:
                if seg.length is not None:
                    self._draw_length_label(ax, seg, centroid)
        
        # Clean up axes
        ax.set_aspect("equal")
        ax.axis("off")
        
        # Add padding
        self._set_bounds(ax, [A, B, C], padding=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
        
        return fig
    
    def _draw_right_angle(
        self,
        ax: plt.Axes,
        vertex: Point,
        p1: Point,
        p2: Point
    ):
        """Draw a right angle marker (small square) at vertex."""
        # Get unit vectors along each arm
        v1 = np.array([p1.x - vertex.x, p1.y - vertex.y])
        v2 = np.array([p2.x - vertex.x, p2.y - vertex.y])
        
        v1 = v1 / np.linalg.norm(v1) * self.right_angle_size
        v2 = v2 / np.linalg.norm(v2) * self.right_angle_size
        
        # Corner points of the square
        corner1 = (vertex.x + v1[0], vertex.y + v1[1])
        corner2 = (vertex.x + v1[0] + v2[0], vertex.y + v1[1] + v2[1])
        corner3 = (vertex.x + v2[0], vertex.y + v2[1])
        
        # Draw the square
        xs = [corner1[0], corner2[0], corner3[0]]
        ys = [corner1[1], corner2[1], corner3[1]]
        ax.plot(xs, ys, color=self.line_color, linewidth=self.line_width * 0.7)
    
    def _draw_vertex_label(
        self,
        ax: plt.Axes,
        vertex: Point,
        other_points: list[Point]
    ):
        """Draw a label for a vertex, positioned away from other points."""
        # Calculate direction away from centroid of other points
        centroid_x = sum(p.x for p in other_points) / len(other_points)
        centroid_y = sum(p.y for p in other_points) / len(other_points)
        
        dx = vertex.x - centroid_x
        dy = vertex.y - centroid_y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            dx, dy = dx / dist, dy / dist
        else:
            dx, dy = 0, 1
        
        # Larger offset to prevent collision with side labels
        label_x = vertex.x + dx * self.label_offset * 3.5
        label_y = vertex.y + dy * self.label_offset * 3.5
        
        ax.text(
            label_x, label_y, vertex.label or "",
            fontsize=self.label_fontsize,
            ha="center", va="center",
            fontweight="bold"
        )
    
    def _draw_length_label(self, ax: plt.Axes, segment: Segment, centroid: Point):
        """Draw a length label at the midpoint of a segment, positioned outside the triangle."""
        mid = segment.midpoint()
        
        # Offset perpendicular to the segment
        dx = segment.end.x - segment.start.x
        dy = segment.end.y - segment.start.y
        length = np.sqrt(dx**2 + dy**2)
        
        # Skip labels on very short segments to avoid clutter
        if length < 0.5:
            return
        
        # Perpendicular unit vector
        perp_x, perp_y = -dy / length, dx / length
        
        # Choose the direction that points AWAY from the centroid (outside the triangle)
        # Test which perpendicular direction is further from centroid
        test_x1, test_y1 = mid.x + perp_x, mid.y + perp_y
        test_x2, test_y2 = mid.x - perp_x, mid.y - perp_y
        
        dist1 = (test_x1 - centroid.x)**2 + (test_y1 - centroid.y)**2
        dist2 = (test_x2 - centroid.x)**2 + (test_y2 - centroid.y)**2
        
        # Use the direction that's further from centroid (i.e., outside)
        if dist2 > dist1:
            perp_x, perp_y = -perp_x, -perp_y
        
        # Scale offset based on label length - longer labels need more buffer
        # Use minimum offset of 0.4 to ensure labels don't overlap segment
        label_str = str(segment.length)
        length_factor = 1 + (len(label_str) - 2) * 0.15
        offset = max(0.4, self.label_offset * 2.5 * length_factor)
        
        label_x = mid.x + perp_x * offset
        label_y = mid.y + perp_y * offset
        
        ax.text(
            label_x, label_y, str(segment.length),
            fontsize=self.label_fontsize - 2,
            ha="center", va="center"
        )
    
    def render_triangle(
        self,
        triangle,
        show_labels: bool = True,
        show_side_lengths: bool = False,
        angles_to_show: Optional[dict] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Render any triangle (Right, Isosceles, Equilateral).
        
        Args:
            triangle: The triangle object (RightTriangle, IsoscelesTriangle, EquilateralTriangle)
            show_labels: Whether to show vertex labels
            show_side_lengths: Whether to show side length labels
            angles_to_show: Dict of angles to display
            save_path: Save path
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        A, B, C = triangle.A, triangle.B, triangle.C
        vertices = {"A": A, "B": B, "C": C}
        
        # Draw edges
        xs = [A.x, B.x, C.x, A.x]
        ys = [A.y, B.y, C.y, A.y]
        ax.plot(xs, ys, color=self.line_color, linewidth=self.line_width)
        
        # Type-specific markers
        if hasattr(triangle, "get_right_angle") and triangle.get_right_angle().is_right:
            # Right Triangle: Draw right angle at C
            self._draw_right_angle(ax, C, A, B)
            
        elif isinstance(triangle, IsoscelesTriangle):
             # Isosceles: Draw ticks on AB and AC
             self._draw_tick_mark(ax, Segment(A, B), count=1)
             self._draw_tick_mark(ax, Segment(A, C), count=1)
             
        elif isinstance(triangle, EquilateralTriangle):
            # Equilateral: Draw ticks on all sides
            self._draw_tick_mark(ax, Segment(A, B), count=1)
            self._draw_tick_mark(ax, Segment(B, C), count=1)
            self._draw_tick_mark(ax, Segment(C, A), count=1)

        # Draw angle arcs
        if angles_to_show:
            for vertex_name, angle_deg in angles_to_show.items():
                v = vertices[vertex_name]
                other_vertices = [vertices[n] for n in ["A", "B", "C"] if n != vertex_name]
                self._draw_angle_arc(ax, v, other_vertices[0], other_vertices[1], angle_deg)
        
        # Draw vertex labels
        if show_labels:
            self._draw_vertex_label(ax, A, [B, C])
            self._draw_vertex_label(ax, B, [A, C])
            self._draw_vertex_label(ax, C, [A, B])
            
        # Draw side lengths
        if show_side_lengths:
            centroid = Point((A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3)
            segments = triangle.get_segments()
            for seg in segments:
                if seg.length is not None:
                    self._draw_length_label(ax, seg, centroid)
        
        # Clean up
        ax.set_aspect("equal")
        ax.axis("off")
        self._set_bounds(ax, [A, B, C], padding=0.5)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            
        return fig

    def render_external_angle(
        self,
        triangle,
        extend_at: str = "B",
        extend_from: str = "C",
        external_angle: float = None,
        interior_angles: dict = None,
        show_labels: bool = True,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Render a triangle with an external angle (line extension beyond a vertex).
        
        Args:
            triangle: Triangle to render
            extend_at: Vertex where line extends beyond (e.g., "B")
            extend_from: Vertex the extension comes from (e.g., "C" extends CB beyond B)
            external_angle: The exterior angle in degrees
            interior_angles: Dict of interior angles to show
            show_labels: Whether to show vertex labels
            save_path: Save path
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        A, B, C = triangle.A, triangle.B, triangle.C
        vertices = {"A": A, "B": B, "C": C}
        
        # Draw main triangle edges
        xs = [A.x, B.x, C.x, A.x]
        ys = [A.y, B.y, C.y, A.y]
        ax.plot(xs, ys, color=self.line_color, linewidth=self.line_width)
        
        # Draw the line extension
        ext_vertex = vertices[extend_at]
        from_vertex = vertices[extend_from]
        
        # Direction from from_vertex to ext_vertex
        dx = ext_vertex.x - from_vertex.x
        dy = ext_vertex.y - from_vertex.y
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx / length, dy / length
        
        # Extend by 1.5 units beyond the vertex
        ext_length = 1.5
        ext_x = ext_vertex.x + dx * ext_length
        ext_y = ext_vertex.y + dy * ext_length
        ext_point = Point(ext_x, ext_y)
        
        # Draw extension line
        ax.plot(
            [ext_vertex.x, ext_x], [ext_vertex.y, ext_y],
            color=self.line_color, linewidth=self.line_width
        )
        
        # Draw external angle arc (between extension and third vertex)
        # Third vertex is the one that's not extend_at or extend_from
        third_name = [n for n in ["A", "B", "C"] if n not in [extend_at, extend_from]][0]
        third_vertex = vertices[third_name]
        
        if external_angle:
            self._draw_angle_arc(ax, ext_vertex, ext_point, third_vertex, external_angle, show_label=True)
        
        # Draw interior angles
        if interior_angles:
            for vertex_name, angle_deg in interior_angles.items():
                v = vertices[vertex_name]
                other_vertices = [vertices[n] for n in ["A", "B", "C"] if n != vertex_name]
                self._draw_angle_arc(ax, v, other_vertices[0], other_vertices[1], angle_deg)
        
        # Draw vertex labels
        if show_labels:
            all_points = [A, B, C, ext_point]
            self._draw_vertex_label(ax, A, all_points)
            self._draw_vertex_label(ax, B, all_points)
            self._draw_vertex_label(ax, C, all_points)
        
        # Set bounds including extension
        ax.set_aspect("equal")
        ax.axis("off")
        self._set_bounds(ax, [A, B, C, ext_point], padding=0.5)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            
        return fig

    def render_cevian_triangle(
        self,
        triangle: TriangleWithCevian,
        angles_to_show: dict = None,
        show_labels: bool = True,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Render a triangle with a cevian from A to D on BC.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        A, B, C, D = triangle.A, triangle.B, triangle.C, triangle.D
        
        # Draw main triangle edges
        for start, end in [(A, B), (B, C), (C, A)]:
            ax.plot(
                [start.x, end.x], [start.y, end.y],
                color=self.line_color, linewidth=self.line_width
            )
        
        # Draw cevian AD
        ax.plot(
            [A.x, D.x], [A.y, D.y],
            color=self.line_color, linewidth=self.line_width
        )
        
        # Draw angle arcs
        if angles_to_show:
            vertices = {"A": A, "B": B, "C": C, "D": D}
            for angle_name, angle_deg in angles_to_show.items():
                if angle_name == "BAD":
                    self._draw_angle_arc(ax, A, B, D, angle_deg)
                elif angle_name == "DAC":
                    self._draw_angle_arc(ax, A, D, C, angle_deg)
                elif angle_name == "ABD" or angle_name == "B":
                    self._draw_angle_arc(ax, B, A, D, angle_deg)
                elif angle_name == "ACD" or angle_name == "C":
                    self._draw_angle_arc(ax, C, A, D, angle_deg)
                elif angle_name == "ADB":
                    self._draw_angle_arc(ax, D, A, B, angle_deg)
                elif angle_name == "ADC":
                    self._draw_angle_arc(ax, D, A, C, angle_deg)
        
        # Draw vertex labels
        if show_labels:
            for vertex in [A, B, C, D]:
                self._draw_vertex_label(ax, vertex, [A, B, C, D])
        
        ax.set_aspect("equal")
        ax.axis("off")
        self._set_bounds(ax, [A, B, C, D], padding=0.5)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            
        return fig

    def _draw_tick_mark(self, ax: plt.Axes, segment: Segment, count: int = 1):
        """Draw tick mark(s) on a segment to indicate equality."""
        mid = segment.midpoint()
        
        # Vector along segment
        dx = segment.end.x - segment.start.x
        dy = segment.end.y - segment.start.y
        length = np.sqrt(dx**2 + dy**2)
        if length == 0: return
        
        # Unit vector along segment
        ux, uy = dx / length, dy / length
        
        # Perpendicular unit vector
        px, py = -uy, ux
        
        tick_size = 0.2
        spacing = 0.15
        
        # Draw ticks
        for i in range(count):
            # Offset along the segment for multiple ticks
            offset = 0
            if count > 1:
                offset = (i - (count - 1) / 2) * spacing
                
            center_x = mid.x + ux * offset
            center_y = mid.y + uy * offset
            
            x1 = center_x + px * tick_size / 2
            y1 = center_y + py * tick_size / 2
            x2 = center_x - px * tick_size / 2
            y2 = center_y - py * tick_size / 2
            
            ax.plot([x1, x2], [y1, y2], color=self.line_color, linewidth=self.line_width)

    def render_altitude_triangle(
        self,
        triangle: RightTriangleWithAltitude,
        show_labels: bool = True,
        show_side_lengths: bool = True,
        hidden_sides: list[str] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Render a right triangle with altitude from A to hypotenuse BC.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_aspect("equal")
        ax.axis("off")
        
        A, B, C, D = triangle.A, triangle.B, triangle.C, triangle.D
        
        # Draw main triangle edges
        for start, end in [(A, B), (B, C), (C, A)]:
            ax.plot(
                [start.x, end.x], [start.y, end.y],
                color=self.line_color, linewidth=self.line_width
            )
        
        # Draw altitude line (A to D)
        ax.plot(
            [A.x, D.x], [A.y, D.y],
            color=self.line_color, linewidth=self.line_width
        )
        
        # Draw right angle marker at A
        self._draw_right_angle(ax, A, B, C)
        
        # Draw right angle marker at D (altitude is perpendicular to BC)
        self._draw_right_angle(ax, D, A, C)
        
        # Vertex labels with extra offset for D since it's inside the triangle
        if show_labels:
            for vertex in [A, B, C]:
                self._draw_vertex_label(ax, vertex, [A, B, C, D])
            # D needs special handling - label below the base line
            # Direction from A to D
            d_dx = D.x - A.x
            d_dy = D.y - A.y
            d_len = np.sqrt(d_dx**2 + d_dy**2)
            if d_len > 0:
                d_dx, d_dy = d_dx / d_len, d_dy / d_len
            # Label D in the direction away from A
            ax.text(
                D.x + d_dx * self.label_offset * 2.5,
                D.y + d_dy * self.label_offset * 2.5,
                "D", fontsize=self.label_fontsize,
                ha="center", va="center", fontweight="bold"
            )
        
        # Side length labels with smarter positioning
        if show_side_lengths:
            hidden = hidden_sides or []
            
            # For outer triangle sides (AB, AC), use the main triangle centroid
            outer_centroid = Point(
                (A.x + B.x + C.x) / 3,
                (A.y + B.y + C.y) / 3
            )
            
            # Label outer sides (away from triangle center)
            if "AB" not in hidden and triangle.AB:
                self._draw_length_label(ax, Segment(A, B, triangle.AB), outer_centroid)
            if "AC" not in hidden and triangle.AC:
                self._draw_length_label(ax, Segment(A, C, triangle.AC), outer_centroid)
            
            # For BD and DC, label below the base (opposite from A)
            if "BD" not in hidden and triangle.BD:
                seg = Segment(B, D, triangle.BD)
                mid = seg.midpoint()
                # Push labels away from A (below the base line)
                dir_from_A_x = mid.x - A.x
                dir_from_A_y = mid.y - A.y
                dist = np.sqrt(dir_from_A_x**2 + dir_from_A_y**2)
                if dist > 0:
                    dir_from_A_x /= dist
                    dir_from_A_y /= dist
                ax.text(
                    mid.x + dir_from_A_x * 0.5,
                    mid.y + dir_from_A_y * 0.5,
                    str(triangle.BD), fontsize=self.label_fontsize - 2,
                    ha="center", va="center"
                )
            if "DC" not in hidden and triangle.DC:
                seg = Segment(D, C, triangle.DC)
                mid = seg.midpoint()
                dir_from_A_x = mid.x - A.x
                dir_from_A_y = mid.y - A.y
                dist = np.sqrt(dir_from_A_x**2 + dir_from_A_y**2)
                if dist > 0:
                    dir_from_A_x /= dist
                    dir_from_A_y /= dist
                ax.text(
                    mid.x + dir_from_A_x * 0.5,
                    mid.y + dir_from_A_y * 0.5,
                    str(triangle.DC), fontsize=self.label_fontsize - 2,
                    ha="center", va="center"
                )
            
            # For AD (altitude), label to the side - toward B
            if "AD" not in hidden and triangle.AD:
                seg = Segment(A, D, triangle.AD)
                mid = seg.midpoint()
                # Push toward B
                dir_to_B_x = B.x - mid.x
                dir_to_B_y = B.y - mid.y
                dist = np.sqrt(dir_to_B_x**2 + dir_to_B_y**2)
                if dist > 0:
                    dir_to_B_x /= dist
                    dir_to_B_y /= dist
                ax.text(
                    mid.x + dir_to_B_x * 0.4,
                    mid.y + dir_to_B_y * 0.4,
                    str(triangle.AD), fontsize=self.label_fontsize - 2,
                    ha="center", va="center"
                )
        
        # Set bounds
        self._set_bounds(ax, [A, B, C, D])
        
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            
        return fig

    def _draw_angle_arc(
        self,
        ax: plt.Axes,
        vertex: Point,
        p1: Point,
        p2: Point,
        angle_deg: float,
        show_label: bool = True,
        arc_radius: float = 0.4
    ):
        """
        Draw an angle arc at a vertex with optional degree label.
        
        Args:
            vertex: The vertex point where the angle is
            p1: One arm endpoint
            p2: Other arm endpoint
            angle_deg: The angle measure in degrees
            show_label: Whether to show the degree label
            arc_radius: Radius of the arc
        """
        from matplotlib.patches import Arc
        
        # Calculate angles of each arm relative to positive x-axis
        angle1 = np.degrees(np.arctan2(p1.y - vertex.y, p1.x - vertex.x))
        angle2 = np.degrees(np.arctan2(p2.y - vertex.y, p2.x - vertex.x))
        
        # Normalize angles to [0, 360)
        angle1 = angle1 % 360
        angle2 = angle2 % 360
        
        # Determine start and end angles (arc goes counterclockwise from theta1 to theta2)
        # We want to draw the smaller arc
        if abs(angle2 - angle1) > 180:
            if angle1 < angle2:
                angle1 += 360
            else:
                angle2 += 360
        
        theta1, theta2 = min(angle1, angle2), max(angle1, angle2)
        
        # Draw the arc
        arc = Arc(
            (vertex.x, vertex.y),
            2 * arc_radius, 2 * arc_radius,
            angle=0,
            theta1=theta1,
            theta2=theta2,
            color=self.line_color,
            linewidth=self.line_width * 0.6
        )
        ax.add_patch(arc)
        
        # Add angle label
        if show_label:
            # Position label at midpoint of arc, slightly further out
            mid_angle = np.radians((theta1 + theta2) / 2)
            label_radius = arc_radius * 1.6
            
            label_x = vertex.x + label_radius * np.cos(mid_angle)
            label_y = vertex.y + label_radius * np.sin(mid_angle)
            
            # Format angle (use degree symbol)
            label_text = f"{int(angle_deg)}°"
            
            ax.text(
                label_x, label_y, label_text,
                fontsize=self.label_fontsize - 3,
                ha="center", va="center"
            )
    
    def _set_bounds(
        self,
        ax: plt.Axes,
        points: list[Point],
        padding: float = 0.5
    ):
        """Set axis bounds to fit all points with padding."""
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        
        x_min, x_max = min(xs) - padding, max(xs) + padding
        y_min, y_max = min(ys) - padding, max(ys) + padding
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
