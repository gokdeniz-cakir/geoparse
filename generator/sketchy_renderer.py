"""
Sketchy Diagram Renderer for Domain Randomization (Sim2Real).

This renderer extends DiagramRenderer with:
1. Hand-drawn/wobble line effects
2. Variable stroke widths
3. Font randomization
4. Background noise/textures
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from pathlib import Path
from typing import Optional
import random

from .renderer import DiagramRenderer
from .primitives import Point, Segment


class SketchyRenderer(DiagramRenderer):
    """Renders diagrams with randomized 'sketchy' style for domain robustness."""
    
    # Available styles for randomization
    BACKGROUND_COLORS = [
        "#FFFFFF",  # White
        "#FFFEF0",  # Cream
        "#F5F5DC",  # Beige
        "#E8E8E8",  # Light gray
        "#FFF8DC",  # Cornsilk
    ]
    
    FONT_FAMILIES = [
        "DejaVu Sans",
        "Liberation Sans",
        "FreeSans",
        "Arial",
        "Helvetica",
        "Times New Roman",
        "Courier New",
    ]
    
    LINE_COLORS = [
        "#000000",  # Black
        "#1A1A1A",  # Near black
        "#2B2B2B",  # Dark gray
        "#0000CD",  # Medium blue (pen)
        "#191970",  # Midnight blue
    ]
    
    def __init__(
        self,
        figsize: tuple[float, float] = (6, 6),
        dpi: int = 100,
        randomize: bool = True,
        wobble_strength: float = 0.03,
        use_xkcd: bool = False
    ):
        super().__init__(figsize=figsize, dpi=dpi, style="sketchy")
        self.randomize = randomize
        self.wobble_strength = wobble_strength
        self.use_xkcd = use_xkcd
        
        # Will be set per-render
        self._current_bg = None
        self._current_font = None
        
    def _randomize_style(self):
        """Randomize visual parameters for this render."""
        if not self.randomize:
            return
            
        # Line style
        self.line_color = random.choice(self.LINE_COLORS)
        self.line_width = random.uniform(1.5, 2.5)  # Reduced max width
        
        # Font
        self._current_font = random.choice(self.FONT_FAMILIES)
        self.label_fontsize = random.randint(12, 15)
        
        # Background
        self._current_bg = random.choice(self.BACKGROUND_COLORS)
        
        # Wobble - MUCH more subtle now
        self.wobble_strength = random.uniform(0.002, 0.008)
        
    def _wobble_line(self, x_start, y_start, x_end, y_end, num_points=20):
        """Generate a subtly wobbly line between two points."""
        t = np.linspace(0, 1, num_points)
        
        # Base line
        x = x_start + t * (x_end - x_start)
        y = y_start + t * (y_end - y_start)
        
        # Perpendicular direction for wobble
        dx = x_end - x_start
        dy = y_end - y_start
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
        else:
            perp_x, perp_y = 0, 0
            
        # Add SUBTLE noise perpendicular to the line
        wobble_scale = self.wobble_strength * length
        
        # Use smooth noise (cumulative random walk, then smooth)
        raw_noise = np.random.randn(num_points) * wobble_scale
        # Smooth with simple moving average
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        noise = np.convolve(raw_noise, kernel, mode='same')
        
        # Reduce noise at endpoints to keep vertices sharp
        noise *= np.sin(np.pi * t) ** 2
        
        x += perp_x * noise
        y += perp_y * noise
        
        return x, y
    
    def _draw_sketchy_line(self, ax, x1, y1, x2, y2):
        """Draw a single line with optional wobble effect."""
        if self.use_xkcd or self.wobble_strength > 0:
            xs, ys = self._wobble_line(x1, y1, x2, y2)
            ax.plot(xs, ys, color=self.line_color, linewidth=self.line_width,
                   solid_capstyle='round', solid_joinstyle='round')
        else:
            ax.plot([x1, x2], [y1, y2], color=self.line_color, 
                   linewidth=self.line_width)
    
    def render_triangle(
        self,
        triangle,
        show_labels: bool = True,
        show_side_lengths: bool = False,
        angles_to_show: Optional[dict] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Render a triangle with sketchy style.
        """
        # Randomize style for this render
        self._randomize_style()
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Apply background
        if self._current_bg:
            fig.patch.set_facecolor(self._current_bg)
            ax.set_facecolor(self._current_bg)
        
        A, B, C = triangle.A, triangle.B, triangle.C
        vertices = {"A": A, "B": B, "C": C}
        
        # Draw edges with wobble
        self._draw_sketchy_line(ax, A.x, A.y, B.x, B.y)
        self._draw_sketchy_line(ax, B.x, B.y, C.x, C.y)
        self._draw_sketchy_line(ax, C.x, C.y, A.x, A.y)
        
        # Draw angle arcs
        if angles_to_show:
            for vertex_name, angle_deg in angles_to_show.items():
                v = vertices[vertex_name]
                other_vertices = [vertices[n] for n in ["A", "B", "C"] if n != vertex_name]
                self._draw_angle_arc(ax, v, other_vertices[0], other_vertices[1], angle_deg)
        
        # Draw vertex labels with randomized font
        if show_labels:
            for vertex, others in [(A, [B, C]), (B, [A, C]), (C, [A, B])]:
                self._draw_sketchy_label(ax, vertex, others)
                
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
    
    def _draw_sketchy_label(self, ax, vertex: Point, other_points: list[Point]):
        """Draw label with randomized font."""
        # Calculate direction away from centroid
        centroid_x = sum(p.x for p in other_points) / len(other_points)
        centroid_y = sum(p.y for p in other_points) / len(other_points)
        
        dx = vertex.x - centroid_x
        dy = vertex.y - centroid_y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            dx, dy = dx / dist, dy / dist
        else:
            dx, dy = 0, 1
        
        label_x = vertex.x + dx * self.label_offset * 3.5
        label_y = vertex.y + dy * self.label_offset * 3.5
        
        # Use randomized font
        font_props = {
            'fontsize': self.label_fontsize,
            'fontfamily': self._current_font or 'sans-serif',
            'fontweight': random.choice(['normal', 'bold']),
        }
        
        ax.text(
            label_x, label_y, vertex.label or "",
            ha="center", va="center",
            **font_props
        )


def demo_sketchy_renderer():
    """Generate a grid of sketchy triangles to demonstrate variation."""
    from .primitives import RightTriangle, IsoscelesTriangle, Point
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    renderer = SketchyRenderer(randomize=True)
    
    for i, ax in enumerate(axes):
        # Create triangle with Point objects
        if i % 2 == 0:
            # Right triangle: A at top, B and C at base, C at right angle
            tri = RightTriangle(
                A=Point(0, 3, "A"),
                B=Point(4, 0, "B"),
                C=Point(0, 0, "C")
            )
        else:
            # Isosceles triangle
            tri = IsoscelesTriangle(
                A=Point(2, 4, "A"),
                B=Point(0, 0, "B"),
                C=Point(4, 0, "C")
            )
            
        # Render
        renderer._randomize_style()
        
        ax.set_facecolor(renderer._current_bg)
        
        A, B, C = tri.A, tri.B, tri.C
        renderer._draw_sketchy_line(ax, A.x, A.y, B.x, B.y)
        renderer._draw_sketchy_line(ax, B.x, B.y, C.x, C.y)
        renderer._draw_sketchy_line(ax, C.x, C.y, A.x, A.y)
        
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Style {i+1}", fontsize=10)
        
    plt.tight_layout()
    plt.savefig("sketchy_demo.png", dpi=150)
    print("Saved sketchy_demo.png")


if __name__ == "__main__":
    demo_sketchy_renderer()
