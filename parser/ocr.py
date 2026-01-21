"""
OCR Module for Geometry Diagram Text Extraction.

Extracts text labels from diagram images:
- Angles: "30°", "α", "θ"
- Lengths: "5", "3.5", "x"
- Vertex labels: "A", "B", "C"

Uses EasyOCR for robust text detection and recognition.
"""

import re
from pathlib import Path
from typing import Optional
import numpy as np

# Lazy import for easyocr (heavy dependency)
_reader = None


def get_ocr_reader(languages=['en']):
    """Get or create OCR reader (lazy initialization)."""
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(languages, gpu=False)
        except ImportError:
            print("Warning: easyocr not installed. Run: pip install easyocr")
            return None
    return _reader


def extract_text(image_path: str, min_confidence: float = 0.3) -> list[dict]:
    """
    Extract text from diagram image.
    
    Args:
        image_path: Path to image file
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of detected text items with bounding boxes
    """
    reader = get_ocr_reader()
    if reader is None:
        return []
    
    results = reader.readtext(str(image_path))
    
    extracted = []
    for bbox, text, confidence in results:
        if confidence < min_confidence:
            continue
            
        # Calculate center of bounding box
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        
        extracted.append({
            'text': text,
            'confidence': confidence,
            'bbox': bbox,
            'center': (center_x, center_y),
            'type': classify_text(text)
        })
        
    return extracted


def classify_text(text: str) -> str:
    """
    Classify text type based on content.
    
    Returns:
        'angle' | 'length' | 'label' | 'unknown'
    """
    text = text.strip()
    
    # Angle patterns: "30°", "30", with degree-like context
    if '°' in text or 'deg' in text.lower():
        return 'angle'
    
    # Greek letters often used for angles
    greek_letters = ['α', 'β', 'γ', 'θ', 'φ', 'alpha', 'beta', 'gamma', 'theta']
    if any(g in text.lower() for g in greek_letters):
        return 'angle'
    
    # Single uppercase letter = vertex label
    if re.match(r'^[A-Z]$', text):
        return 'label'
        
    # Number (possibly with decimal) = length
    if re.match(r'^[\d]+\.?[\d]*$', text):
        return 'length'
    
    # Variable like 'x', 'y' often represent unknown lengths
    if re.match(r'^[xyz]$', text.lower()):
        return 'length'
    
    return 'unknown'


def bind_to_geometry(text_items: list[dict], vertices: dict, image_size: int = 224) -> list[dict]:
    """
    Bind detected text to nearest geometric element.
    
    Args:
        text_items: List of detected text from extract_text()
        vertices: Dict of vertex labels to coordinates {"A": [x, y], ...}
        image_size: Image dimensions for coordinate conversion
        
    Returns:
        Enriched text items with geometric bindings
    """
    # Convert vertex coords to image space if needed
    # Assuming vertices are in geometry space [-scale, scale]
    # and image is [0, image_size]
    
    for item in text_items:
        cx, cy = item['center']
        
        # Find nearest vertex
        min_dist = float('inf')
        nearest_vertex = None
        
        for label, coords in vertices.items():
            # Convert geometry coords to approximate image coords
            # This is a rough estimate since we don't have exact camera mapping
            vx = (coords[0] / 20.0 + 0.5) * image_size
            vy = (0.5 - coords[1] / 20.0) * image_size  # Y is inverted
            
            dist = ((cx - vx) ** 2 + (cy - vy) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_vertex = label
                
        item['nearest_vertex'] = nearest_vertex
        item['distance_to_vertex'] = min_dist
        
    return text_items


def parse_value(text: str) -> Optional[float]:
    """
    Parse numeric value from text.
    
    Args:
        text: Text string like "30°", "5.5", "x"
        
    Returns:
        Float value if parseable, None otherwise
    """
    # Remove degree symbol and whitespace
    cleaned = text.replace('°', '').replace(' ', '').strip()
    
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_diagram_info(image_path: str, vertices: Optional[dict] = None) -> dict:
    """
    Full pipeline to extract text information from diagram.
    
    Args:
        image_path: Path to diagram image
        vertices: Optional dict of known vertices for binding
        
    Returns:
        Structured information dictionary
    """
    text_items = extract_text(image_path)
    
    if vertices:
        text_items = bind_to_geometry(text_items, vertices)
    
    # Organize by type
    result = {
        'angles': [],
        'lengths': [],
        'labels': [],
        'raw_items': text_items
    }
    
    for item in text_items:
        text_type = item['type']
        value = parse_value(item['text'])
        
        entry = {
            'text': item['text'],
            'value': value,
            'confidence': item['confidence'],
            'center': item['center']
        }
        
        if 'nearest_vertex' in item:
            entry['near_vertex'] = item['nearest_vertex']
            
        if text_type == 'angle':
            result['angles'].append(entry)
        elif text_type == 'length':
            result['lengths'].append(entry)
        elif text_type == 'label':
            result['labels'].append(entry)
            
    return result


if __name__ == "__main__":
    # Test on a sample image
    import sys
    
    test_path = "data/testbookpdf/templates/Screenshot 2026-01-20 at 21.22.56.png"
    
    if not Path(test_path).exists():
        # Use a synthetic sample instead
        test_path = "data/dataset/images/angle_sum_alpha_0_0.png"
        
    if Path(test_path).exists():
        print(f"Testing OCR on: {test_path}")
        result = extract_diagram_info(test_path)
        
        print(f"\nDetected Angles: {result['angles']}")
        print(f"Detected Lengths: {result['lengths']}")
        print(f"Detected Labels: {result['labels']}")
        print(f"Total items: {len(result['raw_items'])}")
    else:
        print("No test image found. Please provide an image path.")
        
    print("\n✅ OCR module loaded successfully!")
