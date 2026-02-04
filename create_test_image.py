#!/usr/bin/env python3
"""
Create a simple test image for demonstration
"""

from PIL import Image, ImageDraw
import numpy as np

def create_test_image():
    """Create a simple test image."""
    # Create a 64x64 RGB image
    img = Image.new('RGB', (64, 64), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple pattern
    # Red circle
    draw.ellipse([10, 10, 30, 30], fill='red', outline='black')
    
    # Blue square
    draw.rectangle([35, 10, 55, 30], fill='blue', outline='black')
    
    # Green triangle
    draw.polygon([(20, 40), (30, 40), (25, 50)], fill='green', outline='black')
    
    # Save the image
    img.save('./test_input.png')
    print("Created test input image: test_input.png")

if __name__ == "__main__":
    create_test_image()