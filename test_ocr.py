#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for PaddleOCR functionality
"""

import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

def create_test_image():
    """Create a simple test image with text"""
    # Create a white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    try:
        # Try to use a larger font
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    draw.text((50, 50), "Hello, PaddleOCR!", fill='black', font=font)
    draw.text((50, 100), "This is a test image.", fill='black', font=font)
    draw.text((50, 150), "OCR Test 123", fill='black', font=font)
    
    # Save the image
    img.save('test_image.png')
    return 'test_image.png'

def test_paddleocr():
    """Test PaddleOCR functionality"""
    print("Initializing PaddleOCR...")
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    print("Creating test image...")
    test_image_path = create_test_image()
    
    print("Performing OCR...")
    # Perform OCR
    result = ocr.ocr(test_image_path, cls=True)
    
    print("\nOCR Results:")
    print("-" * 50)
    
    if result[0] is None:
        print("No text detected in the image.")
    else:
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            bbox = line[0]
            print(f"Text: {text}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Bounding Box: {bbox}")
            print("-" * 30)
    
    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    
    print("Test completed successfully!")

if __name__ == '__main__':
    try:
        test_paddleocr()
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Please make sure PaddleOCR is properly installed.")