#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test client for FastAPI PaddleOCR service
"""

import requests
import base64
from PIL import Image, ImageDraw, ImageFont
import os

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
    
    draw.text((50, 50), "Hello, FastAPI OCR!", fill='black', font=font)
    draw.text((50, 100), "This is a test image.", fill='black', font=font)
    draw.text((50, 150), "FastAPI Test 123", fill='black', font=font)
    
    # Save the image
    img.save('test_image.png')
    return 'test_image.png'

def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def test_health_endpoint():
    """Test health check endpoint"""
    print("Testing health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_file_upload_ocr():
    """Test OCR with file upload"""
    print("Testing OCR with file upload...")
    
    # Create test image
    test_image_path = create_test_image()
    
    with open(test_image_path, 'rb') as f:
        files = {'file': ('test_image.png', f, 'image/png')}
        response = requests.post("http://localhost:8000/ocr/file", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Total text: {result['total_text']}")
        print("Detected text with confidence:")
        for item in result['results'][:3]:  # Show first 3 results
            print(f"  - Text: {item['text']}")
            print(f"    Confidence: {item['confidence']:.4f}")
    else:
        print(f"Error: {response.text}")
    
    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    
    print("-" * 50)

def test_base64_ocr():
    """Test OCR with base64 image"""
    print("Testing OCR with base64 image...")
    
    # Create test image
    test_image_path = create_test_image()
    base64_image = image_to_base64(test_image_path)
    
    payload = {"image_base64": base64_image}
    response = requests.post("http://localhost:8000/ocr/base64", json=payload)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Total text: {result['total_text']}")
    else:
        print(f"Error: {response.text}")
    
    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    
    print("-" * 50)

def test_simple_ocr():
    """Test simple OCR endpoint"""
    print("Testing simple OCR endpoint...")
    
    # Create test image
    test_image_path = create_test_image()
    
    with open(test_image_path, 'rb') as f:
        files = {'file': ('test_image.png', f, 'image/png')}
        response = requests.post("http://localhost:8000/ocr/simple/file", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Extracted text: {result['text']}")
    else:
        print(f"Error: {response.text}")
    
    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    
    print("-" * 50)

def main():
    """Run all tests"""
    print("FastAPI PaddleOCR Test Client")
    print("=" * 50)
    
    try:
        test_health_endpoint()
        test_file_upload_ocr()
        test_base64_ocr()
        test_simple_ocr()
        
        print("All tests completed!")
        print("\nAPI Documentation available at: http://localhost:8000/docs")
        print("Redoc documentation at: http://localhost:8000/redoc")
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to the API server.")
        print("Make sure the FastAPI server is running on http://localhost:8000")
        print("Run: python3 app.py")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == '__main__':
    main()