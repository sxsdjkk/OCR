#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple PaddleOCR Application
A basic OCR service using PaddleOCR for text recognition from images.
"""

import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import uvicorn

app = FastAPI(title="PaddleOCR API", description="OCR service using PaddleOCR", version="1.0.0")

# Initialize PaddleOCR
# use_angle_cls=True to enable text direction classification
# lang='en' for English, 'ch' for Chinese, or 'ch+en' for both
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Pydantic models for request/response
class Base64ImageRequest(BaseModel):
    image_base64: str

class OCRResponse(BaseModel):
    success: bool
    results: list
    total_text: str

class SimpleOCRResponse(BaseModel):
    text: str

class HealthResponse(BaseModel):
    status: str
    service: str

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@app.get('/health', response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", service="PaddleOCR")

@app.post('/ocr/file', response_model=OCRResponse)
async def perform_ocr_file(file: UploadFile = File(...)):
    """Perform OCR on uploaded image file"""
    try:
        # Read uploaded file
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform OCR
        result = ocr.ocr(image, cls=True)
        
        if result[0] is None:
            return OCRResponse(success=True, results=[], total_text="")
        
        # Extract text and confidence scores
        extracted_text = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            bbox = line[0]
            extracted_text.append({
                "text": text,
                "confidence": confidence,
                "bbox": bbox
            })
        
        return OCRResponse(
            success=True,
            results=extracted_text,
            total_text=" ".join([item["text"] for item in extracted_text])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/ocr/base64', response_model=OCRResponse)
async def perform_ocr_base64(request: Base64ImageRequest):
    """Perform OCR on base64 encoded image"""
    try:
        # Handle base64 image
        image = base64_to_image(request.image_base64)
        
        # Perform OCR
        result = ocr.ocr(image, cls=True)
        
        if result[0] is None:
            return OCRResponse(success=True, results=[], total_text="")
        
        # Extract text and confidence scores
        extracted_text = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            bbox = line[0]
            extracted_text.append({
                "text": text,
                "confidence": confidence,
                "bbox": bbox
            })
        
        return OCRResponse(
            success=True,
            results=extracted_text,
            total_text=" ".join([item["text"] for item in extracted_text])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/ocr/simple/file', response_model=SimpleOCRResponse)
async def simple_ocr_file(file: UploadFile = File(...)):
    """Simple OCR endpoint that returns only text from uploaded file"""
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        result = ocr.ocr(image, cls=True)
        
        if result[0] is None:
            return SimpleOCRResponse(text="")
        
        text = " ".join([line[1][0] for line in result[0]])
        return SimpleOCRResponse(text=text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/ocr/simple/base64', response_model=SimpleOCRResponse)
async def simple_ocr_base64(request: Base64ImageRequest):
    """Simple OCR endpoint that returns only text from base64 image"""
    try:
        image = base64_to_image(request.image_base64)
        
        result = ocr.ocr(image, cls=True)
        
        if result[0] is None:
            return SimpleOCRResponse(text="")
        
        text = " ".join([line[1][0] for line in result[0]])
        return SimpleOCRResponse(text=text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    # Create data directory for storing images
    os.makedirs('data', exist_ok=True)
    
    # Run the FastAPI app
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)