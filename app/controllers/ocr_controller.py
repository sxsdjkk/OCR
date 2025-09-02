from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time

from pydantic import BaseModel

from app.services.ocr_service import process_simple, process_structure
from app.utils.image_utils import base64_to_image
from app.utils.response_utils import convert_numpy_to_list


router = APIRouter()


class Base64ImageRequest(BaseModel):
    image_base64: str


@router.get('/health')
async def health_check():
    return {"status": "healthy", "service": "PaddleOCR"}


@router.post('/ocr_structure/file')
async def perform_ocr_structure_file(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        structured = process_structure(image)
        elapsed = time.time() - start_time
        return JSONResponse(content=convert_numpy_to_list(structured))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/ocr_simple/file')
async def perform_ocr_file(request: Request, file: UploadFile = File(...)):
    directionCorrection = bool(request.query_params.get('directionCorrection'))
    start_time = time.time()
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        structured = process_simple(image, include_image_info=directionCorrection)
        elapsed = time.time() - start_time
        return JSONResponse(content=convert_numpy_to_list(structured))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/ocr_simple/base64')
async def perform_ocr_base64(request: Base64ImageRequest):
    start_time = time.time()
    try:
        image = base64_to_image(request.image_base64)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        structured = process_simple(image)
        elapsed = time.time() - start_time
        return JSONResponse(content=convert_numpy_to_list(structured))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


