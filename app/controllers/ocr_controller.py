from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time
import logging
import re

from pydantic import BaseModel

from app.services.ocr_service import process_simple
from app.utils.image_utils import base64_to_image
from app.utils.response_utils import convert_numpy_to_list


router = APIRouter()
logger = logging.getLogger("paddleocr_app")


class Base64ImageRequest(BaseModel):
    image_base64: str


def group_questions(ocr_details):
    details_sorted = sorted(ocr_details, key=lambda x: min(p[1] for p in x["Position"]))
    questions = []
    current_q = None
    for item in details_sorted:
        text = (item.get("Value") or "").strip()
        m = re.match(r"^(\d+)\.", text)
        if m:
            if current_q:
                questions.append(current_q)
            current_q = {"id": m.group(1), "texts": [text], "positions": list(item.get("Position", []))}
        else:
            if current_q:
                current_q["texts"].append(text)
                current_q["positions"] += list(item.get("Position", []))
    if current_q:
        questions.append(current_q)
    for q in questions:
        xs = [p[0] for p in q["positions"]]
        ys = [p[1] for p in q["positions"]]
        q["bbox"] = [min(xs), min(ys), max(xs), max(ys)] if xs and ys else [0, 0, 0, 0]
    return questions


@router.get('/health')
async def health_check():
    return {"status": "healthy", "service": "PaddleOCR"}


# @router.post('/ocr_structure/file')
# async def perform_ocr_structure_file(file: UploadFile = File(...)):
#     """Perform OCR (structure).

#     - file: form-data 上传的图片文件
#     - 无额外查询参数
#     """
#     start_time = time.time()
#     try:
#         contents = await file.read()
#         image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
#         if image is None:
#             raise HTTPException(status_code=400, detail="Invalid image file")
#         structured = process_structure(image)
#         elapsed = time.time() - start_time
#         logger.info(f"/ocr_structure/file 耗时: {elapsed:.3f}s")
#         return JSONResponse(content=convert_numpy_to_list(structured))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@router.post('/ocr_simple/file')
async def perform_ocr_file(
    file: UploadFile = File(...),
    directionCorrection: bool = Query(
        False,
        description='为 true 时进行方向矫正并同步旋转 polys'
    ),
    needImg: bool = Query(
        False,
        description='为 true 或 1 时，在结果中附带 ImageBase64'
    ),
):
    """Perform OCR (simple).

    - file: form-data 上传的图片文件
    - directionCorrection (query): 可选，"true"/"false"；为 true 时进行方向矫正并同步旋转 polys
    - needImg (query): 可选，"true"/"1" 返回结果中附带 ImageBase64；否则仅返回 Angle/Height/Width
    """
    include_image = bool(needImg)
    start_time = time.time()
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        structured = process_simple(image, direction_correction=directionCorrection, include_image_info=include_image)
        elapsed = time.time() - start_time
        logger.info(f"/ocr_simple/file 耗时: {elapsed:.3f}s (directionCorrection={directionCorrection}, needImg={include_image})")
        return JSONResponse(content=convert_numpy_to_list(structured))
    except Exception as e:
        logger.error(f"/ocr_simple/file 处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post('/ocr_simple/base64')
async def perform_ocr_base64(
    request: Base64ImageRequest,
    directionCorrection: bool = Query(
        False,
        description='为 true 时进行方向矫正并同步旋转 polys'
    ),
    needImg: bool = Query(
        False,
        description='为 true 或 1 时，在结果中附带 ImageBase64'
    ),
):
    """Perform OCR (simple) with base64 body.

    - body.image_base64: 必填，图片的 base64 字符串（不带 data URI 前缀）
    - 无额外查询参数
    """
    start_time = time.time()
    try:
        image = base64_to_image(request.image_base64)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        include_image = bool(needImg)
        structured = process_simple(image, direction_correction=directionCorrection, include_image_info=include_image)
        elapsed = time.time() - start_time
        logger.info(f"/ocr_simple/base64 耗时: {elapsed:.3f}s (directionCorrection={directionCorrection}, needImg={include_image})")
        return JSONResponse(content=convert_numpy_to_list(structured))
    except Exception as e:
        logger.error(f"/ocr_simple/base64 处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post('/ocr_simple/questions/file')
async def detect_questions_file(
    file: UploadFile = File(...),
    directionCorrection: bool = Query(
        False,
        description='为 true 时进行方向矫正并同步旋转 polys'
    ),
):
    """Detect questions from uploaded image after running ocr_simple pipeline."""
    start_time = time.time()
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        structured = process_simple(image, direction_correction=directionCorrection, include_image_info=False)
        details = []
        try:
            details = (structured or {}).get("OcrInfo", [{}])[0].get("Detail", [])
        except Exception:
            details = []
        questions = group_questions(details)
        elapsed = time.time() - start_time
        logger.info(f"/ocr_simple/questions/file 耗时: {elapsed:.3f}s (directionCorrection={directionCorrection})")
        return JSONResponse(content=convert_numpy_to_list({"questions": questions}))
    except Exception as e:
        logger.error(f"/ocr_simple/questions/file 处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post('/ocr_simple/questions/base64')
async def detect_questions_base64(
    request: Base64ImageRequest,
    directionCorrection: bool = Query(
        False,
        description='为 true 时进行方向矫正并同步旋转 polys'
    ),
):
    """Detect questions from base64 image after running ocr_simple pipeline."""
    start_time = time.time()
    try:
        image = base64_to_image(request.image_base64)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        structured = process_simple(image, direction_correction=directionCorrection, include_image_info=False)
        details = []
        try:
            details = (structured or {}).get("OcrInfo", [{}])[0].get("Detail", [])
        except Exception:
            details = []
        questions = group_questions(details)
        elapsed = time.time() - start_time
        logger.info(f"/ocr_simple/questions/base64 耗时: {elapsed:.3f}s (directionCorrection={directionCorrection})")
        return JSONResponse(content=convert_numpy_to_list({"questions": questions}))
    except Exception as e:
        logger.error(f"/ocr_simple/questions/base64 处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


