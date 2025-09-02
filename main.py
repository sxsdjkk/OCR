#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple PaddleOCR Application
A basic OCR service using PaddleOCR for text recognition from images.
"""

import os
import cv2
import numpy as np
import math
from paddleocr import PaddleOCR, PPStructureV3
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import uvicorn
import time
import logging
from pathlib import Path
from paddleocr import PPStructureV3

app = FastAPI(title="PaddleOCR API", description="OCR service using PaddleOCR", version="1.0.0")

# 配置应用专用的日志器
logger = logging.getLogger("paddleocr_app")
logger.setLevel(logging.INFO)

# 如果还没有处理器，添加一个
if not logger.handlers:
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [APP] %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# 设置传播，让日志也显示在根日志器中
logger.propagate = True

# 测试日志器是否正常工作
logger.info("PaddleOCR 应用已初始化，日志器配置完成")

# Initialize PaddleOCR
# use_angle_cls=True to enable text direction classification
# lang='en' for English, 'ch' for Chinese, or 'ch+en' for both
simple_ocr = PaddleOCR(
    device="gpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

structure_ocr = PPStructureV3(
    device="gpu",
    use_chart_recognition=True,
) # 通过 device 指定模型推理时使用 GPU  # pyright: ignore[reportUndefinedVariable]

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

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    try:
        # 将 BGR 转换为 RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 编码为 JPEG 格式
        _, buffer = cv2.imencode('.jpg', rgb_image)
        # 转换为 base64
        base64_string = base64.b64encode(buffer).decode('utf-8')
        return base64_string
    except Exception as e:
        logger.warning(f"图片转base64失败: {e}")
        return None


def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays and other non-serializable objects to JSON-serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, '__dict__'):
        # Handle objects with attributes (like Font objects) by converting to string
        return str(obj)
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_list(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # Convert any other non-serializable object to string
        return str(obj)

def build_structured_response(extracted_text, image_width, image_height, angle=0, include_image_info=False, image_base64=None):
    """Build structured OCR response as specified by user (without returning original image)."""
    details = []
    concatenated_text = ""
    for item in extracted_text:
        value = item.get("text", "")
        concatenated_text += value
        details.append({
            # "InGraph": False,
            # "Type": "PrintedText",
            "Confidence": item.get("confidence", 0.0),
            "Position": item.get("bbox", []),
            "Value": value,
        })

    structured = {
        "OcrInfo": [
            {
                "Text": concatenated_text,
                "Detail": details,
            }
        ]
    }
    
    # 只有当 include_image_info 为 True 时才包含 ImageInfo
    if include_image_info:
        structured["ImageInfo"] = [
            {
                "Angle": angle,
                "Height": image_height,
                "Width": image_width,
                "ImageBase64": image_base64
            }
        ]
    
    return structured

def _is_quad_points(value):
    if isinstance(value, list) and len(value) == 4:
        for pt in value:
            if not (isinstance(pt, list) and len(pt) == 2):
                return False
        return True
    return False

def _ensure_quad_points(box):
    # Normalize various box formats into 4-point polygon [[x,y],...x4]
    if box is None:
        return None
    if isinstance(box, (list, tuple)):
        if len(box) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box):
            return [[int(round(pt[0])), int(round(pt[1]))] for pt in box]
        # Some formats like [x1, y1, x2, y2] rectangle
        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = box
            return [[int(round(x1)), int(round(y1))], [int(round(x2)), int(round(y1))], [int(round(x2)), int(round(y2))], [int(round(x1)), int(round(y2))]]
        # Some formats nested deeper
        if len(box) > 0 and isinstance(box[0], (list, tuple)):
            # Try take first 4 points if more
            pts = []
            for pt in box:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pts.append([int(round(pt[0])), int(round(pt[1]))])
                if len(pts) == 4:
                    break
            if len(pts) == 4:
                return pts
    return None

def _select_primary_boxes(rec_polys, rec_boxes, dt_polys):
    """选择优先使用的检测框列表，按 rec_polys > rec_boxes > dt_polys 顺序。"""
    if rec_polys and len(rec_polys) > 0:
        return rec_polys
    if rec_boxes and len(rec_boxes) > 0:
        return rec_boxes
    if dt_polys and len(dt_polys) > 0:
        return dt_polys
    return []

def _compute_rotation_angle_from_boxes(boxes, texts, scores):
    """基于候选文本框与文本，估计需要的旋转角度（度）。"""
    candidates = []
    for i, box in enumerate(boxes):
        norm_box = _ensure_quad_points(box)
        if not norm_box:
            continue
        text_val = texts[i] if i < len(texts) else ""
        score_val = float(scores[i]) if i < len(scores) and isinstance(scores[i], (int, float)) else 1.0
        candidates.append({
            "Confidence": score_val,
            "Position": norm_box,
            "Value": text_val,
        })

    if not candidates:
        return 0.0

    best = None
    best_conf = -1.0
    for c in candidates:
        # 选择更像水平正文的候选：高置信度、较长且包含空格
        if (c["Confidence"] > best_conf and len(c["Value"]) > 3 and (" " in c["Value"])):
            best = c
            best_conf = c["Confidence"]

    if best is None:
        best = candidates[0]

    position = best["Position"]
    if len(position) >= 2:
        x1, y1 = position[0]
        x2, y2 = position[1]
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        return math.degrees(angle_rad)
    return 0.0

def _rotate_points(points, center, angle_deg):
    """围绕 center 按角度（度）旋转多边形点集。"""
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx, cy = center
    rotated = []
    for x, y in points:
        tx, ty = x - cx, y - cy
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        rotated.append([int(round(rx + cx)), int(round(ry + cy))])
    return rotated

def _rotate_image_keep_size(image, angle_deg):
    """按给定角度旋转图像，保持原尺寸（可能裁剪）。"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated_image

def _parse_predict_results(predict_results):
    """解析 OCR 结果为基础字段列表。"""
    rec_texts_all, rec_scores_all, rec_polys_all, rec_boxes_all, dt_polys_all = [], [], [], [], []
    for res in predict_results:
        if hasattr(res, 'dict') and callable(getattr(res, 'dict')):
            obj = res.dict()
        elif hasattr(res, '__dict__') and res.__dict__:
            obj = dict(res.__dict__)
        else:
            obj = res if isinstance(res, dict) else {}

        try:
            if hasattr(res, '_to_json'):
                json_obj = res._to_json()
            elif hasattr(res, 'json'):
                json_obj = res.json if not callable(res.json) else res.json()
            else:
                json_obj = obj
            if isinstance(json_obj, dict) and 'res' in json_obj and isinstance(json_obj['res'], dict):
                obj = json_obj['res']
            else:
                obj = json_obj if isinstance(json_obj, dict) else obj
        except Exception:
            pass

        rec_texts = obj.get('rec_texts') or []
        rec_scores = obj.get('rec_scores') or []
        rec_polys = obj.get('rec_polys') or []
        rec_boxes = obj.get('rec_boxes') or []
        dt_polys = obj.get('dt_polys') or []

        # 拼接（不同 res 的结果合并）
        rec_texts_all.extend(rec_texts if isinstance(rec_texts, list) else [])
        rec_scores_all.extend(rec_scores if isinstance(rec_scores, list) else [])
        rec_polys_all.extend(rec_polys if isinstance(rec_polys, list) else [])
        rec_boxes_all.extend(rec_boxes if isinstance(rec_boxes, list) else [])
        dt_polys_all.extend(dt_polys if isinstance(dt_polys, list) else [])

    return rec_texts_all, rec_scores_all, rec_polys_all, rec_boxes_all, dt_polys_all

def build_items_from_predict_results(predict_results, image=None, directionCorrection=False):
    """单次 OCR：解析 → 估计角度 → 可选旋转图像与框 → 构建返回。"""
    rec_texts, rec_scores, rec_polys, rec_boxes, dt_polys = _parse_predict_results(predict_results)

    # 估计旋转角度
    primary_boxes = _select_primary_boxes(rec_polys, rec_boxes, dt_polys)
    rotation_angle = _compute_rotation_angle_from_boxes(primary_boxes, rec_texts, rec_scores)

    # 如需方向校正，则旋转图像与多边形
    if directionCorrection and image is not None and abs(rotation_angle) > 1.0:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        # 旋转图像（保持尺寸，可能裁剪）
        image[:] = _rotate_image_keep_size(image, rotation_angle)

        # 同步旋转多边形点
        def rotate_list_of_boxes(box_list):
            rotated = []
            for b in box_list:
                norm = _ensure_quad_points(b)
                if norm:
                    rotated.append(_rotate_points(norm, center, -rotation_angle))
                else:
                    rotated.append(b)
            return rotated

        rec_polys = rotate_list_of_boxes(rec_polys)
        rec_boxes = rotate_list_of_boxes(rec_boxes)
        dt_polys = rotate_list_of_boxes(dt_polys)

    # 组装结果 items
    extracted = []
    num = max(len(rec_texts), len(rec_scores), len(rec_polys or []), len(rec_boxes or []), len(dt_polys or []))
    for i in range(num):
        text_val = rec_texts[i] if i < len(rec_texts) else ""
        score_val = rec_scores[i] if i < len(rec_scores) else 1.0
        box = None
        if rec_polys and i < len(rec_polys):
            box = _ensure_quad_points(rec_polys[i])
        if box is None and rec_boxes and i < len(rec_boxes):
            box = _ensure_quad_points(rec_boxes[i])
        if box is None and dt_polys and i < len(dt_polys):
            box = _ensure_quad_points(dt_polys[i])
        if box is None:
            continue
        extracted.append({
            'text': text_val,
            'confidence': float(score_val) if isinstance(score_val, (int, float)) else 1.0,
            'bbox': box,
        })

    if extracted:
        extracted[-1]['text'] = f"{extracted[-1]['text']}\n"
    return extracted, rotation_angle

def _process_image_and_build_structure_response(image, include_image_info=False):
    result = structure_ocr.predict(image)
    items, rotation_angle = build_items_from_predict_results(result, image=image, directionCorrection=include_image_info)
    height, width = image.shape[:2]
    
    # 只有当需要包含图片信息时才转换base64
    image_base64 = None
    if include_image_info:
        image_base64 = image_to_base64(image)
    
    return build_structured_response(items, image_width=width, image_height=height, angle=-rotation_angle, include_image_info=include_image_info, image_base64=image_base64)

def _process_image_and_build_response(image, include_image_info=False):
    result = simple_ocr.predict(image)
    items, rotation_angle = build_items_from_predict_results(result, image=image, directionCorrection=include_image_info)
    height, width = image.shape[:2]
    
    # 只有当需要包含图片信息时才转换base64
    image_base64 = None
    if include_image_info:
        image_base64 = image_to_base64(image)
    
    return build_structured_response(items, image_width=width, image_height=height, angle=rotation_angle, include_image_info=include_image_info, image_base64=image_base64)

@app.get('/health', response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", service="PaddleOCR")

@app.post('/ocr_structure/file')
async def perform_ocr_structure_file(file: UploadFile = File(...)):
    """Perform OCR on uploaded image file"""
    start_time = time.time()
    try:
        # Read uploaded file
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform OCR and build response
        structured = _process_image_and_build_structure_response(image)
        elapsed = time.time() - start_time
        logging.info(f"/ocr_structure/file 耗时: {elapsed:.3f}秒")
        return JSONResponse(content=convert_numpy_to_list(structured))
        
    except Exception as e:
        elapsed = time.time() - start_time
        logging.info(f"/ocr_structure/file 异常耗时: {elapsed:.3f}秒")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post('/ocr_simple/file')
async def perform_ocr_file(request: Request, file: UploadFile = File(...)):
    """Perform OCR on uploaded image file"""
    # 读取directionCorrection query参数
    directionCorrection = request.query_params.get('directionCorrection')
    if directionCorrection is None:
        directionCorrection = False
    else:
        directionCorrection = True
    
    start_time = time.time()
    try:
        # Read uploaded file
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform OCR and build response
        structured = _process_image_and_build_response(image, include_image_info=directionCorrection)
        elapsed = time.time() - start_time
        logger.info(f"/ocr_simple/file 耗时: {elapsed:.3f}秒")
        return JSONResponse(content=convert_numpy_to_list(structured))
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.info(f"/ocr_simple/file 异常耗时: {elapsed:.3f}秒")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/ocr_simple/base64')
async def perform_ocr_base64(request: Base64ImageRequest):
    """Perform OCR on base64 encoded image"""
    start_time = time.time()
    try:
        # Handle base64 image
        image = base64_to_image(request.image_base64)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform OCR and build response
        structured = _process_image_and_build_response(image)
        elapsed = time.time() - start_time
        logging.info(f"/ocr_simple/file 耗时: {elapsed:.3f}秒")
        return JSONResponse(content=convert_numpy_to_list(structured))
        
    except Exception as e:
        elapsed = time.time() - start_time
        logging.info(f"/ocr_simple/base64 异常耗时: {elapsed:.3f}秒")
        raise HTTPException(status_code=500, detail=str(e))





if __name__ == '__main__':
    # Create data directory for storing images
    os.makedirs('data', exist_ok=True)
    
    # 测试日志配置
    print("=" * 50)
    print("应用启动中...")
    print("=" * 50)
    
    logger.info("应用启动中...")
    logger.info("日志配置已生效")
    logger.info("应用专用日志器测试")
    
    # Run the FastAPI app
    uvicorn.run(
        app, 
        host='0.0.0.0', 
        port=8008, 
        reload=True,
        log_level="info",  # 启用 uvicorn 日志
        access_log=True,   # 启用访问日志
    )