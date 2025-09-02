import math
import cv2
import time
from paddleocr import PaddleOCR, PPStructureV3
from app.utils.image_utils import image_to_base64
from app.utils.geom_utils import ensure_quad_points, rotate_points


# 延迟加载的 OCR 实例
_simple_ocr = None
_structure_ocr = None

# 记录初始化时间
_simple_ocr_init_time = None
_structure_ocr_init_time = None


def get_simple_ocr():
    """延迟加载 simple OCR 实例"""
    global _simple_ocr, _simple_ocr_init_time
    if _simple_ocr is None:
        start_time = time.time()
        print("初始化 Simple OCR 模型...")
        _simple_ocr = PaddleOCR(
            device="gpu",
            use_angle_cls=True,
            use_doc_unwarping=False,
            # 移除不必要的参数以加快初始化
            show_log=False,
        )
        _simple_ocr_init_time = time.time() - start_time
        print(f"Simple OCR 模型初始化完成，耗时: {_simple_ocr_init_time:.2f}秒")
    return _simple_ocr


def get_structure_ocr():
    """延迟加载 structure OCR 实例"""
    global _structure_ocr, _structure_ocr_init_time
    if _structure_ocr is None:
        start_time = time.time()
        print("初始化 Structure OCR 模型...")
        _structure_ocr = PPStructureV3(
            device="gpu",
            use_chart_recognition=True,
            # 减少输出日志以加快初始化
            show_log=False,
        )
        _structure_ocr_init_time = time.time() - start_time
        print(f"Structure OCR 模型初始化完成，耗时: {_structure_ocr_init_time:.2f}秒")
    return _structure_ocr

def _select_primary_boxes(rec_polys, rec_boxes, dt_polys):
    if rec_polys and len(rec_polys) > 0:
        return rec_polys
    if rec_boxes and len(rec_boxes) > 0:
        return rec_boxes
    if dt_polys and len(dt_polys) > 0:
        return dt_polys
    return []

def _parse_predict_results(predict_results):
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
        except Exception as e:
            obj = {}

        rec_texts = obj.get('rec_texts') or []
        rec_scores = obj.get('rec_scores') or []
        rec_polys = obj.get('rec_polys') or []
        rec_boxes = obj.get('rec_boxes') or []
        dt_polys = obj.get('dt_polys') or []

        rec_texts_all.extend(rec_texts if isinstance(rec_texts, list) else [])
        rec_scores_all.extend(rec_scores if isinstance(rec_scores, list) else [])
        rec_polys_all.extend(rec_polys if isinstance(rec_polys, list) else [])
        rec_boxes_all.extend(rec_boxes if isinstance(rec_boxes, list) else [])
        dt_polys_all.extend(dt_polys if isinstance(dt_polys, list) else [])

    return rec_texts_all, rec_scores_all, rec_polys_all, rec_boxes_all, dt_polys_all

def _compute_rotation_angle_from_boxes(boxes, texts, scores):
    candidates = []
    for i, box in enumerate(boxes):
        norm_box = ensure_quad_points(box)
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

def _rotate_image_keep_size(image, angle_deg):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated_image

def build_items_from_predict_results(predict_results, image=None, directionCorrection=False):
    rec_texts, rec_scores, rec_polys, rec_boxes, dt_polys = _parse_predict_results(predict_results)
    primary_boxes = _select_primary_boxes(rec_polys, rec_boxes, dt_polys)
    rotation_angle = _compute_rotation_angle_from_boxes(primary_boxes, rec_texts, rec_scores)

    if directionCorrection and image is not None and abs(rotation_angle) > 1.0:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        image[:] = _rotate_image_keep_size(image, rotation_angle)

        def rotate_list_of_boxes(box_list):
            rotated = []
            for b in box_list:
                norm = ensure_quad_points(b)
                if norm:
                    rotated.append(rotate_points(norm, center, -rotation_angle))
                else:
                    rotated.append(b)
            return rotated

        rec_polys = rotate_list_of_boxes(rec_polys)
        rec_boxes = rotate_list_of_boxes(rec_boxes)
        dt_polys = rotate_list_of_boxes(dt_polys)

    extracted = []
    num = max(len(rec_texts), len(rec_scores), len(rec_polys or []), len(rec_boxes or []), len(dt_polys or []))
    for i in range(num):
        text_val = rec_texts[i] if i < len(rec_texts) else ""
        score_val = rec_scores[i] if i < len(rec_scores) else 1.0
        box = None

        if rec_polys and i < len(rec_polys):
            box = ensure_quad_points(rec_polys[i])
        if box is None and rec_boxes and i < len(rec_boxes):
            box = ensure_quad_points(rec_boxes[i])
        if box is None and dt_polys and i < len(dt_polys):
            box = ensure_quad_points(dt_polys[i])
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

def build_structured_response(extracted_text, image_width, image_height, angle=0, include_image_info=False, image_base64=None):
    details = []
    concatenated_text = ""
    for item in extracted_text:
        value = item.get("text", "")
        concatenated_text += value
        details.append({
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
        ],
        "ImageInfo": [
            {
                "Angle": -angle,
                "Height": image_height,
                "Width": image_width,
            }
        ]
    }
    # 仅当需要时才包含 ImageBase64
    if include_image_info and image_base64 is not None:
        structured["ImageInfo"][0]["ImageBase64"] = image_base64
    return structured

def process_structure(image, direction_correction=False, include_image_info=False):
    ocr = get_structure_ocr()
    result = ocr.predict(image)
    items, rotation_angle = build_items_from_predict_results(result, image=image, directionCorrection=direction_correction)
    h, w = image.shape[:2]
    img_b64 = image_to_base64(image) if include_image_info else None
    return build_structured_response(items, image_width=w, image_height=h, angle=rotation_angle, include_image_info=include_image_info, image_base64=img_b64)

def process_simple(image, direction_correction=False, include_image_info=False):
    ocr = get_simple_ocr()
    result = ocr.predict(image)

    items, rotation_angle = build_items_from_predict_results(result, image=image, directionCorrection=direction_correction)
    h, w = image.shape[:2]
    img_b64 = image_to_base64(image) if include_image_info else None
    return build_structured_response(items, image_width=w, image_height=h, angle=rotation_angle, include_image_info=include_image_info, image_base64=img_b64)


