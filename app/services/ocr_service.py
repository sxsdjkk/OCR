import math
import cv2
import json
import logging
from paddleocr import PaddleOCR, PPStructureV3
from app.utils.image_utils import image_to_base64
from app.utils.geom_utils import ensure_quad_points, rotate_points

logger = logging.getLogger("paddleocr_app")


def debug_result_object(result_obj):
    """调试 result[0] 对象的结构和内容"""
    logger.info("=" * 50)
    logger.info("DEBUG: 分析 result[0] 对象结构")
    logger.info(f"对象类型: {type(result_obj)}")

    # 检查基本属性
    if hasattr(result_obj, '__dict__'):
        attrs = [attr for attr in dir(result_obj) if not attr.startswith('_')]
        logger.info(f"可用属性: {attrs[:15] if len(attrs) > 15 else attrs}")

    # 检查是否有 _to_json 方法
    if hasattr(result_obj, '_to_json'):
        logger.info("✓ 对象有 _to_json() 方法")
        try:
            json_result = result_obj._to_json()
            logger.info(f"_to_json() 返回类型: {type(json_result)}")

            if isinstance(json_result, dict):
                logger.info(f"_to_json() 包含 {len(json_result)} 个键")
                logger.info(f"键列表: {list(json_result.keys())[:10] if len(json_result) > 10 else list(json_result.keys())}")

                # 检查前几个键的值类型
                for i, (key, value) in enumerate(json_result.items()):
                    if i >= 5:  # 只显示前5个
                        break
                    value_type = type(value)
                    if isinstance(value, (list, dict)):
                        size_info = f" (长度: {len(value)})"
                    else:
                        size_info = ""
                    logger.info(f"  {key}: {value_type}{size_info}")
            else:
                logger.info(f"_to_json() 返回: {str(json_result)[:200]}{'...' if len(str(json_result)) > 200 else ''}")

        except Exception as e:
            logger.error(f"调用 _to_json() 失败: {e}")
    else:
        logger.warning("✗ 对象没有 _to_json() 方法")

    # 检查是否有其他序列化方法
    other_methods = ['json', 'dict', 'to_dict', 'serialize']
    for method in other_methods:
        if hasattr(result_obj, method):
            logger.info(f"✓ 对象有 {method}() 方法")

    logger.info("=" * 50)


simple_ocr = PaddleOCR(
    device="gpu",
    use_angle_cls=True,
    use_doc_unwarping=False,
)

structure_ocr = PPStructureV3(
    device="gpu",
    use_chart_recognition=True,
)

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
                # 调试：打印 _to_json() 的结构信息
                logger.debug(f"_to_json() result type: {type(json_obj)}")
                if isinstance(json_obj, dict):
                    logger.debug(f"_to_json() keys: {list(json_obj.keys()) if len(json_obj) <= 10 else list(json_obj.keys())[:10] + ['...']}")
                    logger.debug(f"_to_json() total keys: {len(json_obj)}")
                    # 检查是否有嵌套的复杂结构
                    for key, value in list(json_obj.items())[:3]:  # 只检查前3个
                        logger.debug(f"  {key}: {type(value)} - {'large' if isinstance(value, (list, dict)) and len(value) > 100 else 'small'}")
            elif hasattr(res, 'json'):
                json_obj = res.json if not callable(res.json) else res.json()
            else:
                json_obj = obj
            if isinstance(json_obj, dict) and 'res' in json_obj and isinstance(json_obj['res'], dict):
                obj = json_obj['res']
            else:
                obj = json_obj if isinstance(json_obj, dict) else obj
        except Exception as e:
            logger.warning(f"解析 result 对象失败: {e}")
            # 调试：打印对象的属性
            if hasattr(res, '__dict__'):
                attrs = [attr for attr in dir(res) if not attr.startswith('_')]
                logger.debug(f"对象可用属性: {attrs[:10] if len(attrs) > 10 else attrs}")
            obj = {}

    # 如果调试模式，记录 _to_json 结果的摘要
    if hasattr(res, '_to_json'):
        try:
            json_data = res._to_json()
            logger.debug(f"result._to_json() 摘要: 类型={type(json_data)}, 键数量={len(json_data) if isinstance(json_data, dict) else 'N/A'}")
        except Exception:
            logger.debug("result._to_json() 调用失败")

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
    result = structure_ocr.predict(image)
    items, rotation_angle = build_items_from_predict_results(result, image=image, directionCorrection=direction_correction)
    h, w = image.shape[:2]
    img_b64 = image_to_base64(image) if include_image_info else None
    return build_structured_response(items, image_width=w, image_height=h, angle=rotation_angle, include_image_info=include_image_info, image_base64=img_b64)

def process_simple(image, direction_correction=False, include_image_info=False):
    result = simple_ocr.predict(image)

    # 调试：分析 result[0] 的结构
    if result and len(result) > 0:
        debug_result_object(result[0])

    items, rotation_angle = build_items_from_predict_results(result, image=image, directionCorrection=direction_correction)
    h, w = image.shape[:2]
    img_b64 = image_to_base64(image) if include_image_info else None
    return build_structured_response(items, image_width=w, image_height=h, angle=rotation_angle, include_image_info=include_image_info, image_base64=img_b64)


