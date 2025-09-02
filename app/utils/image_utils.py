import base64
import math
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def _rotate_image_keep_size(image, angle_deg):
    """根据预处理角度旋转图像，保持原尺寸（可能裁剪）"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated_image


def _rotate_image_resize(image, angle_deg):
    """根据预处理角度旋转图像，确保完整显示不裁剪"""
    if abs(angle_deg) < 0.1:  # 如果角度很小，直接返回原图
        return image

    height, width = image.shape[:2]
    angle_rad = angle_deg * math.pi / 180.0

    # 更简单可靠的边界框计算
    # 对于旋转，新的尺寸就是宽高互换
    if abs(abs(angle_deg) - 90) < 5 or abs(abs(angle_deg) - 270) < 5:
        # 90度或270度旋转，宽高互换
        print(f"DEBUG: 90度旋转 - 原始尺寸: ({height}, {width})")
        new_width = height  # 旋转90度后，高度变成宽度
        new_height = width  # 旋转90度后，宽度变成高度
        print(f"DEBUG: 90度旋转 - 新尺寸: ({new_height}, {new_width})")
    elif abs(abs(angle_deg) - 180) < 5:
        # 180度旋转，尺寸不变
        new_width = width
        new_height = height
    else:
        # 其他角度，使用几何计算
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))
        new_width = int(width * cos_a + height * sin_a)
        new_height = int(height * cos_a + width * sin_a)

    # 确保最小尺寸，避免尺寸过小
    new_width = max(new_width, 1)
    new_height = max(new_height, 1)

    # 创建一个足够大的白色画布
    print(f"DEBUG: 创建画布 - new_height={new_height}, new_width={new_width}")
    canvas = np.full((new_height, new_width, 3), 255, dtype=np.uint8)

    # 计算原图在新画布上的位置
    # 将原图放置在画布中心
    center_x = new_width // 2
    center_y = new_height // 2

    # 计算图像在画布上的偏移量
    x_offset = center_x - width // 2
    y_offset = center_y - height // 2

    # 确保偏移量不会导致图像超出画布边界
    x_offset = max(0, min(x_offset, new_width - width))
    y_offset = max(0, min(y_offset, new_height - height))

    # 如果图像比画布大，需要调整放置位置
    if width > new_width:
        x_offset = 0
    if height > new_height:
        y_offset = 0

    # 将原图复制到新画布中心
    try:
        print(f"DEBUG: 复制图像 - canvas.shape={canvas.shape}, image.shape={image.shape}")
        print(f"DEBUG: 偏移量 - x_offset={x_offset}, y_offset={y_offset}")
        print(f"DEBUG: 图像尺寸 - height={height}, width={width}")
        target_height = y_offset + height
        target_width = x_offset + width
        print(f"DEBUG: 目标范围 - y: {y_offset}:{target_height}, x: {x_offset}:{target_width}")
        print(f"DEBUG: 检查边界 - canvas_height={canvas.shape[0]}, canvas_width={canvas.shape[1]}")
        print(f"DEBUG: 目标切片 - [{y_offset}:{target_height}, {x_offset}:{target_width}]")
        canvas[y_offset:y_offset+height, x_offset:x_offset+width] = image
    except ValueError as e:
        print(f"Canvas shape: {canvas.shape}, Image shape: {image.shape}")
        print(f"x_offset: {x_offset}, y_offset: {y_offset}")
        print(f"Target slice: {y_offset}:{y_offset+height}, {x_offset}:{x_offset+width}")
        raise e

    # 以画布中心为旋转轴心进行旋转
    center = (new_width // 2, new_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 应用旋转变换
    rotated_image = cv2.warpAffine(
        canvas,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return rotated_image


def image_to_base64(image):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', rgb_image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return None


