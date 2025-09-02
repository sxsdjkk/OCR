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
    angle_rad = abs(angle_deg) * math.pi / 180.0

    # 计算旋转后需要的画布尺寸
    # 使用更精确的边界框计算
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))

    new_width = int(width * cos_a + height * sin_a)
    new_height = int(height * cos_a + width * sin_a)

    # 创建一个足够大的白色画布
    canvas = np.full((new_height, new_width, 3), 255, dtype=np.uint8)

    # 计算原图在新画布上的位置（居中）
    x_offset = (new_width - width) // 2
    y_offset = (new_height - height) // 2

    # 将原图复制到新画布中心
    canvas[y_offset:y_offset+height, x_offset:x_offset+width] = image

    # 以新画布的中心为旋转轴心进行旋转
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


