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
    print(f"DEBUG: 开始旋转 - 原始尺寸: ({height}, {width}), 角度: {angle_deg}°")

    # 直接使用 OpenCV 旋转，自动处理边界
    center = (width // 2, height // 2)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 计算旋转后的边界框
    cos_a = abs(math.cos(math.radians(angle_deg)))
    sin_a = abs(math.sin(math.radians(angle_deg)))

    new_width = int(width * cos_a + height * sin_a)
    new_height = int(height * cos_a + width * sin_a)

    # 确保最小尺寸
    new_width = max(new_width, width)
    new_height = max(new_height, height)

    print(f"DEBUG: 计算边界框 - new_width={new_width}, new_height={new_height}")

    # 调整旋转矩阵的平移部分，以适应新的画布尺寸
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # 应用旋转变换到新尺寸的画布
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # 白色背景
    )

    print(f"DEBUG: 旋转完成 - 结果尺寸: {rotated_image.shape}")
    return rotated_image


def image_to_base64(image):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', rgb_image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return None


