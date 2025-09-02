import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def image_to_base64(image):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', rgb_image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return None


