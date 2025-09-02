from fastapi import FastAPI
import logging
import sys

# 统一配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [APP] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 确保应用自定义日志器输出 INFO 级别并向上冒泡
paddle_logger = logging.getLogger("paddleocr_app")
paddle_logger.setLevel(logging.INFO)
paddle_logger.propagate = True

app = FastAPI(title="PaddleOCR API", description="OCR service using PaddleOCR", version="1.0.0")

# 挂载控制器路由
try:
    from app.controllers.ocr_controller import router as ocr_router
    app.include_router(ocr_router)
except Exception:
    # 避免导入阶段的循环依赖导致失败
    pass


